# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from omegaconf import DictConfig
from torch import Tensor

from torchtune.models.chameleon._vocab import VocabInfo
from torchtune.modules import TransformerDecoder


def allowed_modality_logits_process(
    logits: Tensor,
    vocab: VocabInfo,
    modality: str,
) -> Tensor:
    if modality == "text":
        token_ids = [vocab.eos_id] + vocab.text_tokens + [vocab.begin_image]
    elif modality == "image":
        token_ids = vocab.image_tokens
    else:
        raise ValueError(f"Disallowed Modality {modality}")

    replacement = torch.full_like(logits, -torch.inf)
    replacement[:, token_ids] = logits[:, token_ids]
    logits[:] = replacement
    return logits


def disallow_begin_image_logits_process(logits: Tensor, vocab: VocabInfo) -> Tensor:
    disallowed_tokens = [vocab.begin_image]
    logits[:, disallowed_tokens] = -torch.inf
    return logits


def split_inputs_for_cfg_v1(
    list_of_tokens: list[list[int]], vocab: VocabInfo
) -> list[list[int]]:
    batch_size = len(list_of_tokens)
    image_conditioned_allowed = set(vocab.image_tokens) | {
        vocab.bos_id,
        vocab.begin_image,
        vocab.end_image,
    }

    full_conditioned = list_of_tokens

    image_conditioned = [
        [id for id in sample if id in image_conditioned_allowed]
        for sample in list_of_tokens
    ]

    unconditioned = [[vocab.bos_id, vocab.begin_image]] * batch_size

    return full_conditioned + image_conditioned + unconditioned


def split_inputs_for_cfg(input_ids: torch.Tensor, vocab: VocabInfo) -> torch.Tensor:
    batch_size = input_ids.size(0)
    image_conditioned_allowed = set(vocab.image_tokens) | {
        vocab.bos_id,
        vocab.begin_image,
        vocab.end_image,
    }

    full_conditioned = input_ids

    image_conditioned = torch.tensor(
        [
            [id for id in sample if id.item() in image_conditioned_allowed]
            for sample in input_ids
        ],
        device=input_ids.device,
    )
    # Create unconditioned tokens tensor
    unconditioned = torch.tensor(
        [[vocab.bos_id, vocab.begin_image]] * batch_size, device=input_ids.device
    )

    max_length = max(
        full_conditioned.size(1), image_conditioned.size(1), unconditioned.size(1)
    )

    def pad_left(tensor: torch.Tensor, max_length: int):
        pad_size = max_length - tensor.size(1)
        return torch.nn.functional.pad(tensor, (pad_size, 0), "constant", vocab.pad_id)

    result = torch.cat(
        (
            pad_left(full_conditioned, max_length),
            pad_left(image_conditioned, max_length),
            pad_left(unconditioned, max_length),
        ),
        dim=0,
    )

    return result


def in_batch_instruct_cfg_logits_process(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    guidance_scale_text: int,
    guidance_scale_image: int,
) -> torch.Tensor:
    # input_ids.shape=[3*batch, seq-len]
    # logits.shape=[3*batch, vocab]
    (
        full_conditioned_logits,
        image_conditioned_logits,
        unconditioned_logits,
    ) = logits.chunk(3)

    mixed_logits = (
        unconditioned_logits
        + guidance_scale_image * (image_conditioned_logits - unconditioned_logits)
        + guidance_scale_text * (full_conditioned_logits - image_conditioned_logits)
    )
    return mixed_logits.repeat(3, 1)


def multinomial_sample_one(probs: torch.Tensor) -> torch.Tensor:
    """Samples from a multinomial distribution."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits: torch.Tensor, top_k: int | None, temperature: float = 1.0
) -> torch.Tensor:
    """Generic sample from a probability distribution."""
    # scale the logits based on temperature
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        # select the very last value from the top_k above as the pivot
        pivot = v.select(-1, -1).unsqueeze(-1)
        # set everything smaller than pivot value to inf since these
        # should be pruned
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    # change logits into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return multinomial_sample_one(probs)


def generate_next_token(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    vocab: VocabInfo,
    modality: str,
    option: DictConfig,
    image_gen_count: int = 0,
) -> torch.Tensor:
    """Generates the next tokens."""
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits = model(x, input_pos=input_pos)[:, -1]

    if modality == "image":
        logits = in_batch_instruct_cfg_logits_process(
            x, logits, option.cfg_scale_text, option.cfg_scale_text
        )
    logits = allowed_modality_logits_process(logits, vocab, modality)

    if image_gen_count == 1024:
        logits = torch.full(logits.shape, -torch.inf, device=logits.device)
        logits[:, vocab.end_image] = 0

    if x.shape[1] >= 4096 - (1024 + 2):
        logits = disallow_begin_image_logits_process(logits, vocab)

    selected_tokens = sample(logits, temperature=option.temperature, top_k=option.top_k)
    # if (bsz := selected_tokens.size(0)) > 1:
    #     selected_tokens = selected_tokens.chunk(bsz)[0]
    return selected_tokens


def update_stop_tokens_tracker(
    tokens: torch.Tensor, stop_tokens: torch.Tensor, stop_token_reached: torch.Tensor
) -> torch.Tensor:
    """Updates which sequences have reached a stop token."""
    # tokens: [bsz, 1]

    # stop_tokens: [num_stop_tokens]
    # stop_token_reached: [bsz]
    stop_token_reached_curr = torch.isin(tokens, stop_tokens).flatten()
    stop_token_reached |= stop_token_reached_curr
    return stop_token_reached


@torch.inference_mode()
def generate(
    model: TransformerDecoder,
    list_of_tokens: list[list[int]],
    *,
    max_generated_tokens: int,
    vocab: VocabInfo,
    options: DictConfig,
    stop_token_list: list[int],
    device: torch.device,
) -> list[list[int]]:
    if not options.text.enable and options.image.enable:  # image modality only
        modality = "image"
        max_generated_tokens = 1024
        for tokens in list_of_tokens:
            if tokens[-1] != vocab.begin_image:
                tokens.append(vocab.begin_image)
        list_of_tokens = split_inputs_for_cfg_v1(list_of_tokens, vocab)
    elif options.text.enable and not options.image.enable:  # text modality only
        stop_token_list += [vocab.boi_id]
        modality = "text"
    elif options.text.enable and options.image.enable:  # mix modality
        modality = "text"
    else:
        raise ValueError("Must enable at least one modality [image or text] !")

    max_length = max(len(tokens) for tokens in list_of_tokens)
    list_of_tokens = [
        ([vocab.pad_id] * (max_length - len(tokens))) + tokens
        for tokens in list_of_tokens
    ]
    prompts = torch.tensor(list_of_tokens, dtype=torch.int, device=device)
    prompts = prompts.view(1, -1) if prompts.ndim == 1 else prompts

    # convert stop tokens to tensor for easy matching
    stop_tokens = torch.tensor(stop_token_list, device=prompts.device)

    bsz, prompt_length = prompts.size()
    generated_tokens = prompts.clone()
    # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompts.device)
    # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
    # that already hit a stop token
    stop_token_mask = torch.ones(
        (bsz, prompt_length + 1), dtype=torch.int32, device=prompts.device
    )

    image_gen_count = 0

    # generate the first tokens conditioned on the prompt
    tokens = generate_next_token(
        model,
        input_pos=torch.arange(0, prompt_length, device=prompts.device),
        x=prompts,
        option=options[modality],
        image_gen_count=image_gen_count,
        vocab=vocab,
        modality=modality,
    )
    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    if tokens[0].item() in vocab.image_tokens:
        image_gen_count += 1
    if tokens[0].item() == vocab.begin_image:
        modality = "image"
    if tokens[0].item() == vocab.end_image:
        image_gen_count = 0
        modality = "text"

    # stop early if we reach a stop token in every seq
    stop_token_reached = update_stop_tokens_tracker(
        tokens, stop_tokens, stop_token_reached
    )
    if stop_token_reached.all().item():
        return generated_tokens.tolist()

    input_pos = torch.tensor([prompt_length], device=prompts.device)
    for _ in range(max_generated_tokens - 1):
        # update stop_token_mask if we reached a stop token in a previous step
        # by appending the logical not of stop_token_reached to the end of the mask
        # reshaped to be bsz first

        stop_token_mask = torch.cat(
            [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
        )

        tokens = generate_next_token(
            model,
            input_pos=input_pos,
            x=tokens,
            option=options[modality],
            image_gen_count=image_gen_count,
            vocab=vocab,
            modality=modality,
        )

        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        input_pos += 1

        if modality == "image":
            image_gen_count += 1
        if tokens[0].item() == vocab.begin_image:
            modality = "image"
        if tokens[0].item() == vocab.end_image:
            image_gen_count = 0
            modality = "text"

        stop_token_reached = update_stop_tokens_tracker(
            tokens,
            stop_tokens,
            stop_token_reached,
        )
        if stop_token_reached.all().item():
            break

    # mask out generated tokens in seqs that already hit a stop token
    generated_tokens = generated_tokens * stop_token_mask
    # if pad_id is not 0, replace 0 with pad_id
    if vocab.pad_id != 0:
        generated_tokens[generated_tokens == 0] = vocab.pad_id

    return generated_tokens.tolist()
