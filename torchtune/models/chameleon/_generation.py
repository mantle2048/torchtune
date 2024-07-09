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
    top_k: int | None,
    image_gen_count: int = 0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generates the next tokens."""
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits = model(x, input_pos=input_pos)[:, -1]

    if image_gen_count == 1024:
        logits = torch.full(logits.shape, -torch.inf, device=logits.device)
        logits[:, vocab.end_image] = 0

    logits = allowed_modality_logits_process(logits, vocab, modality)
    if x.shape[1] >= 4096 - (1024 + 2):
        logits = disallow_begin_image_logits_process(logits, vocab)

    return sample(logits, temperature=temperature, top_k=top_k)


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
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    vocab: VocabInfo,
    options: DictConfig,
    pad_id: int,
    stop_tokens: torch.Tensor,
) -> list[list[int]]:
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt
    # convert stop tokens to tensor for easy matching
    stop_tokens = torch.tensor(stop_tokens, device=prompt.device)
    bsz, prompt_length = prompt.size()
    generated_tokens = prompt.clone()
    # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
    # that already hit a stop token
    stop_token_mask = torch.ones(
        (bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device
    )

    if not options.text.enable:
        modality = "image"
    else:
        modality = "text"

    image_gen_count = 0

    # generate the first tokens conditioned on the prompt
    tokens = generate_next_token(
        model,
        input_pos=torch.arange(0, prompt_length, device=prompt.device),
        x=prompt,
        temperature=options[modality]["temperature"],
        top_k=options[modality]["top_k"],
        image_gen_count=image_gen_count,
        vocab=vocab,
        modality=modality,
    )
    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    if tokens.item() in vocab.image_tokens:
        image_gen_count += 1
    elif tokens.item() == vocab.begin_image:
        modality = "image"
    elif tokens.item() == vocab.end_image:
        image_gen_count = 0
        modality = "text"

    # stop early if we reach a stop token in every seq
    stop_token_reached = update_stop_tokens_tracker(
        tokens, stop_tokens, stop_token_reached
    )
    if stop_token_reached.all().item():
        return generated_tokens.tolist()

    input_pos = torch.tensor([prompt_length], device=prompt.device)
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
            temperature=options[modality]["temperature"],
            top_k=options[modality]["top_k"],
            image_gen_count=image_gen_count,
            vocab=vocab,
            modality=modality,
        )

        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        input_pos += 1

        if tokens.item() in vocab.image_tokens:
            image_gen_count += 1
        elif tokens.item() == vocab.begin_image:
            modality = "image"
        elif tokens.item() == vocab.end_image:
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
    if pad_id != 0:
        generated_tokens[generated_tokens == 0] = pad_id

    return generated_tokens.tolist()
