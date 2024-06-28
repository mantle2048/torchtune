# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional

import torch

from torchtune.models.chameleon._model_utils import (
    allowed_modality_logits_process,
    disallow_begin_image_logits_process,
)
from torchtune.models.chameleon._vocab import VocabInfo
from torchtune.modules import TransformerDecoder


def multinomial_sample_one(probs: torch.Tensor) -> torch.Tensor:
    """Samples from a multinomial distribution."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None
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
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Generates the next tokens."""
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits = model(x, input_pos=input_pos)[:, -1]

    logits = allowed_modality_logits_process(logits, vocab, modality)
    if x.shape[1] >= 4096 - (1024 + 2):
        disallow_begin_image_logits_process(logits, vocab)

    return sample(logits, temperature, top_k)


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
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[torch.Tensor] = None,
    custom_generate_next_token: Optional[Callable] = None,
) -> List[List[int]]:
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt
    # convert stop tokens to tensor for easy matching
    stop_tokens = (
        torch.tensor(stop_tokens, device=prompt.device) if stop_tokens else None
    )
    bsz, prompt_length = prompt.size()
    generated_tokens = prompt.clone()
    # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
    # that already hit a stop token
    stop_token_mask = torch.ones(
        (bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device
    )

    modality = "txt"
    if custom_generate_next_token is None:
        custom_generate_next_token = generate_next_token

    # generate the first tokens conditioned on the prompt
    tokens = generate_next_token(
        model,
        input_pos=torch.arange(0, prompt_length, device=prompt.device),
        x=prompt,
        temperature=temperature,
        top_k=top_k,
        vocab=vocab,
        modality=modality,
    )
    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    # stop early if we reach a stop token in every seq
    if stop_tokens is not None:
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
        if stop_tokens is not None:
            stop_token_mask = torch.cat(
                [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
            )

        tokens = custom_generate_next_token(
            model,
            input_pos=input_pos,
            x=tokens,
            temperature=temperature,
            top_k=top_k,
            vocab=vocab,
            modality=modality,
        )

        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        input_pos += 1

        if stop_tokens is not None:
            stop_token_reached = update_stop_tokens_tracker(
                tokens, stop_tokens, stop_token_reached
            )
            if stop_token_reached.all().item():
                break

    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        generated_tokens = generated_tokens * stop_token_mask
        # if pad_id is not 0, replace 0 with pad_id
        if pad_id != 0:
            generated_tokens[generated_tokens == 0] = pad_id

    return generated_tokens.tolist()
