# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key

# state dict key mappings from chameleon's format to torchtune's format

_FROM_CHAMELEON = {
    "tok_embeddings.weight": "tok_embeddings.weight",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention.q_normalization.weight": "layers.{}.attn.q_normalization.weight",
    "layers.{}.attention.k_normalization.weight": "layers.{}.attn.k_normalization.weight",
    "layers.{}.attention.q_normalization.bias": "layers.{}.attn.q_normalization.bias",
    "layers.{}.attention.k_normalization.bias": "layers.{}.attn.k_normalization.bias",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
}


def chameleon_meta_to_tune(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from Meta's format to torchtune's format. State dicts
    from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.

    Eg of Meta-format state dict can be found in the ``meta-llama/Llama-2-7b``
    repo in HF (https://huggingface.co/meta-llama/Llama-2-7b).

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in Meta's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key not in ["rope.freqs"]:  # Skip loading the position embeddings
            new_key = get_mapped_key(key, _FROM_CHAMELEON)
            converted_state_dict[new_key] = value

    return converted_state_dict


def chameleon_tune_to_meta(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_CHAMELEON.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict
