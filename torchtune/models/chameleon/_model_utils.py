import torch
from torch import Tensor

from torchtune.models.chameleon._vocab import VocabInfo

text_allow_tokens = []


def scale_hidden_dim_for_mlp(dim: int, multiple_of: int = 256) -> int:
    # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
    # parameters and computation constant
    hidden_dim = 4 * int(2 * dim / 3)
    # Round hidden dimension to nearest multiple of `multiple_of`
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def _allowed_tokens(vocab: VocabInfo, modality: str) -> list[int]:
    allowed_tokens = [vocab.eos_id]
    if modality in ["txt", "text"]:
        allowed_tokens += vocab.text_tokens
    if modality in ["img", "image"]:
        allowed_tokens += [vocab.begin_image]
    return allowed_tokens


def allowed_modality_logits_process(
    logits: Tensor,
    vocab: VocabInfo,
    modality: str,
) -> Tensor:
    token_ids = _allowed_tokens(vocab, modality)
    replacement = torch.full_like(logits, -torch.inf)
    replacement[:, token_ids] = logits[:, token_ids]
    logits[:] = replacement
    return logits


def disallow_begin_image_logits_process(logits: Tensor, vocab: VocabInfo) -> Tensor:
    disallowed_tokens = [vocab.begin_image]
    logits[:, disallowed_tokens] = -torch.inf
    return logits
