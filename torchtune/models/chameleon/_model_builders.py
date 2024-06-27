from torchtune.models.chameleon._component_builders import chameleon
from torchtune.models.chameleon._token_manager import ChameleonTokenManager
from torchtune.modules import TransformerDecoder


def chameleon_7b() -> TransformerDecoder:
    return chameleon(
        vocab_size=65_536,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=4096,
        intermediate_dim=11_008,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def chameleon_token_manager(
    tokenizer_path: str,
    vqgan_cfg_path: str,
    vqgan_ckpt_path: str,
    device: str | None = None,
) -> ChameleonTokenManager:
    token_manager = ChameleonTokenManager(
        tokenizer_path,
        vqgan_cfg_path,
        vqgan_ckpt_path,
        device,
    )
    return token_manager
