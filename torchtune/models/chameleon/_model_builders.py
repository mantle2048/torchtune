from torchtune.modules import TransformerDecoder
from torchtune.models.chameleon._component_builders import chameleon

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
