#!/usr/bin/env bash

CKPT_PATH="/nas/shared/GAIR/ckpts/Anole-7b-v0.1/models/7b"

TOKENIZER_PATH="/nas/shared/GAIR/ckpts/Anole-7b-v0.1/tokenizer"


# PROMPT='[
#     {"type": "image", "value": "file:./data/red_flower.png"},
#     {"type": "text", "value": "Tell me the color of the flower?"},
# ]'

PROMPT='[
    {"type": "text", "value": "Draw a red flower."},
]'
    # {"type": "sentinel", "value": "<END-OF-TURN>"}
# PROMPT="Hi, how are you today?"
tune run mm_generate \
  --config "chameleon/generation" \
  model._component_="torchtune.models.chameleon.chameleon_7b" \
  checkpointer.checkpoint_dir=$CKPT_PATH \
  checkpointer.checkpoint_files=["consolidated.pth"] \
  checkpointer.output_dir=$CKPT_PATH \
  device="cuda" \
  token_manager._component_="torchtune.models.chameleon.chameleon_token_manager" \
  token_manager.tokenizer_path=$TOKENIZER_PATH"/text_tokenizer.json" \
  token_manager.vqgan_cfg_path=$TOKENIZER_PATH"/vqgan.yaml" \
  token_manager.vqgan_ckpt_path=$TOKENIZER_PATH"/vqgan.ckpt" \
  options.image.enable="true" \
  options.text.enable="false" \
  prompts="${PROMPT}" \
  max_new_tokens="2048" \
  dtype="bf16"
