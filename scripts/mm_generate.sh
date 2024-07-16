#!/usr/bin/env bash

CKPT_PATH="/nas/shared/GAIR/ckpts/Anole-7b-v0.1/models/7b"

TOKENIZER_PATH="/nas/shared/GAIR/ckpts/Anole-7b-v0.1/tokenizer"

PROMPT="[
    {'type': 'text', 'value': 'Tell me the color of the flower?'},
    {'type': 'image', 'value': 'file:./data/red_flower.png'},
    {'type': 'sentinel', 'value': '<END-OF-TURN>'}
]"

PROMPT="[
  {'type': 'text', 'value': 'A vibrant coral reef teeming with colorful fish, sea turtles gliding through the water, and sunlight filtering down from the surface.'},
  {'type': 'sentinel', 'value': '<END-OF-TURN>'}
]"

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
  options.text.enable="true" \
  prompts="${PROMPT}" \
  max_new_tokens="2048" \
  dtype="bf16"
