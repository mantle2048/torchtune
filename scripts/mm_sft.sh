#!/usr/bin/env bash

CKPT_PATH="/nas/shared/GAIR/ckpts/Anole-7b-v0.1/models/7b"

TOKENIZER_PATH="/nas/shared/GAIR/ckpts/Anole-7b-v0.1/tokenizer"

tune run --nproc_per_node 2 mm_full_finetune_distributed \
  --config "chameleon/7B_full" \
  model._component_="torchtune.models.chameleon.chameleon_7b" \
  checkpointer.checkpoint_dir=$CKPT_PATH \
  checkpointer.checkpoint_files=["consolidated.pth"] \
  checkpointer.output_dir=$CKPT_PATH \
  device="cuda" \
  token_manager._component_="torchtune.models.chameleon.chameleon_token_manager" \
  token_manager.tokenizer_path=$TOKENIZER_PATH"/text_tokenizer.json" \
  token_manager.vqgan_cfg_path=$TOKENIZER_PATH"/vqgan.yaml" \
  token_manager.vqgan_ckpt_path=$TOKENIZER_PATH"/vqgan.ckpt" \
  dtype="bf16" \
  epochs=1 \
  output_dir="/nas/shared/GAIR/ckpts/Anole-7b-v0.1/models/7b-dalle-sft"
