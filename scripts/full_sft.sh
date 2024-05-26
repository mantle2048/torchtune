#!/usr/bin/env bash

CKPT_PATH="/data/ckpts/TinyLlama-1.1B-Chat-v1.0"

tune run --nproc_per_node 1 full_finetune_distributed \
  --config llama2/1.1B_full \
  model._component_="torchtune.models.llama2.tinyllama" \
  checkpointer.checkpoint_dir=$CKPT_PATH \
  checkpointer.checkpoint_files=["model.safetensors"] \
  checkpointer.output_dir=$CKPT_PATH \
  device="cuda" \
  tokenizer.path="${CKPT_PATH}/tokenizer.model" \
  dtype="fp32"

