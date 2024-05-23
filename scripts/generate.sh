#!/usr/bin/env bash

CKPT_PATH="/data2/ckpts/TinyLlama-1.1B-Chat-v1.0"

QUESTION="Please tell me a short story."

PROMPT="<|system|>

You are a friendly chatbot who always responds in the style of a pirate.</s>

<|user|>

${QUESTION}</s>

<|assistant|>
"

tune run generate \
  --config generation \
  model._component_="torchtune.models.llama2.tinyllama" \
  checkpointer.checkpoint_dir=$CKPT_PATH \
  checkpointer.checkpoint_files=["model.safetensors"] \
  checkpointer.output_dir=$CKPT_PATH \
  device="cpu" \
  tokenizer.path="${CKPT_PATH}/tokenizer.model" \
  prompt="${PROMPT}" \
  max_new_tokens="64" \
  temperature="0.6" \
  top_k="300"
