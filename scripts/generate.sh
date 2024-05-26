#!/usr/bin/env bash

CKPT_PATH="/path/to/your/model"

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
  device="cuda" \
  tokenizer.path="${CKPT_PATH}/tokenizer.model" \
  prompt="${PROMPT}" \
  max_new_tokens="512" \
  temperature="0.6" \
  top_k="300" \
  dtype="fp32"
