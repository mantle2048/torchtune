import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from rich import print
from transformers.generation import GenerationConfig
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer


def main(ckpt_path: Path):
    device = "cpu"
    model_type = torch.bfloat16
    model = LlamaForCausalLM.from_pretrained(
        ckpt_path, torch_dtype=model_type, device_map=device
    )
    tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
    generation_config = GenerationConfig(
        max_new_tokens=128,
        num_beams=1,
        do_sample=True,
        use_cache=True,
        temperature=0.6,
    )
    assert isinstance(model, LlamaForCausalLM)

    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate.",
        },
        {"role": "user", "content": "Please tell me a short story."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, generation_config=generation_config)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(completion)


if __name__ == "__main__":
    load_dotenv(override=True)
    ckpt_path = Path(os.getenv("CKPT_PATH", ""))
    model_name = "TinyLlama-1.1B-Chat-v1.0"
    main(ckpt_path / model_name)
