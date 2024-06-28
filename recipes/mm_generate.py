# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig

from torchtune import config, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message
from torchtune.models.chameleon._generation import generate as chameleon_generate
from torchtune.modules.transformer import TransformerDecoder

logger = utils.get_logger("DEBUG")


class MMInferenceRecipe:
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._token_manager = config.instantiate(cfg.token_manager)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> TransformerDecoder:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(batch_size=1, dtype=self._dtype)

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Union[DictConfig, ListConfig, str],
        chat_format_name: Optional[str],  # ChatFormat
        instruct_template_name: Optional[str],  # InstructTemplate
    ) -> List[int]:
        # Should only be chat-style prompt or instruct-style prompt
        if chat_format_name and instruct_template_name:
            raise ValueError(
                "Cannot pass both chat format and instruct template for generation"
            )

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template_name:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template_name)
            assert isinstance(instruct_template, InstructTemplate)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self._token_manager.tokenize_text(
                prompt, add_bos=True, add_eos=False
            )

        # dict.items() will respect order for Python >= 3.7
        elif isinstance(prompt, DictConfig):
            messages = [Message(role=k, content=v) for k, v in prompt.items()]  # type: ignore[reportArgumentType]
            messages += [Message(role="assistant", content="")]
            if chat_format_name:
                chat_format = _get_component_from_path(chat_format_name)
                assert isinstance(chat_format, ChatFormat)
                messages = chat_format.format(messages)
            return self._token_manager.tokenize_messages(messages)[0]

        else:
            return self._token_manager.tokens_from_ui(prompt)

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        tokens = self.convert_prompt_to_tokens(
            cfg.prompt, cfg.get("chat_format", None), cfg.get("instruct_template", None)
        )
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        custom_generate_next_token = None

        # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
        # to get the accurate performance measurement
        if self._quantization_mode is not None:
            logger.info("Starting compilation to improve generation performance ...")
            custom_generate_next_token = torch.compile(
                custom_generate_next_token, mode="max-autotune", fullgraph=True
            )
            t0 = time.perf_counter()
            _ = chameleon_generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=2,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._token_manager.stop_tokens,
                pad_id=self._token_manager.vocab.pad_id,
                vocab=self._token_manager.vocab,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0
            logger.info(f"Warmup run for quantized model takes: {t:.02f} sec")

        t0 = time.perf_counter()
        generated_tokens = chameleon_generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=self._token_manager.stop_tokens,
            pad_id=self._token_manager.vocab.pad_id,
            vocab=self._token_manager.vocab,
            custom_generate_next_token=custom_generate_next_token,
        )
        t = time.perf_counter() - t0

        logger.info(self._token_manager.decode_text(generated_tokens))

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        tokens_generated = len(generated_tokens[0]) - prompt.size(0)
        tokens_sec = tokens_generated / t
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="MMInferenceRecipe", cfg=cfg)
    recipe = MMInferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())  # type: ignore
