# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import time
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig

from torchtune import config, utils
from torchtune.models.chameleon._generation import generate as chameleon_generate
from torchtune.modules.transformer import TransformerDecoder

logger = utils.get_logger("DEBUG")


class MMInferenceRecipe:
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        _ = utils.set_seed(seed=cfg.seed)

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
            model_state_dict=ckpt_dict[utils.constants.MODEL_KEY],
        )
        self._token_manager = config.instantiate(cfg.token_manager)
        self._options = cfg.options

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: dict[str, Any],
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
            # kv_cache batch_size = 3 for instruct cfg
            model.setup_caches(batch_size=3, dtype=self._dtype)

        return model

    def convert_prompts_to_list_of_tokens(
        self,
        prompt_list: ListConfig,
    ) -> list[list[int]]:
        prompts: list[dict[str, str]] | list[list[dict[str, str]]] = list(prompt_list)
        if not isinstance(prompts[0], list):
            prompts = [prompts]
        list_of_tokens = [
            self._token_manager.tokens_from_ui(prompt) for prompt in prompts
        ]
        return list_of_tokens

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        list_of_tokens = self.convert_prompts_to_list_of_tokens(cfg.prompts)

        # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
        # to get the accurate performance measurement
        if self._quantization_mode is not None:
            logger.info("Starting compilation to improve generation performance ...")
            t0 = time.perf_counter()
            _ = chameleon_generate(
                model=self._model,
                list_of_tokens=list_of_tokens,
                max_generated_tokens=2,
                options=self._options,
                stop_token_list=self._token_manager.stop_tokens,
                vocab=self._token_manager.vocab,
                device=self._device,
            )
            t = time.perf_counter() - t0
            logger.info(f"Warmup run for quantized model takes: {t:.02f} sec")

        t0 = time.perf_counter()
        generated_tokens = chameleon_generate(
            model=self._model,
            list_of_tokens=list_of_tokens,
            max_generated_tokens=cfg.max_new_tokens,
            options=self._options,
            stop_token_list=self._token_manager.stop_tokens,
            vocab=self._token_manager.vocab,
            device=self._device,
        )
        t = time.perf_counter() - t0

        texts, images = self._token_manager.decode(generated_tokens, self._device)

        for text in texts:
            logger.info(text)

        for idx, image in enumerate(images):
            image_path = f"data/image{idx}.png"
            logger.info(f"saving image to {image_path}")
            image.save(image_path)

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        tokens_generated = len(generated_tokens[0]) - len(list_of_tokens[0])
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
    main()  # type: ignore[reportCallIssue]
