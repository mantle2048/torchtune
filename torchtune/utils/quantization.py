# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional

import torch
from torchao.quantization.quant_api import (
    Int4WeightOnlyGPTQQuantizer,
    Int4WeightOnlyQuantizer,
    quantize,
    Quantizer,
)

# importing TORCH_VERSION_AFTER_2_3 because `Int8DynActInt4WeightQuantizer`
# is only available after 2.3 so we have to guard the pytorch versions to decide
# the list of supported quantizers
from torchao.utils import TORCH_VERSION_AFTER_2_3

__all__ = [
    "Int4WeightOnlyQuantizer",
    "Int4WeightOnlyGPTQQuantizer",
    "Int8WeightOnlyQuantizer",
    "get_quantizer_mode",
]


class Int8WeightOnlyQuantizer(Quantizer):
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        return quantize(model, "int8_weight_only")
        return model


_quantizer_to_mode = {
    Int4WeightOnlyQuantizer: "4w",
    Int8WeightOnlyQuantizer: "8w",
    Int4WeightOnlyGPTQQuantizer: "4w-gptq",
}


if TORCH_VERSION_AFTER_2_3:
    from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

    __all__.append("Int8DynActInt4WeightQuantizer")
    _quantizer_to_mode[Int8DynActInt4WeightQuantizer] = "8da4w"


def get_quantizer_mode(quantizer: Optional[Callable]) -> Optional[str]:
    """Given a quantizer object, returns a string that specifies the type of quantization.

    For example, in the case of int4 weight only quantization, we'll return "4w".
    If the quantizer is not recognized as a known quantizer, we'll return None

    Args:
        quantizer (Optional[Callable]): A callable object that implements the `quantize` method.

    Returns:
        Optional[str]: The quantization mode.
    """
    return _quantizer_to_mode.get(type(quantizer), None)
