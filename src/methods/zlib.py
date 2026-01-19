# Copyright (c) 2025 Nikkei Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import zlib
from typing import Any

import numpy as np
from vllm.outputs import RequestOutput

from .base import BaseMethod


class ZlibMethod(BaseMethod):
    """Zlib compression-based membership inference method"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize Zlib compression-based method

        Args:
            method_config: Method configuration
        """
        super().__init__("zlib", method_config)

    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate zlib-compressed information content ratio

        Args:
            output: Model output

        Returns:
            zlib-compressed information content ratio
        """
        token_log_probs = self._extract_token_log_probs(output)
        ll = np.mean(token_log_probs)

        # Calculate ratio to information content compressed by zlib
        compressed_size = len(zlib.compress(bytes(output.prompt, "utf-8")))
        normalized_ll = ll / compressed_size

        return normalized_ll
