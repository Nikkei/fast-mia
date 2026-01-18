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

from typing import Any

import numpy as np
from vllm.outputs import RequestOutput

from .base import BaseMethod


class MinKMethod(BaseMethod):
    """Min-K% Prob based membership inference method"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize Min-K% Prob based method

        Args:
            method_config: Method configuration
                - ratio: Ratio of lowest probability tokens to use (0.0-1.0)
        """
        # Set default ratio to 0.5
        if method_config is None:
            method_config = {"ratio": 0.5}
        elif "ratio" not in method_config:
            method_config["ratio"] = 0.5

        # Include ratio in method_name
        method_name = f"mink_{method_config['ratio']}"
        super().__init__(method_name, method_config)

    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate Min-K% score

        Args:
            output: Model output

        Returns:
            Min-K% score
        """
        token_log_probs = self._extract_token_log_probs(output)

        # Calculate mean log-likelihood of the lowest K% tokens
        ratio = self.method_config["ratio"]
        k_length = max(1, int(len(token_log_probs) * ratio))
        topk = np.sort(token_log_probs)[:k_length]  # Get lowest by ascending sort
        mink_score = np.mean(topk)

        return mink_score
