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


class LossMethod(BaseMethod):
    """Loss (log-likelihood) based membership inference method"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize Loss based method

        Args:
            method_config: Method configuration
        """
        super().__init__("loss", method_config)

    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate loss

        Args:
            output: Model output

        Returns:
            Loss
        """
        token_log_probs = []
        for prompt_logprob in output.prompt_logprobs:
            if prompt_logprob is None:
                continue
            token_log_probs.append(list(prompt_logprob.values())[0].logprob)

        # Calculate loss (lower for data included in training data)
        loss = np.mean(token_log_probs)
        return loss
