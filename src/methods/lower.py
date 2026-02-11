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
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .base import BaseMethod


class LowerMethod(BaseMethod):
    """Lower based membership inference method"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize Lower based method

        Args:
            method_config: Method configuration
        """
        super().__init__("lower", method_config)

    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate loss

        Args:
            output: Model output

        Returns:
            Negative mean log-likelihood (loss)
        """
        token_log_probs = self._extract_token_log_probs(output)

        loss = -np.mean(token_log_probs)
        return loss

    def run(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """Run Lower algorithm and calculate scores for a list of texts

        Args:
            texts: List of texts
            model: LLM model
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of scores
        """
        # Get model outputs
        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )

        lower_texts = [text.lower() for text in texts]
        lower_outputs = self.get_outputs(
            lower_texts, model, sampling_params, lora_request, data_config
        )

        # Calculate scores from outputs
        losses = [self.process_output(output) for output in outputs]
        lower_losses = [
            self.process_output(lower_output) for lower_output in lower_outputs
        ]

        eps = 1e-10
        scores = [
            lower_loss / (loss + eps)
            for lower_loss, loss in zip(lower_losses, losses, strict=True)
        ]

        return scores
