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

import logging
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .base import BaseMethod


class RefMethod(BaseMethod):
    """Reference model based membership inference method"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize Reference model based method
        Args:
            method_config: Method configuration
        """
        super().__init__("ref", method_config)

        self.ref_model_config = self.method_config.get("reference_model")
        if not self.ref_model_config:
            raise ValueError("Ref method requires a 'reference_model' configuration in params.")

        self.ref_model_id = self.ref_model_config.get("model_id")
        if not self.ref_model_id:
            raise ValueError("reference_model configuration must include 'model_id'.")

        self.ref_model = None

    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate loss

        Args:
            output: Model output

        Returns:
            Loss
        """
        token_log_probs = self._extract_token_log_probs(output)

        # Calculate loss (lower for data included in training data)
        loss = np.mean(token_log_probs)
        return loss

    def run(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """Ref algorithm to calculate scores for a list of texts
        
        Args:
            texts: List of texts
            model: LLM model (target)
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration
        Returns:
            List of Ref scores
        """
        # Lazy load reference model
        if self.ref_model is None:
            llm_kwargs = {k: v for k, v in self.ref_model_config.items() if k != "model_id"}
            logging.info(f"Loading reference model for Ref: {self.ref_model_id}")
            self.ref_model = LLM(model=self.ref_model_id, **llm_kwargs)

        # Get target model outputs
        target_outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )

        # Get reference model outputs
        # Note: We do not pass lora_request to reference model typically.
        ref_outputs = self.get_outputs(
            texts, self.ref_model, sampling_params, None, data_config
        )

        target_losses = [self.process_output(target_output) for target_output in target_outputs]
        ref_losses = [self.process_output(ref_output) for ref_output in ref_outputs]

        scores = [
            target_loss - ref_loss for target_loss, ref_loss in zip(target_losses, ref_losses, strict=True)
        ]

        return scores