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


class WBCMethod(BaseMethod):
    """Window-Based Comparison (WBC) membership inference method"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize WBC method

        Args:
            method_config: Method configuration
        """
        super().__init__("wbc", method_config)
        # Default settings
        self.window_sizes = self.method_config.get("window_sizes", [1, 5, 10, 20])
        
        self.ref_model_config = self.method_config.get("reference_model")
        if not self.ref_model_config:
            raise ValueError("WBC method requires a 'reference_model' configuration in params.")
            
        self.ref_model_id = self.ref_model_config.get("model_id")
        if not self.ref_model_id:
            raise ValueError("reference_model configuration must include 'model_id'.")

        self.ref_model = None

    def process_output(self, output: RequestOutput) -> float:
        """Not supported for single output."""
        raise NotImplementedError(
            "process_output is not supported in WBCMethod. Use run method instead."
        )

    def _compute_window_score(
        self, target_losses: np.ndarray, ref_losses: np.ndarray, window_size: int
    ) -> float:
        min_length = min(len(target_losses), len(ref_losses))

        if min_length == 0:
            return 0.0

        effective_window_size = min(window_size, min_length)

        target_trimmed = target_losses[:min_length]
        ref_trimmed = ref_losses[:min_length]

        kernel = np.ones(effective_window_size)
        target_sums = np.convolve(target_trimmed, kernel, mode="valid")
        ref_sums = np.convolve(ref_trimmed, kernel, mode="valid")

        return float(np.mean(ref_sums > target_sums))

    def run(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """Run WBC algorithm to calculate scores for a list of texts
        
        Args:
            texts: List of texts
            model: LLM model (target)
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of WBC scores
        """
        # Lazy load reference model
        if self.ref_model is None:
            llm_kwargs = {k: v for k, v in self.ref_model_config.items() if k != "model_id"}
            logging.info(f"Loading reference model for WBC: {self.ref_model_id}")
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
        
        scores = []
        for target_output, ref_output in zip(target_outputs, ref_outputs, strict=True):
            target_logprobs = self._extract_token_log_probs(target_output)
            ref_logprobs = self._extract_token_log_probs(ref_output)
            
            target_losses = -np.array(target_logprobs)
            ref_losses = -np.array(ref_logprobs)
            
            window_scores = [
                self._compute_window_score(target_losses, ref_losses, window_size)
                for window_size in self.window_sizes
            ]
            
            scores.append(float(np.mean(window_scores)) if window_scores else 0.0)
            
        return scores
