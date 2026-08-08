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

from pathlib import Path
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.tokenizers import TokenizerLike

from .base import BaseMethod
from .token_freq import freq_dist_cache_path, load_or_build_freq_dist


class DCPDDMethod(BaseMethod):
    """DC-PDD membership inference method"""

    requires_tokenizer: bool = True

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize DC-PDD method

        Args:
            method_config: Method configuration
        """
        super().__init__("dcpdd", method_config)
        # Default settings
        self.file_num = self.method_config.get("file_num", 15)
        self.max_token_length = self.method_config.get("max_token_length", 1024)
        self.alpha = self.method_config.get("alpha", 0.01)

    def _freq_dist_cache_path(self, model_id: str) -> Path:
        """Build the frequency distribution cache path for this method.

        Args:
            model_id: Model ID of the target model

        Returns:
            Path to the cache file
        """
        return freq_dist_cache_path(model_id, self.file_num, self.max_token_length)

    def process_output(
        self, output: RequestOutput, input_ids: list[int], freq_dist: list[int]
    ) -> float:
        """Process model output and calculate DC-PDD score

        Args:
            output: Model output

        Returns:
            DC-PDD score
        """
        token_log_probs = self._extract_token_log_probs(output)

        # tokens with first occurance in text
        indexes = []
        current_ids = []
        for i, input_id in enumerate(input_ids):
            if input_id not in current_ids:
                indexes.append(i)
                current_ids.append(input_id)

        x_prob = np.exp(token_log_probs)[indexes]
        x_freq = np.array(freq_dist)[np.array(input_ids)[indexes]]
        eps = 1e-10
        ce = x_prob * np.log(1 / (x_freq + eps))
        ce[ce > self.alpha] = self.alpha
        dcpdd_score = -np.mean(ce)

        return dcpdd_score

    def run(
        self,
        texts: list[str],
        model: LLM,
        tokenizer: TokenizerLike,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """DC-PDD algorithm to calculate scores for a list of texts
        Args:
            texts: List of texts
            labels: List of labels
            model: LLM model
            tokenizer: Tokenizer
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of DC-PDD scores
        """
        freq_dist = load_or_build_freq_dist(
            model,
            tokenizer,
            self._freq_dist_cache_path(self._get_model_id(model)),
            self.file_num,
            self.max_token_length,
        )

        # Apply the same normalization as get_outputs so that the token IDs
        # encoded below stay aligned with the logprobs of the inference input
        if data_config and not data_config.get("space_delimited_language", True):
            texts = [text.replace(" ", "") for text in texts]

        # Get model outputs
        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )

        # Calculate scores from outputs
        scores = [
            self.process_output(output, tokenizer.encode(text)[1:], freq_dist)
            for output, text in zip(outputs, texts, strict=True)
        ]

        return scores
