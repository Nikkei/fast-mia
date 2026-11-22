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
        self, output: RequestOutput, freq_dist: list[int] | np.ndarray
    ) -> float:
        """Process model output and calculate DC-PDD score

        Args:
            output: Model output
            freq_dist: Token frequency distribution over reference corpus

        Returns:
            DC-PDD score
        """
        token_log_probs = np.asarray(
            self._extract_token_log_probs(output), dtype=np.float64
        )
        if token_log_probs.size == 0:
            return 0.0

        # ``_extract_token_log_probs`` drops the leading position, whose prompt
        # logprob is None, so align the token IDs to the tail of the prompt.
        token_ids = np.asarray(output.prompt_token_ids[-token_log_probs.size :])

        # Keep the first occurrence of each token in the prompt tail, excluding
        # token IDs outside the reference vocabulary (e.g. added special tokens).
        _, indexes = np.unique(token_ids, return_index=True)
        indexes.sort()

        freq_array = np.asarray(freq_dist, dtype=np.float64)
        unique_token_ids = token_ids[indexes]
        in_range = (unique_token_ids >= 0) & (unique_token_ids < freq_array.size)
        indexes = indexes[in_range]
        if indexes.size == 0:
            return 0.0

        x_freq = freq_array[token_ids[indexes]]
        x_prob = np.exp(token_log_probs[indexes])
        eps = 1e-10
        ce = x_prob * np.log(1.0 / (x_freq + eps))
        ce[ce > self.alpha] = self.alpha
        return float(-np.mean(ce))

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

        # Get model outputs
        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )

        # Calculate scores from outputs
        return [self.process_output(output, freq_dist) for output in outputs]
