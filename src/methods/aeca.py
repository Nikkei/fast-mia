# Copyright (c) 2026 Nikkei Inc.

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


class AECAMethod(BaseMethod):
    """AECA (Adaptive Entropic Convolutional Analysis) membership inference method

    The reference implementation scales logits by a temperature of 1.5 before
    the softmax. vLLM only exposes the logprob of the prompt token, not
    full-vocabulary logits, so the partition function cannot be recomputed at
    another temperature and the streams below are always evaluated at T = 1.0.
    Setting ``sampling_parameters.temperature`` does not change this: prompt
    logprobs are computed from raw logits (vLLM defaults to
    ``logprobs_mode='raw_logprobs'``), and temperature only affects sampling of
    generated tokens. Since ``sigma(S)`` and ``sigma(L)`` are scaled
    differently at T = 1, the best ``lambda_coef`` differs from the value the
    paper reports.
    """

    requires_tokenizer: bool = True

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize AECA method

        Args:
            method_config: Method configuration

        Raises:
            ValueError: If 'alpha' is not positive
        """
        super().__init__("aeca", method_config)
        # Default settings
        self.lambda_coef = float(self.method_config.get("lambda_coef", 1.0))
        self.alpha = float(self.method_config.get("alpha", 1.0))
        self.file_num = self.method_config.get("file_num", 15)
        self.max_token_length = self.method_config.get("max_token_length", 1024)
        self.default_i_ref = float(self.method_config.get("default_i_ref", 0.5))

        if self.alpha <= 0:
            raise ValueError("AECAMethod requires a positive 'alpha' in params.")

    def _freq_dist_cache_path(self, model_id: str) -> Path:
        """Build the frequency distribution cache path.

        AECA counts C4 tokens exactly like DC-PDD does, so both methods share
        the same cache file and the ~30 minute count runs only once.

        Args:
            model_id: Model ID of the target model

        Returns:
            Path to the cache file
        """
        return freq_dist_cache_path(model_id, self.file_num, self.max_token_length)

    def _build_i_ref_table(self, freq_dist: list[int]) -> np.ndarray:
        """Convert raw token counts into self-information values

        ``P_ref(x) = (C(x) + alpha) / (N + alpha * |V|)`` with Laplace
        smoothing, and ``I_self(x) = -log P_ref(x)``.

        Args:
            freq_dist: Token counts over the reference corpus

        Returns:
            Self-information value per token ID
        """
        counts = np.asarray(freq_dist, dtype=np.float64)
        total = counts.sum() + self.alpha * counts.size
        return -(np.log(counts + self.alpha) - np.log(total))

    def process_output(self, output: RequestOutput, i_ref_table: np.ndarray) -> float:
        """Process model output and calculate AECA score

        Args:
            output: Model output
            i_ref_table: Self-information value per token ID

        Returns:
            AECA score
        """
        token_log_probs = np.asarray(
            self._extract_token_log_probs(output), dtype=np.float64
        )
        if token_log_probs.size == 0:
            return 0.0

        # ``_extract_token_log_probs`` drops the leading position, whose prompt
        # logprob is None, so align the token IDs to the tail of the prompt.
        token_ids = np.asarray(output.prompt_token_ids[-token_log_probs.size :])

        # Self-information of each token, falling back to a constant for IDs
        # outside the reference vocabulary (e.g. added special tokens).
        i_ref = np.full(token_ids.shape, self.default_i_ref, dtype=np.float64)
        in_range = token_ids < i_ref_table.size
        i_ref[in_range] = i_ref_table[token_ids[in_range]]

        # Information potential stream, then the high-pass difference filter
        # S[t] = Phi(t) - Phi(t + 1), with Phi(T + 1) treated as zero.
        potential = np.exp(token_log_probs) * i_ref
        spectrum = potential.copy()
        spectrum[:-1] = potential[:-1] - potential[1:]

        nll = -token_log_probs
        return float(np.std(spectrum) - self.lambda_coef * np.std(nll))

    def run(
        self,
        texts: list[str],
        model: LLM,
        tokenizer: TokenizerLike,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """AECA algorithm to calculate scores for a list of texts

        Args:
            texts: List of texts
            model: LLM model
            tokenizer: Tokenizer
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of AECA scores
        """
        freq_dist = load_or_build_freq_dist(
            model,
            tokenizer,
            self._freq_dist_cache_path(self._get_model_id(model)),
            self.file_num,
            self.max_token_length,
        )
        i_ref_table = self._build_i_ref_table(freq_dist)

        # Get model outputs
        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )

        # Calculate scores from outputs
        return [self.process_output(output, i_ref_table) for output in outputs]
