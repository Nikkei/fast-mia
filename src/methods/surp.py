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

from typing import Any

import numpy as np
from vllm.outputs import RequestOutput

from .base import BaseMethod


class SURPMethod(BaseMethod):
    """SURP (Surprising Token Selection) membership inference method.

    Selects low-probability tokens specifically at low-uncertainty (low-entropy)
    positions, filtering out genuinely ambiguous positions where even training
    members incur low token probabilities (Zhang & Wu, 2024; arXiv:2407.21248).

    Supports zero-extra-inference operation under both ``prompt_logprobs: 0``
    (estimating local contextual uncertainty from a causal sliding window of
    preceding token NLLs) and ``prompt_logprobs > 0`` (computing truncated
    Shannon entropy directly from top-K candidate logprobs).
    """

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize SURP method.

        Args:
            method_config: Method configuration
                - ratio: Fraction of lowest log-prob tokens selected among
                  low-entropy positions, in (0.0, 1.0] (default 0.2)
                - entropy_percentile: Percentile cutoff in (0.0, 100.0] used to
                  identify low-entropy positions when entropy_threshold is None
                  (default 50.0)
                - entropy_threshold: Optional explicit upper bound on position
                  entropy (default None)
                - window_size: Causal prefix window size for local entropy-rate
                  estimation when top-K prompt logprobs are unavailable
                  (default 5)

        Raises:
            ValueError: If parameters are out of range
        """
        config = dict(method_config) if method_config is not None else {}
        ratio = float(config.get("ratio", 0.2))
        entropy_percentile = float(config.get("entropy_percentile", 50.0))
        raw_threshold = config.get("entropy_threshold")
        entropy_threshold = float(raw_threshold) if raw_threshold is not None else None
        window_size = int(config.get("window_size", 5))

        if not (0.0 < ratio <= 1.0):
            raise ValueError("SURPMethod requires 'ratio' in (0.0, 1.0].")
        if not (0.0 < entropy_percentile <= 100.0):
            raise ValueError(
                "SURPMethod requires 'entropy_percentile' in (0.0, 100.0]."
            )
        if entropy_threshold is not None and entropy_threshold < 0.0:
            raise ValueError("SURPMethod requires 'entropy_threshold' >= 0.0.")
        if window_size < 1:
            raise ValueError("SURPMethod requires 'window_size' >= 1.")

        super().__init__("surp", config)
        self.ratio = ratio
        self.entropy_percentile = entropy_percentile
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size

    @staticmethod
    def _topk_shannon_entropy(prompt_logprob: dict[int, Any]) -> float:
        """Estimate position Shannon entropy from top-K prompt logprob entries."""
        cand_log_probs = np.asarray(
            [entry.logprob for entry in prompt_logprob.values()], dtype=np.float64
        )
        probs = np.exp(np.clip(cand_log_probs, -100.0, 0.0))
        prob_sum = float(np.sum(probs))
        if prob_sum <= 0.0:
            return 0.0

        if prob_sum >= 1.0:
            norm_probs = probs / prob_sum
            return float(-np.sum(norm_probs * np.log(np.maximum(norm_probs, 1e-12))))

        entropy = float(-np.sum(probs * cand_log_probs))
        residual = 1.0 - prob_sum
        if residual > 1e-12:
            entropy -= float(residual * np.log(residual))
        return max(0.0, entropy)

    def _causal_window_uncertainty(self, nll: np.ndarray) -> np.ndarray:
        """Estimate local entropy rate from preceding window NLLs."""
        num_tokens = nll.size
        entropies = np.empty(num_tokens, dtype=np.float64)
        for idx in range(num_tokens):
            if idx == 0:
                entropies[0] = nll[0]
            else:
                start = max(0, idx - self.window_size)
                entropies[idx] = float(np.mean(nll[start:idx]))
        return entropies

    def _extract_log_probs_and_entropies(
        self, output: RequestOutput
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract aligned token logprobs and per-position entropy estimates."""
        token_log_probs: list[float] = []
        topk_entropies: list[float] = []
        has_topk = False

        for token_id, prompt_logprob in zip(
            output.prompt_token_ids, output.prompt_logprobs, strict=True
        ):
            if prompt_logprob is None:
                continue
            token_log_probs.append(prompt_logprob[token_id].logprob)
            if len(prompt_logprob) > 1:
                has_topk = True
                topk_entropies.append(self._topk_shannon_entropy(prompt_logprob))
            else:
                topk_entropies.append(0.0)

        log_probs_arr = np.asarray(token_log_probs, dtype=np.float64)
        if log_probs_arr.size == 0:
            return log_probs_arr, np.array([], dtype=np.float64)

        if has_topk:
            return log_probs_arr, np.asarray(topk_entropies, dtype=np.float64)

        return log_probs_arr, self._causal_window_uncertainty(-log_probs_arr)

    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate SURP score.

        Args:
            output: Model output

        Returns:
            SURP membership score (higher indicates more likely member)
        """
        token_log_probs, entropies = self._extract_log_probs_and_entropies(output)
        if token_log_probs.size == 0:
            return 0.0

        cutoff = (
            self.entropy_threshold
            if self.entropy_threshold is not None
            else float(np.percentile(entropies, self.entropy_percentile)) + 1e-12
        )
        low_entropy_mask = entropies <= cutoff
        candidate_log_probs = (
            token_log_probs[low_entropy_mask]
            if np.any(low_entropy_mask)
            else token_log_probs
        )

        k_length = max(1, int(candidate_log_probs.size * self.ratio))
        lowest_k = np.sort(candidate_log_probs)[:k_length]
        return float(np.mean(lowest_k))
