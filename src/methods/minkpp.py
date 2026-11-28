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

import copy
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .base import BaseMethod


class MinKPPMethod(BaseMethod):
    """Min-K%++ membership inference method.

    Normalizes each token's log-probability by the expectation and standard
    deviation of candidate log-probabilities at that context position (estimated
    from vLLM's top-K ``prompt_logprobs``) before averaging the lowest K%
    standardized scores (Zhang et al., ICLR 2025; arXiv:2404.02936).
    """

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize Min-K%++ method.

        Args:
            method_config: Method configuration
                - ratio: Ratio of lowest standardized token scores to average,
                  in (0.0, 1.0] (default 0.2)
                - top_k: Minimum number of top prompt_logprobs requested from
                  vLLM to approximate mu_t and sigma_t (default 20)
                - eps: Numerical stability floor for sigma_t (default 1e-6)

        Raises:
            ValueError: If 'ratio', 'top_k', or 'eps' is invalid
        """
        config = dict(method_config) if method_config is not None else {}
        ratio = float(config.get("ratio", 0.2))
        top_k = int(config.get("top_k", 20))
        eps = float(config.get("eps", 1e-6))

        if not (0.0 < ratio <= 1.0):
            raise ValueError("MinKPPMethod requires 'ratio' in (0.0, 1.0].")
        if top_k < 2:
            raise ValueError("MinKPPMethod requires 'top_k' >= 2.")
        if eps <= 0.0:
            raise ValueError("MinKPPMethod requires positive 'eps'.")

        config["ratio"] = ratio
        config["top_k"] = top_k
        config["eps"] = eps

        method_name = f"minkpp_{ratio}"
        super().__init__(method_name, config)
        self.ratio = ratio
        self.top_k = top_k
        self.eps = eps

    def _ensure_top_k_sampling_params(
        self, sampling_params: SamplingParams
    ) -> SamplingParams:
        """Ensure ``sampling_params.prompt_logprobs`` requests at least ``top_k``."""
        current = getattr(sampling_params, "prompt_logprobs", None)
        if (
            isinstance(current, int)
            and not isinstance(current, bool)
            and current >= self.top_k
        ):
            return sampling_params

        cloned = copy.copy(sampling_params)
        cloned.prompt_logprobs = self.top_k
        return cloned

    def _standardized_token_scores(self, output: RequestOutput) -> np.ndarray:
        """Compute per-position z-scored logprobs (log p(x_t) - mu_t) / sigma_t."""
        z_scores: list[float] = []
        for token_id, prompt_logprob in zip(
            output.prompt_token_ids, output.prompt_logprobs, strict=True
        ):
            if prompt_logprob is None:
                continue

            target_lp = float(prompt_logprob[token_id].logprob)
            if len(prompt_logprob) <= 1:
                z_scores.append(target_lp)
                continue

            cand_lps = np.asarray(
                [entry.logprob for entry in prompt_logprob.values()], dtype=np.float64
            )
            shifted = cand_lps - float(np.max(cand_lps))
            weights = np.exp(shifted)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0:
                z_scores.append(target_lp)
                continue

            weights /= weight_sum
            mu_t = float(np.sum(weights * cand_lps))
            var_t = float(np.sum(weights * np.square(cand_lps - mu_t)))
            sigma_t = max(float(np.sqrt(max(0.0, var_t))), self.eps)
            z_scores.append((target_lp - mu_t) / sigma_t)

        return np.asarray(z_scores, dtype=np.float64)

    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate Min-K%++ score.

        Args:
            output: Model output

        Returns:
            Min-K%++ membership score
        """
        z_scores = self._standardized_token_scores(output)
        if z_scores.size == 0:
            return 0.0

        k_length = max(1, int(z_scores.size * self.ratio))
        lowest_k = np.sort(z_scores)[:k_length]
        return float(np.mean(lowest_k))

    def run(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """Run Min-K%++ inference, upgrading prompt_logprobs to top_k if needed.

        Args:
            texts: List of input texts
            model: LLM model
            sampling_params: Sampling parameters
            lora_request: Optional LoRA request
            data_config: Optional data configuration

        Returns:
            List of Min-K%++ scores
        """
        effective_params = self._ensure_top_k_sampling_params(sampling_params)
        outputs = self.get_outputs(
            texts, model, effective_params, lora_request, data_config
        )
        return [self.process_output(output) for output in outputs]
