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

import math
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .base import BaseMethod

_SUPPORTED_SIGNALS = ("combined", "slope", "cbm", "cbpm", "mean_below")
_SUPPORTED_AGGREGATIONS = ("edgington", "linear")
_COMBINED_COMPONENTS = ("mean_below", "cbm", "cbpm", "slope")


def _empirical_cdf(values: np.ndarray) -> np.ndarray:
    """Compute tie-aware empirical CDF values in (0, 1] using average ranks."""
    n = values.size
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    diffs = np.concatenate(([True], sorted_vals[1:] != sorted_vals[:-1], [True]))
    boundaries = np.flatnonzero(diffs)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
    return ranks / float(n)


class CAMIAMethod(BaseMethod):
    """Context-Aware Membership Inference Attack (CAMIA).

    Captures sequence-level token loss dynamics (count below mean, count below
    prefix mean, loss trajectory slope, and below-mean log-likelihood) calibrated
    by lexical token diversity, requiring zero additional model forward passes
    beyond standard prompt logprobs (Chang et al., 2025; arXiv:2409.13745).
    """

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize CAMIA method.

        Args:
            method_config: Method configuration
                - signal: Which dynamic signal to score ('combined', 'slope',
                  'cbm', 'cbpm', 'mean_below'; default 'combined')
                - aggregation: Signal combination rule when signal='combined'
                  ('edgington' or 'linear'; default 'edgington')
                - diversity_calibration: Penalize low-diversity repetitive
                  token sequences (default True)

        Raises:
            ValueError: If 'signal' or 'aggregation' is unsupported
        """
        config = dict(method_config) if method_config is not None else {}
        signal = str(config.get("signal", "combined"))
        aggregation = str(config.get("aggregation", "edgington"))
        if signal not in _SUPPORTED_SIGNALS:
            supported = ", ".join(_SUPPORTED_SIGNALS)
            raise ValueError(
                f"Unsupported CAMIA signal '{signal}'. Supported signals: {supported}"
            )
        if aggregation not in _SUPPORTED_AGGREGATIONS:
            supported = ", ".join(_SUPPORTED_AGGREGATIONS)
            raise ValueError(
                f"Unsupported CAMIA aggregation '{aggregation}'. "
                f"Supported aggregations: {supported}"
            )

        method_name = "camia" if signal == "combined" else f"camia_{signal}"
        super().__init__(method_name, config)
        self.signal = signal
        self.aggregation = aggregation
        self.diversity_calibration = bool(config.get("diversity_calibration", True))

    def _compute_diversity(self, output: RequestOutput, num_tokens: int) -> float:
        """Compute token diversity ratio in (0, 1] aligned to extracted logprobs."""
        if not self.diversity_calibration or num_tokens <= 0:
            return 1.0
        prompt_token_ids = getattr(output, "prompt_token_ids", None)
        if prompt_token_ids is None or len(prompt_token_ids) < num_tokens:
            return 1.0
        aligned_ids = np.asarray(prompt_token_ids[-num_tokens:])
        return float(np.unique(aligned_ids).size) / float(aligned_ids.size)

    def _extract_signals(self, output: RequestOutput) -> dict[str, float]:
        """Extract sequence-dynamics signals for a single prompt output."""
        token_log_probs = np.asarray(
            self._extract_token_log_probs(output), dtype=np.float64
        )
        if token_log_probs.size == 0:
            return dict.fromkeys(_COMBINED_COMPONENTS, 0.0)

        nll = -token_log_probs
        num_tokens = int(nll.size)
        diversity = self._compute_diversity(output, num_tokens)

        mean_nll = float(np.mean(nll))
        below_mean = nll <= mean_nll
        cbm = float(np.mean(below_mean)) * diversity

        if num_tokens >= 2:
            prefix_means = np.cumsum(nll[:-1]) / np.arange(
                1, num_tokens, dtype=np.float64
            )
            cbpm = float(np.mean(nll[1:] <= prefix_means)) * diversity

            positions = np.linspace(0.0, 1.0, num_tokens, dtype=np.float64) - 0.5
            denom = float(np.dot(positions, positions))
            raw_slope = (
                float(np.dot(positions, nll - mean_nll) / denom) if denom > 0.0 else 0.0
            )
            # Memorized sequences transition from uncertain start to low loss
            # (negative NLL slope -> higher membership score when negated).
            neg_slope = -raw_slope * diversity
        else:
            cbpm = cbm
            neg_slope = 0.0

        # Log-space diversity penalty lowers scores for repetitive sequences
        # while leaving fully diverse sequences (diversity == 1.0) unchanged.
        mean_below = float(np.mean(token_log_probs[below_mean])) + math.log(
            max(diversity, 1e-6)
        )

        return {
            "mean_below": mean_below,
            "cbm": cbm,
            "cbpm": cbpm,
            "slope": neg_slope,
        }

    def process_output(self, output: RequestOutput) -> float:
        """Process a single model output and calculate its CAMIA score.

        Args:
            output: Model output

        Returns:
            CAMIA membership score (higher indicates more likely member)
        """
        signals = self._extract_signals(output)
        if self.signal != "combined":
            return signals[self.signal]
        return float(
            signals["mean_below"]
            + signals["cbm"]
            + signals["cbpm"]
            + 0.1 * signals["slope"]
        )

    def run(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """Run CAMIA scoring over a list of texts.

        Args:
            texts: List of input texts
            model: LLM model
            sampling_params: Sampling parameters
            lora_request: Optional LoRA request
            data_config: Optional data configuration

        Returns:
            List of CAMIA membership scores
        """
        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )
        if not outputs:
            return []

        if (
            self.signal != "combined"
            or self.aggregation == "linear"
            or len(outputs) == 1
        ):
            return [self.process_output(output) for output in outputs]

        per_sample_signals = [self._extract_signals(output) for output in outputs]
        cdf_sum = np.zeros(len(outputs), dtype=np.float64)
        for comp in _COMBINED_COMPONENTS:
            comp_values = np.asarray(
                [sample[comp] for sample in per_sample_signals], dtype=np.float64
            )
            cdf_sum += _empirical_cdf(comp_values)

        return (cdf_sum / float(len(_COMBINED_COMPONENTS))).tolist()
