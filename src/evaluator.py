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
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import Config
from .data_loader import DataLoader
from .methods import BaseMethod
from .model_loader import ModelLoader
from .utils import get_metrics


@dataclass
class EvaluationResult:
    """Container for evaluation results with detailed information"""

    results_df: pd.DataFrame
    detailed_results: list[dict[str, Any]]
    labels: list[int]
    data_stats: dict[str, Any]
    cache_stats: dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """Evaluator for membership inference attacks"""

    def __init__(
        self,
        data_loader: DataLoader,
        model_loader: ModelLoader,
        methods: list[BaseMethod],
        max_cache_size: int = 1000,
    ) -> None:
        """Initialize the evaluator

        Args:
            data_loader: Data loader
            model_loader: Model loader
            methods: List of methods to use for evaluation
            max_cache_size: Maximum cache size
        """
        self.data_loader = data_loader
        self.model_loader = model_loader
        self.methods = methods

        # Set cache size
        BaseMethod.set_max_cache_size(max_cache_size)

    def evaluate(
        self,
        config: Config,
    ) -> EvaluationResult:
        """Evaluate membership inference attacks on data with specified number of words

        Args:
            config: Configuration

        Returns:
            EvaluationResult containing DataFrame, detailed results, labels, and stats
        """
        # Get required parameters from config
        text_length = config.data.get("text_length", 32)
        lora_config = config.lora
        sampling_parameters_config = config.sampling_parameters

        # Get data
        texts, labels = self.data_loader.get_data(text_length)

        # Calculate data statistics
        data_stats = {
            "num_samples": len(texts),
            "num_members": sum(labels),
            "num_nonmembers": len(labels) - sum(labels),
        }

        # Get sampling parameters
        sampling_params = self.model_loader.get_sampling_params(
            sampling_parameters_config or {}
        )

        # Get LoRA request (if configured)
        lora_request = None
        if lora_config:
            lora_request = self.model_loader.get_lora_request(lora_config)

        # Evaluate with each method
        results: dict[str, list[Any]] = defaultdict(list)
        detailed_results: list[dict[str, Any]] = []

        for method in self.methods:
            # Build keyword arguments based on method requirements
            kwargs = {
                "texts": texts,
                "model": self.model_loader.model,
                "lora_request": lora_request,
                "data_config": config.data,
            }
            if method.requires_labels:
                kwargs["labels"] = labels
            if method.requires_tokenizer:
                kwargs["tokenizer"] = self.model_loader.tokenizer
            if method.requires_sampling_params:
                kwargs["sampling_params"] = sampling_params

            cache_before = BaseMethod.get_cache_stats()
            method_start = time.perf_counter()
            scores = method.run(**kwargs)
            elapsed_seconds = time.perf_counter() - method_start
            cache_after = BaseMethod.get_cache_stats()

            cache_hits_delta = cache_after.get("model_hits", 0) - cache_before.get(
                "model_hits", 0
            )
            cache_misses_delta = cache_after.get("model_misses", 0) - cache_before.get(
                "model_misses", 0
            )

            # Calculate metrics
            auroc, fpr95, tpr05, tpr01 = get_metrics(scores, labels)

            # Add results for DataFrame (raw values; format only for display)
            results["method"].append(method.method_name)
            results["auroc"].append(auroc)
            results["fpr95"].append(fpr95)
            results["tpr05"].append(tpr05)
            results["tpr01"].append(tpr01)
            results["elapsed_seconds"].append(elapsed_seconds)
            results["cache_hits"].append(cache_hits_delta)
            results["cache_misses"].append(cache_misses_delta)

            # Add detailed results for visualization and reporting
            detailed_results.append(
                {
                    "method_name": method.method_name,
                    "scores": scores,
                    "auroc": auroc,
                    "fpr95": fpr95,
                    "tpr05": tpr05,
                    "tpr01": tpr01,
                    "elapsed_seconds": elapsed_seconds,
                    "cache_hits": cache_hits_delta,
                    "cache_misses": cache_misses_delta,
                }
            )

        # Get cache stats after evaluation
        cache_stats = BaseMethod.get_cache_stats()
        logging.info(f"Cache stats after evaluation: {cache_stats}")

        return EvaluationResult(
            results_df=pd.DataFrame(results),
            detailed_results=detailed_results,
            labels=labels,
            data_stats=data_stats,
            cache_stats=cache_stats,
        )
