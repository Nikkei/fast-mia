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
        results = defaultdict(list)
        detailed_results = []

        for method in self.methods:
            # Build arguments based on method requirements
            args = [texts]
            if method.requires_labels:
                args.append(labels)
            args.append(self.model_loader.model)
            if method.requires_tokenizer:
                args.append(self.model_loader.tokenizer)
            if method.requires_sampling_params:
                args.append(sampling_params)
            args.append(lora_request)

            scores = method.run(*args, data_config=config.data)

            # Calculate metrics
            auroc, fpr95, tpr05 = get_metrics(scores, labels)

            # Add results for DataFrame
            results["method"].append(method.method_name)
            results["auroc"].append(f"{auroc:.1%}")
            results["fpr95"].append(f"{fpr95:.1%}")
            results["tpr05"].append(f"{tpr05:.1%}")

            # Add detailed results for visualization
            detailed_results.append({
                "method_name": method.method_name,
                "scores": scores,
                "auroc": auroc,
                "fpr95": fpr95,
                "tpr05": tpr05,
            })

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
