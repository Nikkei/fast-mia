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

import numpy as np
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
    labels: list[int] | None
    data_stats: dict[str, Any]
    cache_stats: dict[str, Any] = field(default_factory=dict)
    texts: list[str] | None = None


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
        audit_mode = labels is None

        if audit_mode:
            for method in self.methods:
                if method.requires_labels:
                    raise ValueError(
                        f"Method '{method.method_name}' requires ground-truth labels "
                        "and cannot be used in unlabeled audit mode "
                        "(data.label_column: null)."
                    )
            data_stats: dict[str, Any] = {
                "num_samples": len(texts),
                "num_members": None,
                "num_nonmembers": None,
                "audit_mode": True,
            }
        else:
            assert labels is not None
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
            kwargs: dict[str, Any] = {
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

            scores = method.run(**kwargs)

            if audit_mode:
                scores_arr = np.asarray(scores, dtype=float)
                if scores_arr.size > 0 and not np.all(np.isnan(scores_arr)):
                    mean_score = float(np.nanmean(scores_arr))
                    std_score = float(np.nanstd(scores_arr))
                    min_score = float(np.nanmin(scores_arr))
                    median_score = float(np.nanmedian(scores_arr))
                    max_score = float(np.nanmax(scores_arr))
                else:
                    mean_score = float("nan")
                    std_score = float("nan")
                    min_score = float("nan")
                    median_score = float("nan")
                    max_score = float("nan")

                results["method"].append(method.method_name)
                results["mean_score"].append(mean_score)
                results["std_score"].append(std_score)
                results["min_score"].append(min_score)
                results["median_score"].append(median_score)
                results["max_score"].append(max_score)

                detailed_results.append(
                    {
                        "method_name": method.method_name,
                        "scores": scores,
                        "auroc": float("nan"),
                        "fpr95": float("nan"),
                        "tpr05": float("nan"),
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "min_score": min_score,
                        "median_score": median_score,
                        "max_score": max_score,
                    }
                )
            else:
                assert labels is not None
                # Calculate metrics
                auroc, fpr95, tpr05 = get_metrics(scores, labels)

                # Add results for DataFrame (raw values; format only for display)
                results["method"].append(method.method_name)
                results["auroc"].append(auroc)
                results["fpr95"].append(fpr95)
                results["tpr05"].append(tpr05)

                # Add detailed results for visualization
                detailed_results.append(
                    {
                        "method_name": method.method_name,
                        "scores": scores,
                        "auroc": auroc,
                        "fpr95": fpr95,
                        "tpr05": tpr05,
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
            texts=texts,
        )
