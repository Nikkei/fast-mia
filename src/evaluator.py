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
from typing import Any

import pandas as pd

from .data_loader import DataLoader
from .methods import BaseMethod
from .model_loader import ModelLoader
from .utils import get_metrics


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
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Evaluate membership inference attacks on data with specified number of words

        Args:
            config: Configuration

        Returns:
            DataFrame of evaluation results
        """
        # Get required parameters from config
        token_length = config.data.get("token_length", 32)
        lora_config = config.lora
        sampling_parameters_config = config.sampling_parameters

        # Get data
        texts, labels = self.data_loader.get_data(token_length)

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
        for method in self.methods:
            if method.method_name == "samia":
                # Run method
                scores = method.run(
                    texts,
                    self.model_loader.model,
                    lora_request,
                    data_config=config.data,
                )
            elif method.method_name in ["recall", "conrecall"]:
                # Run method
                scores = method.run(
                    texts,
                    labels,
                    self.model_loader.model,
                    self.model_loader.tokenizer,
                    sampling_params,
                    lora_request,
                    data_config=config.data,
                )
            else:
                # Run method
                scores = method.run(
                    texts,
                    self.model_loader.model,
                    sampling_params,
                    lora_request,
                    data_config=config.data,
                )

            # Calculate metrics
            auroc, fpr95, tpr05 = get_metrics(scores, labels)

            # Add results
            results["method"].append(method.method_name)
            results["auroc"].append(f"{auroc:.1%}")
            results["fpr95"].append(f"{fpr95:.1%}")
            results["tpr05"].append(f"{tpr05:.1%}")

        # Log cache stats after evaluation
        logging.info(f"Cache stats after evaluation: {BaseMethod.get_cache_stats()}")

        return pd.DataFrame(results)
