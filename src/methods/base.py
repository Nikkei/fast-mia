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

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput


class BaseMethod(ABC):
    """Base class for membership inference methods"""

    # Model output cache (shared by all methods) - uses OrderedDict to manage LRU
    _model_cache = OrderedDict()
    # Maximum cache size (number of entries)
    _max_cache_size = 1000
    # Cache statistics
    _cache_stats = {
        "model_hits": 0,
        "model_misses": 0,
        "score_hits": 0,
        "score_misses": 0,
    }

    def __init__(self, method_name: str, method_config: dict[str, Any] = None) -> None:
        """Initialize membership inference method

        Args:
            method_name: Name of the method
            method_config: Method configuration
        """
        self.method_name = method_name
        self.method_config = method_config or {}

    @abstractmethod
    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate score

        Args:
            output: Model output

        Returns:
            Score
        """
        pass

    @staticmethod
    def _get_model_cache_key(
        texts: list[str],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> str:
        """Generate model cache key

        Args:
            texts: List of texts
            sampling_params: Sampling parameters
            lora_request: LoRA request

        Returns:
            Cache key
        """
        # Hash each text
        text_hashes = [hashlib.md5(text.encode()).hexdigest() for text in sorted(texts)]

        # Concatenate all text hashes and hash again
        texts_hash = hashlib.md5("|".join(text_hashes).encode()).hexdigest()

        # Convert sampling parameters to string
        params_str = f"{sampling_params.max_tokens}_{sampling_params.temperature}_{sampling_params.top_p}"

        # If LoRA request exists, add its ID and name
        lora_str = ""
        if lora_request:
            lora_str = f"_{lora_request.lora_int_id}_{lora_request.lora_name}"

        return f"{texts_hash}_{params_str}{lora_str}"

    @classmethod
    def get_cache_stats(cls) -> dict[str, Any]:
        """Get cache statistics

        Returns:
            Cache statistics
        """
        stats = cls._cache_stats.copy()

        # Model cache hit rate
        model_total = stats["model_hits"] + stats["model_misses"]
        if model_total > 0:
            stats["model_hit_rate"] = f"{stats['model_hits'] / model_total:.2%}"
        else:
            stats["model_hit_rate"] = "0.00%"

        # Score cache hit rate
        score_total = stats["score_hits"] + stats["score_misses"]
        if score_total > 0:
            stats["score_hit_rate"] = f"{stats['score_hits'] / score_total:.2%}"
        else:
            stats["score_hit_rate"] = "0.00%"

        # Cache size
        stats["model_cache_size"] = len(cls._model_cache)

        return stats

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cache"""
        cls._model_cache.clear()
        logging.info("Cache cleared")

    @classmethod
    def set_max_cache_size(cls, size: int) -> None:
        """Set maximum cache size

        Args:
            size: Maximum cache size (number of entries)
        """
        cls._max_cache_size = size
        logging.info(f"Max cache size set to {size}")

        # Remove excess entries
        cls._trim_cache()

    @classmethod
    def _trim_cache(cls) -> None:
        """Remove old entries if cache size exceeds maximum"""
        # Clean up model cache
        while len(cls._model_cache) > cls._max_cache_size:
            cls._model_cache.popitem(last=False)  # Remove oldest item

    def get_outputs(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[RequestOutput]:
        """Get model outputs for a list of texts

        Args:
            texts: List of texts
            model: LLM model
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of model outputs
        """
        # If data_config is provided and space_delimited_language is False, remove spaces
        if data_config and not data_config.get("space_delimited_language", True):
            texts = [text.replace(" ", "") for text in texts]

        # Generate model cache key
        model_cache_key = self._get_model_cache_key(
            texts, sampling_params, lora_request
        )

        # If in model cache, return cached outputs
        if model_cache_key in BaseMethod._model_cache:
            BaseMethod._cache_stats["model_hits"] += 1
            logging.info(
                f"Model cache hit for method {self.method_name}: {BaseMethod.get_cache_stats()}"
            )
            # Update LRU order
            BaseMethod._model_cache.move_to_end(model_cache_key)
            return BaseMethod._model_cache[model_cache_key]

        # If not in model cache, generate with model
        BaseMethod._cache_stats["model_misses"] += 1
        logging.info(
            f"Model cache miss for method {self.method_name}: {BaseMethod.get_cache_stats()}"
        )

        outputs = model.generate(texts, sampling_params, lora_request=lora_request)

        # Save outputs to cache
        BaseMethod._model_cache[model_cache_key] = outputs
        # Check and adjust cache size
        self._trim_cache()

        return outputs

    def run(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """Run inference for a list of texts

        Args:
            texts: List of texts
            model: LLM model
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of scores
        """
        # Get model outputs
        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )

        # Calculate scores from outputs
        scores = [self.process_output(output) for output in outputs]

        return scores
