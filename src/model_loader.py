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

from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class ModelLoader:
    """vLLM model loader class"""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Initialize model loader

        Args:
            model_config: Model configuration
        """
        self.model_config = model_config
        self.model = self._load_model()
        self.tokenizer = self.model.get_tokenizer()
        self._setup_tokenizer()

    def _load_model(self) -> LLM:
        """Load model

        Returns:
            LLM model
        """
        model_id = self.model_config.get("model_id")
        if model_id is None:
            raise ValueError(
                "model_id is required. Please specify the 'model_id' key in model_config."
            )
        # Extract all parameters except model_id
        llm_kwargs = {k: v for k, v in self.model_config.items() if k != "model_id"}

        model = LLM(model=model_id, **llm_kwargs)

        return model

    def _setup_tokenizer(self) -> None:
        """Setup tokenizer"""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def get_lora_request(self, lora_config: dict[str, Any]) -> LoRARequest:
        """Get LoRA request

        Args:
            lora_config: LoRA configuration

        Returns:
            LoRA request
        """
        return LoRARequest(**lora_config)

    def get_sampling_params(
        self, sampling_parameters: dict[str, Any]
    ) -> SamplingParams:
        """Get sampling parameters

        Args:
            sampling_parameters: Sampling parameters configuration

        Returns:
            Sampling parameters
        """
        return SamplingParams(**sampling_parameters)
