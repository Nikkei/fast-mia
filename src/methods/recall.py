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
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.tokenizers import TokenizerLike

from .base import BaseMethod
from .prefix_utils import (
    build_conditional_texts,
    compute_conditional_prefix_length,
    compute_prefix_loss,
    get_bos_token_id,
    get_separator_token,
    has_post_prefix_logprobs,
    process_prefix,
    select_prefix_candidates,
)


class ReCaLLMethod(BaseMethod):
    """ReCaLL membership inference method"""

    requires_labels: bool = True
    requires_tokenizer: bool = True

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize ReCaLL-based method

        Args:
            method_config: Method configuration
        """
        super().__init__("recall", method_config)
        # Default settings
        self.num_shots = self.method_config.get("num_shots", 10)
        self.pass_window = self.method_config.get("pass_window", False)

    def process_output(self, output: RequestOutput, prefix_token_length: int) -> float:
        """Process model output and calculate loss (negative log-likelihood)

        Args:
            output: Model output
            prefix_token_length: Number of prefix tokens to exclude

        Returns:
            Loss
        """
        return compute_prefix_loss(output, prefix_token_length)

    def run(
        self,
        texts: list[str],
        labels: list[int],
        model: LLM,
        tokenizer: TokenizerLike,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """ReCaLL algorithm to calculate scores for a list of texts
        Args:
            texts: List of texts
            labels: List of labels
            model: LLM model
            tokenizer: Tokenizer
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of ReCaLL scores
        """
        if not texts:
            return []

        strip_spaces = bool(
            data_config and not data_config.get("space_delimited_language", True)
        )
        if strip_spaces:
            texts = [text.replace(" ", "") for text in texts]

        # Get texts with label 0
        nonmembership_texts = [
            text for text, label in zip(texts, labels, strict=True) if label == 0
        ]
        nonmember_prefix = select_prefix_candidates(nonmembership_texts, self.num_shots)
        requested_shots = (
            len(nonmember_prefix)
            if len(set(nonmembership_texts)) > 1
            else self.num_shots
        )

        avg_length = int(np.mean([len(tokenizer.encode(text)) for text in texts]))
        processed_nonmember_prefix, processed_num_shots = process_prefix(
            model,
            tokenizer,
            nonmember_prefix,
            avg_length,
            self.pass_window,
            requested_shots,
        )
        if strip_spaces:
            processed_nonmember_prefix = [
                prefix.replace(" ", "") for prefix in processed_nonmember_prefix
            ]

        if processed_num_shots != self.num_shots:
            logging.warning(
                f"num_shots was changed: {self.num_shots} -> {processed_num_shots}"
            )

        fallback_indices: set[int] = set()
        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )
        losses: list[float] = []
        for idx, output in enumerate(outputs):
            if not has_post_prefix_logprobs(output, 0):
                fallback_indices.add(idx)
            losses.append(self.process_output(output, 0))

        sep_token = get_separator_token(tokenizer)
        bos_token_id = get_bos_token_id(tokenizer)
        conditional_texts = build_conditional_texts(
            texts=texts,
            pool_texts=nonmembership_texts,
            processed_prefix=processed_nonmember_prefix,
            model=model,
            tokenizer=tokenizer,
            avg_length=avg_length,
            pass_window=self.pass_window,
            sep_token=sep_token,
        )
        conditional_outputs = self.get_outputs(
            conditional_texts, model, sampling_params, lora_request, data_config
        )
        conditional_losses: list[float] = []
        for idx, (cond_output, uncond_output) in enumerate(
            zip(conditional_outputs, outputs, strict=True)
        ):
            prefix_len = compute_conditional_prefix_length(
                cond_output, uncond_output, bos_token_id
            )
            if not has_post_prefix_logprobs(cond_output, prefix_len):
                fallback_indices.add(idx)
            conditional_losses.append(self.process_output(cond_output, prefix_len))

        if fallback_indices:
            logging.warning(
                f"{self.method_name}: {len(fallback_indices)}/{len(outputs)} sample(s) "
                "had no valid target token logprobs; fell back to 0.0 loss."
            )

        eps = 1e-10
        scores = [
            conditional_loss / (loss + eps)
            for conditional_loss, loss in zip(conditional_losses, losses, strict=True)
        ]

        return scores
