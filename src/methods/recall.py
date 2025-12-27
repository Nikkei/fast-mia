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
import random
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .base import BaseMethod


def extract_nonmember_prefix(
    not_membership_texts: list[str], num_shots: int
) -> list[str]:
    random.shuffle(not_membership_texts)
    num_shots = int(num_shots)
    nonmember_prefix = not_membership_texts[:num_shots]

    return nonmember_prefix


## https://github.com/ruoyuxie/recall/blob/main/src/run.py
def process_prefix(
    model: LLM,
    tokenizer: AnyTokenizer,
    prefix: list[str],
    avg_length: int,
    pass_window: bool,
    num_shots: int,
) -> tuple[list[str], int]:
    if pass_window:
        return prefix, num_shots
    max_length = model.llm_engine.get_model_config().max_model_len
    token_counts = [len(tokenizer.encode(shot)) for shot in prefix]
    target_token_count = avg_length
    total_tokens = sum(token_counts) + target_token_count
    if total_tokens <= max_length:
        return prefix, num_shots
    # Determine the maximum number of shots that can fit within the max_length
    max_shots = 0
    cumulative_tokens = target_token_count
    for count in token_counts:
        if cumulative_tokens + count <= max_length:
            max_shots += 1
            cumulative_tokens += count
        else:
            break
    # Truncate the prefix to include only the maximum number of shots
    truncated_prefix = prefix[-max_shots:]
    num_shots = max_shots
    return truncated_prefix, num_shots


class ReCaLLMethod(BaseMethod):
    """ReCaLL membership inference method"""

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

        Returns:
            Loss
        """
        # Get log-probabilities for each token
        token_log_probs = []
        for i, prompt_logprob in enumerate(output.prompt_logprobs):
            # Do not include prefix part in loss calculation
            if i < prefix_token_length:
                continue
            if prompt_logprob is None:
                continue
            token_log_probs.append(list(prompt_logprob.values())[0].logprob)

        loss = -np.mean(token_log_probs)
        return loss

    def run(
        self,
        texts: list[str],
        labels: list[int],
        model: LLM,
        tokenizer: AnyTokenizer,
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
        # Get texts with label 0
        nonmembership_texts = [
            text for text, label in zip(texts, labels, strict=False) if label == 0
        ]
        # Randomly select num_shots texts
        nonmember_prefix = extract_nonmember_prefix(nonmembership_texts, self.num_shots)

        if data_config and not data_config.get("space_delimited_language", True):
            texts = [text.replace(" ", "") for text in texts]
            nonmember_prefix = [prefix.replace(" ", "") for prefix in nonmember_prefix]

        avg_length = int(np.mean([len(tokenizer.encode(text)) for text in texts]))
        processed_nonmember_prefix, processed_num_shots = process_prefix(
            model,
            tokenizer,
            nonmember_prefix,
            avg_length,
            self.pass_window,
            self.num_shots,
        )

        if processed_num_shots != self.num_shots:
            logging.warning(
                f"num_shots was changed: {self.num_shots} -> {processed_num_shots}"
            )

        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )
        losses = [self.process_output(output, 0) for output in outputs]

        prefix = "".join(processed_nonmember_prefix)
        # prefix [bos] text
        # If tokenizer.bos_token is None, use ''
        sep_token = tokenizer.bos_token if tokenizer.bos_token is not None else ""
        conditional_texts = [prefix + sep_token + text for text in texts]
        # Calculate the length of the prefix tokens to exclude from loss calculation
        prefix_token_length = tokenizer(prefix, return_tensors="pt").input_ids.size(1)
        conditional_outputs = self.get_outputs(
            conditional_texts, model, sampling_params, lora_request, data_config
        )
        conditional_losses = [
            self.process_output(output, prefix_token_length)
            for output in conditional_outputs
        ]

        eps = 1e-10
        scores = [
            conditional_loss / (loss + eps)
            for conditional_loss, loss in zip(conditional_losses, losses, strict=False)
        ]

        return scores
