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

import random
from copy import deepcopy
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .base import BaseMethod


def random_swap(words: list[str], n: int) -> list[str]:
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words: list[str]) -> list[str]:
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = (
        new_words[random_idx_2],
        new_words[random_idx_1],
    )
    return new_words


def eda(sentence: str, alpha: float, num_aug: int) -> list[str]:
    words = sentence.split(" ")
    num_words = len(words)
    augmented_sentences = []

    if alpha > 0:
        n_rs = max(1, int(alpha * num_words))
        for _ in range(num_aug):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(" ".join(a_words))

    augmented_sentences = [sentence for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [
            s for s in augmented_sentences if random.uniform(0, 1) < keep_prob
        ]

    return augmented_sentences


def calculate_polarized_distance(
    prob_list: list[float], ratio_local: float = 0.3, ratio_far: float = 0.05
) -> float:
    local_region_length = max(int(len(prob_list) * ratio_local), 1)
    far_region_length = max(int(len(prob_list) * ratio_far), 1)
    local_region = np.sort(prob_list)[:local_region_length]
    far_region = np.sort(prob_list)[::-1][:far_region_length]
    return np.mean(far_region) - np.mean(local_region)


class PACMethod(BaseMethod):
    """PAC (Polarized Augment Calibration) based membership inference method"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize PAC (Polarized Augment Calibration) based method

        Args:
            method_config: Method configuration
        """
        super().__init__("pac", method_config)
        # Default settings
        self.alpha = self.method_config.get("alpha", 0.3)
        self.N = self.method_config.get("N", 5)

    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate Polarized Distance

        Args:
            output: Model output

        Returns:
            Polarized Distance
        """
        token_log_probs = self._extract_token_log_probs(output)
        return calculate_polarized_distance(token_log_probs)

    def run(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """PAC algorithm to calculate scores for a list of texts.
        Args:
            texts: List of texts
            model: LLM model
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of PAC scores
        """
        # eda
        eda_texts = []
        for text in texts:
            eda_text = eda(text, self.alpha, self.N)
            eda_texts.extend(deepcopy(eda_text))

        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )
        eda_outputs = self.get_outputs(
            eda_texts, model, sampling_params, lora_request, data_config
        )

        pds = [self.process_output(output) for output in outputs]
        eda_pds = [self.process_output(eda_output) for eda_output in eda_outputs]
        calibrated_pds = [
            np.mean(eda_pds[i : i + self.N]) for i in range(0, len(eda_pds), self.N)
        ]
        scores = list(np.array(pds) - np.array(calibrated_pds))

        # https://github.com/yyy01/PAC/blob/79c6eca14fed3a64a7a4201cdca263de52501dae/attack.py#L49
        return [-score for score in scores]
