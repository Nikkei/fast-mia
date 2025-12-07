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

import zlib
from collections import Counter
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .base import BaseMethod


def get_prefix(text: str, prefix_ratio: float) -> str:
    num_words = len(text.split())
    num_prefix_words = int(num_words * prefix_ratio)
    prefix = " ".join(text.split()[:num_prefix_words])
    return prefix


def get_suffix(text: str, prefix_ratio: float, text_length: int) -> list:
    """
    Extracts a suffix from the given text, based on the specified prefix ratio and text length.
    """
    words = text.split(" ")
    words = [word for word in words if word != ""]
    words = words[round(text_length * prefix_ratio) :]
    return words


def ngrams(sequence: str, n: int) -> zip:
    """
    Generates n-grams from a sequence.
    """
    return zip(*[sequence[i:] for i in range(n)], strict=False)


def rouge_n(candidate: list, reference: list, n: int = 1) -> float:
    """
    Calculates the ROUGE-N score between a candidate and a reference.
    """
    if not candidate or not reference:
        return 0
    candidate_ngrams = list(ngrams(candidate, n))
    reference_ngrams = list(ngrams(reference, n))
    ref_words_count = Counter(reference_ngrams)
    cand_words_count = Counter(candidate_ngrams)
    overlap = ref_words_count & cand_words_count
    recall = sum(overlap.values()) / len(reference)
    return recall


class SaMIAMethod(BaseMethod):
    """SaMIA membership inference method"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize SaMIA-based method

        Args:
            method_config: Method configuration
        """
        super().__init__("samia", method_config)
        # Default settings
        self.num_samples = self.method_config.get("num_samples", 5)
        self.prefix_ratio = self.method_config.get("prefix_ratio", 0.5)
        self.zlib = self.method_config.get("zlib", True)

    def process_output(self, output: RequestOutput) -> float:
        """Calculate SaMIA score from a single model output
        Note: This method is called from BaseMethod.run, but
        for SaMIA, a custom implementation using multiple samples is used,
        so this method is not supported for single output.
        Use run method instead.

        Args:
            output: Model output

        Returns:
            SaMIA score
        """
        raise NotImplementedError(
            "process_output is not supported in SaMIAMethod. Use run method instead."
        )

    def run(
        self,
        texts: list[str],
        model: LLM,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """SaMIA algorithm to calculate scores for a list of texts
        Args:
            texts: List of texts
            model: LLM model
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of SaMIA scores
        """

        prefixs = [get_prefix(text, self.prefix_ratio) for text in texts]

        samia_params = SamplingParams(
            max_tokens=1024,
            temperature=1,
            n=self.num_samples,
            top_k=50,
            top_p=1,
        )

        outputs = self.get_outputs(
            prefixs, model, samia_params, lora_request, data_config
        )
        scores = []
        for text, output in zip(texts, outputs, strict=True):
            suffix_ref = get_suffix(
                text, 1 - self.prefix_ratio, data_config.get("text_length")
            )
            rouge_scores = []
            for i in range(self.num_samples):
                output_text = output.outputs[i].text
                suffix_cand = get_suffix(
                    output_text, 1 - self.prefix_ratio, data_config.get("text_length")
                )
                if self.zlib:
                    if data_config.get("space_delimited_language"):
                        zlib_cand = zlib.compress(" ".join(suffix_cand).encode("utf-8"))
                    else:
                        zlib_cand = zlib.compress("".join(suffix_cand).encode("utf-8"))
                    rouge_scores.append(
                        rouge_n(suffix_cand, suffix_ref, n=1) * len(zlib_cand)
                    )
                else:
                    rouge_scores.append(rouge_n(suffix_cand, suffix_ref, n=1))
            scores.append(rouge_scores)

        scores = np.array(scores).mean(axis=1).tolist()
        return scores
