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

import gzip
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .base import BaseMethod


def download_c4_data(file_num: int) -> None:
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/"

    for i in range(file_num):
        fname = f"c4-train.{i:05d}-of-01024.json.gz"
        url = f"{base_url}{fname}"
        save_path = output_dir / fname

        if save_path.exists():
            logging.info(f"Skipping {fname} (already exists)")
            continue

        logging.info(f"Downloading {fname}...")
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            logging.error(f"Failed to download {fname}: {e}")


def upadate_freq_dist(
    examples: list[str],
    tokenizer: AnyTokenizer,
    freq_dist: list[int],
    max_token_length: int,
) -> list[int]:
    example_texts = [example["text"] for example in examples]
    logging.info("encoding start")
    example_input_ids = tokenizer.batch_encode_plus(
        example_texts, truncation=True, max_length=max_token_length
    )["input_ids"]
    logging.info("encoding end")
    for input_ids in example_input_ids:
        for token_id in input_ids:
            freq_dist[token_id] += 1
    return freq_dist


class DCPDDMethod(BaseMethod):
    """DC-PDD membership inference method"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize DC-PDD method

        Args:
            method_config: Method configuration
        """
        super().__init__("dcpdd", method_config)
        # Default settings
        self.file_num = self.method_config.get("file_num", 15)
        self.max_token_length = self.method_config.get("max_token_length", 1024)
        self.alpha = self.method_config.get("alpha", 0.01)

    def process_output(
        self, output: RequestOutput, input_ids: list[int], freq_dist: list[int]
    ) -> float:
        """Process model output and calculate DC-PDD score

        Args:
            output: Model output

        Returns:
            DC-PDD score
        """
        token_log_probs = []
        for prompt_logprob in output.prompt_logprobs:
            if prompt_logprob is None:
                continue
            token_log_probs.append(list(prompt_logprob.values())[0].logprob)

        # tokens with first occurance in text
        indexes = []
        current_ids = []
        for i, input_id in enumerate(input_ids):
            if input_id not in current_ids:
                indexes.append(i)
                current_ids.append(input_id)

        x_prob = np.array(token_log_probs)[indexes]
        x_freq = np.array(freq_dist)[np.array(input_ids)[indexes]]
        eps = 1e-10
        ce = x_prob * np.log(1 / (x_freq + eps))
        ce[ce > self.alpha] = self.alpha
        dcpdd_score = np.mean(ce)

        return dcpdd_score

    def run(
        self,
        texts: list[str],
        model: LLM,
        tokenizer: AnyTokenizer,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """DC-PDD algorithm to calculate scores for a list of texts
        Args:
            texts: List of texts
            labels: List of labels
            model: LLM model
            tokenizer: Tokenizer
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of DC-PDD scores
        """
        cache_dir = Path(".fastmia_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"freq_dist_{self.file_num}.json"

        if cache_path.exists():
            logging.info(f"Loading freq_dist from {cache_path}")
            with cache_path.open() as f:
                freq_dist = json.load(f)
        else:
            # Download C4 data files
            download_c4_data(self.file_num)

            # calculate frequency distribution from C4 data
            freq_dist = [0] * len(tokenizer)

            for i in range(self.file_num):
                fname = f"c4-train.{i:05d}-of-01024.json.gz"
                with gzip.open(
                    Path("data") / fname, "rt", encoding="utf-8"
                ) as f:
                    examples = []
                    for example in f:
                        example = json.loads(example)
                        examples.append(example)
                    freq_dist = upadate_freq_dist(
                        examples, tokenizer, freq_dist, self.max_token_length
                    )

            logging.info(f"Saving freq_dist to {cache_path}")
            with cache_path.open("w") as f:
                json.dump(freq_dist, f)

        # Get model outputs
        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )

        # Calculate scores from outputs
        scores = [
            self.process_output(output, tokenizer.encode(text), freq_dist)
            for output, text in zip(outputs, texts, strict=False)
        ]

        return scores
