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

import numpy as np
from vllm import LLM
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer


def extract_prefix(texts: list[str], num_shots: int) -> list[str]:
    """Randomly select num_shots texts from the list without modifying the original.

    Args:
        texts: List of texts to sample from
        num_shots: Number of texts to select

    Returns:
        List of randomly selected texts
    """
    num_shots = min(int(num_shots), len(texts))
    return random.sample(texts, num_shots)


## https://github.com/ruoyuxie/recall/blob/main/src/run.py
def process_prefix(
    model: LLM,
    tokenizer: AnyTokenizer,
    prefix: list[str],
    avg_length: int,
    pass_window: bool,
    num_shots: int,
) -> tuple[list[str], int]:
    """Process prefix to fit within model's max length.

    Args:
        model: LLM model
        tokenizer: Tokenizer
        prefix: List of prefix texts
        avg_length: Average token length of texts
        pass_window: If True, skip window check
        num_shots: Number of shots

    Returns:
        Tuple of (processed prefix, actual number of shots)
    """
    if pass_window:
        return prefix, num_shots
    max_length = model.llm_engine.model_config.max_model_len
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


def compute_prefix_loss(output: RequestOutput, prefix_token_length: int) -> float:
    """Compute negative mean log-likelihood excluding prefix tokens.

    Shared by ReCaLL and CON-ReCaLL methods.

    Args:
        output: Model output
        prefix_token_length: Number of prefix tokens to exclude

    Returns:
        Negative mean log-likelihood (loss)
    """
    token_log_probs = []
    for i, prompt_logprob in enumerate(output.prompt_logprobs):
        if i < prefix_token_length:
            continue
        if prompt_logprob is None:
            continue
        token_log_probs.append(list(prompt_logprob.values())[0].logprob)
    return -np.mean(token_log_probs)
