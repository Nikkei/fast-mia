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
from vllm.tokenizers import TokenizerLike


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
    tokenizer: TokenizerLike,
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
    # prefix[-0:] would return the whole list, so handle zero shots explicitly
    if max_shots == 0:
        return [], 0
    # Truncate the prefix to include only the maximum number of shots
    truncated_prefix = prefix[-max_shots:]
    num_shots = max_shots
    return truncated_prefix, num_shots


def get_separator_token(tokenizer: TokenizerLike) -> str:
    """Return a boundary separator token that avoids merging prefix and target.

    Uses ``tokenizer.bos_token`` when configured as a non-empty string, and
    falls back to ``"\\n"`` for BOS-less tokenizers so adjacent characters do
    not glue across the prefix/target boundary (and are not stripped when
    ``space_delimited_language`` is ``False``).
    """
    bos_token = getattr(tokenizer, "bos_token", None)
    if isinstance(bos_token, str) and bos_token:
        return bos_token
    return "\n"


def get_bos_token_id(tokenizer: TokenizerLike) -> int | None:
    """Extract an integer ``bos_token_id`` from a tokenizer if present."""
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if isinstance(bos_id, int) and not isinstance(bos_id, bool):
        return bos_id
    return None


def select_prefix_candidates(pool_texts: list[str], num_shots: int) -> list[str]:
    """Select prefix shot candidates while keeping room to avoid self-leakage.

    When more than one distinct candidate exists in ``pool_texts`` and
    ``num_shots`` covers all distinct texts, capping at ``len(unique) - 1``
    ensures every sample (whether inside or outside the pool) receives the same
    number of shots after excluding itself.
    """
    unique_count = len(set(pool_texts))
    if unique_count > 1:
        effective_shots = min(int(num_shots), unique_count - 1)
    else:
        effective_shots = int(num_shots)
    return extract_prefix(pool_texts, effective_shots)


def build_conditional_texts(
    texts: list[str],
    pool_texts: list[str],
    processed_prefix: list[str],
    model: LLM,
    tokenizer: TokenizerLike,
    avg_length: int,
    pass_window: bool,
    sep_token: str,
) -> list[str]:
    """Construct per-sample conditional prompts without self-reference leakage.

    If a target sample matches one of the selected shots in ``processed_prefix``,
    that shot is removed for this sample and backfilled from ``pool_texts``
    (respecting ``max_model_len`` when ``pass_window`` is ``False``).
    """
    if not processed_prefix:
        return list(texts)

    max_length = getattr(
        getattr(getattr(model, "llm_engine", None), "model_config", None),
        "max_model_len",
        None,
    )
    enforce_window = (
        not pass_window
        and isinstance(max_length, int)
        and not isinstance(max_length, bool)
    )

    base_prefix = "".join(processed_prefix)
    conditional_texts: list[str] = []

    for text in texts:
        if all(shot != text for shot in processed_prefix):
            conditional_texts.append(base_prefix + sep_token + text)
            continue

        filtered_shots = [shot for shot in processed_prefix if shot != text]
        needed = len(processed_prefix) - len(filtered_shots)
        backfill_pool = [
            candidate
            for candidate in pool_texts
            if candidate != text and candidate not in filtered_shots
        ]

        sample_shots = list(filtered_shots)
        if needed > 0 and backfill_pool:
            extra_candidates = extract_prefix(backfill_pool, needed)
            if enforce_window:
                used_tokens = (
                    sum(len(tokenizer.encode(s)) for s in sample_shots) + avg_length
                )
                for extra in extra_candidates:
                    extra_tokens = len(tokenizer.encode(extra))
                    if used_tokens + extra_tokens <= max_length:
                        sample_shots.append(extra)
                        used_tokens += extra_tokens
            else:
                sample_shots.extend(extra_candidates)

        if sample_shots:
            conditional_texts.append("".join(sample_shots) + sep_token + text)
        else:
            conditional_texts.append(text)

    return conditional_texts


def compute_conditional_prefix_length(
    cond_output: RequestOutput,
    uncond_output: RequestOutput,
    bos_token_id: int | None = None,
) -> int:
    """Compute the number of leading prefix tokens to exclude from loss.

    Aligns ``cond_output.prompt_token_ids`` against ``uncond_output.prompt_token_ids``
    produced by vLLM rather than tokenizing the prefix in isolation, avoiding
    both missing-separator off-by-one errors and BPE boundary merges.
    """
    cond_ids = list(cond_output.prompt_token_ids)
    uncond_ids = list(uncond_output.prompt_token_ids)
    if not cond_ids or not uncond_ids:
        return 0

    if bos_token_id is not None and uncond_ids[0] == bos_token_id:
        target_ids = uncond_ids[1:]
    else:
        target_ids = uncond_ids

    if not target_ids:
        return len(cond_ids)

    if len(cond_ids) >= len(target_ids) and cond_ids[-len(target_ids) :] == target_ids:
        return len(cond_ids) - len(target_ids)

    if len(target_ids) > 1 and cond_ids[-1] == target_ids[-1]:
        match_len = 0
        max_check = min(len(cond_ids), len(target_ids))
        while (
            match_len < max_check
            and cond_ids[-1 - match_len] == target_ids[-1 - match_len]
        ):
            match_len += 1
        unaligned_target = len(target_ids) - match_len
        # When BPE merges across the prefix/target boundary, the first target
        # token(s) map to at least 1 boundary token in cond_ids (up to
        # unaligned_target tokens).
        remaining_cond = len(cond_ids) - match_len
        boundary_tokens = min(unaligned_target, remaining_cond)
        return max(0, len(cond_ids) - match_len - boundary_tokens)

    return max(0, len(cond_ids) - len(target_ids))


def compute_prefix_loss(output: RequestOutput, prefix_token_length: int) -> float:
    """Compute negative mean log-likelihood excluding prefix tokens.

    Shared by ReCaLL and CON-ReCaLL methods.

    Args:
        output: Model output
        prefix_token_length: Number of prefix tokens to exclude

    Returns:
        Negative mean log-likelihood (loss)
    """
    # Look up each logprob by the actual prompt token ID (see
    # BaseMethod._extract_token_log_probs for the rationale)
    token_log_probs = []
    token_logprob_pairs = zip(
        output.prompt_token_ids, output.prompt_logprobs, strict=True
    )
    for i, (token_id, prompt_logprob) in enumerate(token_logprob_pairs):
        if i < prefix_token_length:
            continue
        if prompt_logprob is None:
            continue
        token_log_probs.append(prompt_logprob[token_id].logprob)
    if not token_log_probs:
        return 0.0
    return float(-np.mean(token_log_probs))
