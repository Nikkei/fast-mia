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

"""Dataset Inference statistical testing and chunk-scaling utilities."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

VALID_ALTERNATIVES = ("greater", "less", "two-sided")
DEFAULT_CANDIDATE_CHUNK_SIZES = (2, 5, 10, 20, 50, 100, 200, 500)


def compute_cohens_d(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """Compute pooled-variance Cohen's d effect size (member minus non-member)."""
    n1 = member_scores.size
    n0 = nonmember_scores.size
    if n1 == 0 or n0 == 0 or (n1 + n0) <= 2:
        return float("nan")

    mean_diff = float(np.mean(member_scores) - np.mean(nonmember_scores))
    var1 = float(np.var(member_scores, ddof=1)) if n1 > 1 else 0.0
    var0 = float(np.var(nonmember_scores, ddof=1)) if n0 > 1 else 0.0
    pooled_var = ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)

    if pooled_var <= 0.0:
        if mean_diff == 0.0:
            return 0.0
        return float(np.copysign(np.inf, mean_diff))

    return float(mean_diff / np.sqrt(pooled_var))


def _welch_t_test(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    alternative: str = "greater",
) -> tuple[float, float]:
    """Run Welch's unequal-variance two-sample t-test with zero-variance guards."""
    if member_scores.size < 2 or nonmember_scores.size < 2:
        return float("nan"), float("nan")

    var1 = float(np.var(member_scores, ddof=1))
    var0 = float(np.var(nonmember_scores, ddof=1))
    mean_diff = float(np.mean(member_scores) - np.mean(nonmember_scores))

    if var1 == 0.0 and var0 == 0.0:
        if mean_diff == 0.0:
            return 0.0, 1.0
        t_stat = float(np.copysign(np.inf, mean_diff))
        if alternative == "two-sided":
            return t_stat, 0.0
        if alternative == "greater":
            return t_stat, (0.0 if mean_diff > 0 else 1.0)
        return t_stat, (0.0 if mean_diff < 0 else 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = stats.ttest_ind(
            member_scores,
            nonmember_scores,
            equal_var=False,
            alternative=alternative,
        )
    return float(res.statistic), float(res.pvalue)


def _mann_whitney_test(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    alternative: str = "greater",
) -> tuple[float, float]:
    """Run Mann-Whitney U rank test."""
    if member_scores.size < 1 or nonmember_scores.size < 1:
        return float("nan"), float("nan")

    res = stats.mannwhitneyu(member_scores, nonmember_scores, alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def _permutation_test_mean_diff(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    alternative: str = "greater",
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """Vectorized Monte Carlo permutation test on difference in sample means."""
    n1 = member_scores.size
    n0 = nonmember_scores.size
    if n1 < 1 or n0 < 1:
        return float("nan")

    combined = np.concatenate([member_scores, nonmember_scores])
    total_sum = float(np.sum(combined))
    obs_diff = float(np.mean(member_scores) - np.mean(nonmember_scores))

    rng = np.random.default_rng(seed)
    n_total = n1 + n0
    batch_size = min(n_permutations, 512)
    exceed_count = 0
    remaining = n_permutations

    while remaining > 0:
        cur = min(batch_size, remaining)
        # Draw random permutations via argsort of uniform random matrix
        perm_indices = np.argpartition(rng.random((cur, n_total)), n1 - 1, axis=1)[
            :, :n1
        ]
        perm_member_sums = np.sum(combined[perm_indices], axis=1)
        perm_nonmember_sums = total_sum - perm_member_sums
        perm_diffs = (perm_member_sums / n1) - (perm_nonmember_sums / n0)

        if alternative == "greater":
            exceed_count += int(np.sum(perm_diffs >= obs_diff - 1e-12))
        elif alternative == "less":
            exceed_count += int(np.sum(perm_diffs <= obs_diff + 1e-12))
        else:
            exceed_count += int(np.sum(np.abs(perm_diffs) >= abs(obs_diff) - 1e-12))

        remaining -= cur

    return float((exceed_count + 1) / (n_permutations + 1))


def compute_two_sample_tests(
    member_scores: list[float] | np.ndarray,
    nonmember_scores: list[float] | np.ndarray,
    alternative: str = "greater",
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute two-sample statistical tests and effect sizes between two score sets.

    Args:
        member_scores: Scores for member samples (label == 1)
        nonmember_scores: Scores for non-member samples (label == 0)
        alternative: Alternative hypothesis ("greater", "less", or "two-sided")
        n_permutations: Number of Monte Carlo permutations
        seed: Random seed for permutation testing

    Returns:
        Dictionary with sample counts, means, Cohen's d, test statistics, and p-values
    """
    if alternative not in VALID_ALTERNATIVES:
        raise ValueError(
            f"Unsupported alternative '{alternative}'. "
            f"Choose one of: {', '.join(VALID_ALTERNATIVES)}."
        )
    if n_permutations <= 0:
        raise ValueError(
            f"n_permutations must be a positive integer, got {n_permutations}."
        )

    mem_arr = np.asarray(member_scores, dtype=float)
    non_arr = np.asarray(nonmember_scores, dtype=float)
    mem_arr = mem_arr[~np.isnan(mem_arr)]
    non_arr = non_arr[~np.isnan(non_arr)]

    mean_member = float(np.mean(mem_arr)) if mem_arr.size > 0 else float("nan")
    mean_nonmember = float(np.mean(non_arr)) if non_arr.size > 0 else float("nan")
    cohens_d = compute_cohens_d(mem_arr, non_arr)
    welch_t, welch_p = _welch_t_test(mem_arr, non_arr, alternative=alternative)
    mwu_stat, mwu_p = _mann_whitney_test(mem_arr, non_arr, alternative=alternative)
    perm_p = _permutation_test_mean_diff(
        mem_arr,
        non_arr,
        alternative=alternative,
        n_permutations=n_permutations,
        seed=seed,
    )

    return {
        "n_members": int(mem_arr.size),
        "n_nonmembers": int(non_arr.size),
        "mean_member": mean_member,
        "mean_nonmember": mean_nonmember,
        "cohens_d": cohens_d,
        "welch_t_stat": welch_t,
        "welch_p_value": welch_p,
        "mannwhitney_u_stat": mwu_stat,
        "mannwhitney_p_value": mwu_p,
        "permutation_p_value": perm_p,
    }


def compute_group_tests(
    scores: list[float] | np.ndarray,
    labels: list[int],
    groups: list[str],
    method_name: str,
    alternative: str = "greater",
    n_permutations: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run group-level Dataset Inference tests.

    Supports two common evaluation layouts:
    1. Within-group paired layout: group ``g`` contains both member (1) and
       non-member (0) chunks (e.g., domain-stratified benchmarks). Tests
       member chunks in ``g`` against non-member chunks in ``g``.
    2. Suspect-document vs reference-pool layout: group ``g`` contains only
       member chunks (1) (e.g., a book or repository split into chunks) while
       other samples form the non-member reference pool (0). Tests chunks in
       ``g`` against the global non-member pool.
    """
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    groups_arr = np.asarray([str(g) for g in groups])

    global_nonmembers = scores_arr[labels_arr == 0]
    distinct_groups = list(dict.fromkeys(groups_arr.tolist()))
    rows: list[dict[str, Any]] = []

    for idx, group_id in enumerate(distinct_groups):
        mask = groups_arr == group_id
        group_scores = scores_arr[mask]
        group_labels = labels_arr[mask]

        group_members = group_scores[group_labels == 1]
        group_nonmembers = group_scores[group_labels == 0]

        if group_members.size > 0 and group_nonmembers.size > 0:
            comparison_mode = "within_group"
            ref_scores = group_nonmembers
            target_scores = group_members
        elif group_members.size > 0 and global_nonmembers.size > 0:
            comparison_mode = "vs_global_nonmembers"
            ref_scores = global_nonmembers
            target_scores = group_members
        else:
            # Pure non-member reference group; skip suspect-membership test row
            continue

        stats_row = compute_two_sample_tests(
            target_scores,
            ref_scores,
            alternative=alternative,
            n_permutations=n_permutations,
            seed=seed + idx,
        )
        rows.append(
            {
                "method": method_name,
                "group": group_id,
                "comparison_mode": comparison_mode,
                **stats_row,
            }
        )

    return rows


def compute_chunk_scaling(
    member_scores: list[float] | np.ndarray,
    nonmember_scores: list[float] | np.ndarray,
    method_name: str,
    chunk_sizes: list[int] | None = None,
    n_bootstrap: int = 50,
    alternative: str = "greater",
    alpha: float = 0.05,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate how statistical significance scales with number of chunks ``m``."""
    mem_arr = np.asarray(member_scores, dtype=float)
    non_arr = np.asarray(nonmember_scores, dtype=float)
    mem_arr = mem_arr[~np.isnan(mem_arr)]
    non_arr = non_arr[~np.isnan(non_arr)]

    max_chunks = min(mem_arr.size, non_arr.size)
    if max_chunks < 2:
        return []

    raw_sizes = (
        list(chunk_sizes)
        if chunk_sizes is not None
        else list(DEFAULT_CANDIDATE_CHUNK_SIZES)
    )
    valid_sizes = sorted({int(m) for m in raw_sizes if 2 <= int(m) <= max_chunks})
    if max_chunks not in valid_sizes:
        valid_sizes.append(max_chunks)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for m in valid_sizes:
        p_values: list[float] = []
        d_values: list[float] = []
        replace_mem = m >= mem_arr.size
        replace_non = m >= non_arr.size

        for _ in range(max(1, n_bootstrap)):
            sub_mem = rng.choice(mem_arr, size=m, replace=replace_mem)
            sub_non = rng.choice(non_arr, size=m, replace=replace_non)
            _, p_val = _welch_t_test(sub_mem, sub_non, alternative=alternative)
            d_val = compute_cohens_d(sub_mem, sub_non)
            if not np.isnan(p_val):
                p_values.append(p_val)
            if np.isfinite(d_val):
                d_values.append(d_val)

        if not p_values:
            continue

        p_arr = np.clip(np.asarray(p_values, dtype=float), 1e-300, 1.0)
        neg_log10_p = -np.log10(p_arr)
        rows.append(
            {
                "method": method_name,
                "chunk_size": m,
                "mean_p_value": float(np.mean(p_arr)),
                "median_p_value": float(np.median(p_arr)),
                "mean_neg_log10_p": float(np.mean(neg_log10_p)),
                "q25_neg_log10_p": float(np.percentile(neg_log10_p, 25)),
                "q75_neg_log10_p": float(np.percentile(neg_log10_p, 75)),
                "power_at_alpha": float(np.mean(p_arr < alpha)),
                "mean_cohens_d": (
                    float(np.mean(d_values)) if d_values else float("nan")
                ),
            }
        )

    return rows


def evaluate_dataset_inference(
    detailed_results: list[dict[str, Any]],
    labels: list[int],
    groups: list[str] | None = None,
    di_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Dataset Inference statistical evaluation across all methods.

    Args:
        detailed_results: List of per-method dicts with ``method_name`` and ``scores``
        labels: Ground-truth binary membership labels (1: member, 0: non-member)
        groups: Optional per-sample group identifiers
        di_config: Optional ``dataset_inference`` configuration block

    Returns:
        Tuple of ``(dataset_inference_df, group_inference_df, chunk_scaling_df)``
    """
    cfg = di_config or {}
    if not cfg.get("enabled", True):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if len(set(labels)) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    alternative = str(cfg.get("alternative", "greater"))
    n_permutations = int(cfg.get("n_permutations", 1000))
    chunk_sizes = cfg.get("chunk_sizes")
    n_bootstrap = int(cfg.get("n_bootstrap", 50))
    alpha = float(cfg.get("alpha", 0.05))
    seed = int(cfg.get("seed", 42))

    labels_arr = np.asarray(labels, dtype=int)
    overall_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []

    for idx, item in enumerate(detailed_results):
        method_name = str(item["method_name"])
        scores_arr = np.asarray(item["scores"], dtype=float)
        mem_scores = scores_arr[labels_arr == 1]
        non_scores = scores_arr[labels_arr == 0]

        two_sample = compute_two_sample_tests(
            mem_scores,
            non_scores,
            alternative=alternative,
            n_permutations=n_permutations,
            seed=seed + idx,
        )
        overall_rows.append({"method": method_name, **two_sample})

        scaling_rows.extend(
            compute_chunk_scaling(
                mem_scores,
                non_scores,
                method_name=method_name,
                chunk_sizes=chunk_sizes,
                n_bootstrap=n_bootstrap,
                alternative=alternative,
                alpha=alpha,
                seed=seed + idx,
            )
        )

        if groups is not None and len(groups) == len(labels):
            group_rows.extend(
                compute_group_tests(
                    scores_arr,
                    labels,
                    groups,
                    method_name=method_name,
                    alternative=alternative,
                    n_permutations=n_permutations,
                    seed=seed + 1000 + idx,
                )
            )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(scaling_rows),
    )
