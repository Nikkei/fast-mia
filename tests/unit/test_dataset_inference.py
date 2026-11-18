import json
import math
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data_loader import DataLoader
from src.dataset_inference import (
    compute_chunk_scaling,
    compute_cohens_d,
    compute_group_tests,
    compute_two_sample_tests,
    evaluate_dataset_inference,
)
from src.evaluator import Evaluator
from src.result_writer import ResultWriter
from src.visualizer import Visualizer


class TestTwoSampleTests:
    def test_clearly_separated_distributions_give_small_p_and_positive_d(self):
        rng = np.random.default_rng(0)
        members = rng.normal(loc=2.0, scale=0.3, size=30)
        nonmembers = rng.normal(loc=0.0, scale=0.3, size=30)

        res = compute_two_sample_tests(
            members, nonmembers, alternative="greater", n_permutations=1000, seed=7
        )
        assert res["n_members"] == 30
        assert res["n_nonmembers"] == 30
        assert res["cohens_d"] > 4.0
        assert res["welch_p_value"] < 1e-6
        assert res["mannwhitney_p_value"] < 1e-6
        assert res["permutation_p_value"] < 0.01

    def test_identical_distributions_give_large_p_and_near_zero_d(self):
        scores_a = [0.1, 0.2, 0.3, 0.4, 0.5]
        scores_b = [0.1, 0.2, 0.3, 0.4, 0.5]

        res = compute_two_sample_tests(
            scores_a, scores_b, alternative="two-sided", n_permutations=500, seed=42
        )
        assert abs(res["cohens_d"]) < 1e-12
        assert abs(res["welch_p_value"] - 1.0) < 1e-12
        assert abs(res["mannwhitney_p_value"] - 1.0) < 1e-12
        assert res["permutation_p_value"] > 0.5

    def test_zero_variance_edge_case_handled_cleanly(self):
        res = compute_two_sample_tests(
            [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], alternative="greater"
        )
        assert math.isinf(res["cohens_d"]) and res["cohens_d"] > 0
        assert res["welch_p_value"] == 0.0

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="Unsupported alternative"):
            compute_two_sample_tests([1.0, 2.0], [0.0, 1.0], alternative="invalid")

    def test_cohens_d_too_few_samples_returns_nan(self):
        assert math.isnan(compute_cohens_d(np.array([1.0]), np.array([0.0])))


class TestGroupInferenceAndChunkScaling:
    def test_group_tests_support_within_group_and_suspect_doc_modes(self):
        # book_a: suspect member book (only label=1 chunks) -> vs_global_nonmembers
        # domain_b: paired benchmark group (both label=1 and label=0 chunks) -> within_group
        # ref_pool: pure non-member reference chunks -> skipped as target
        scores = [0.9, 0.85, 0.95, 0.88, 0.8, 0.75, 0.2, 0.15, 0.1, 0.12, 0.18]
        labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        groups = [
            "book_a",
            "book_a",
            "book_a",
            "book_a",
            "domain_b",
            "domain_b",
            "domain_b",
            "domain_b",
            "ref_pool",
            "ref_pool",
            "ref_pool",
        ]

        rows = compute_group_tests(
            scores, labels, groups, method_name="loss", n_permutations=500, seed=11
        )
        by_group = {row["group"]: row for row in rows}
        assert set(by_group.keys()) == {"book_a", "domain_b"}
        assert by_group["book_a"]["comparison_mode"] == "vs_global_nonmembers"
        assert by_group["book_a"]["welch_p_value"] < 0.01
        assert by_group["domain_b"]["comparison_mode"] == "within_group"
        assert by_group["domain_b"]["cohens_d"] > 1.0

    def test_chunk_scaling_power_increases_with_chunk_count(self):
        rng = np.random.default_rng(123)
        members = rng.normal(loc=0.8, scale=0.5, size=60)
        nonmembers = rng.normal(loc=0.0, scale=0.5, size=60)

        rows = compute_chunk_scaling(
            members,
            nonmembers,
            method_name="loss",
            chunk_sizes=[3, 10, 40],
            n_bootstrap=60,
            seed=99,
        )
        df = pd.DataFrame(rows)
        assert list(df["chunk_size"]) == [3, 10, 40, 60]
        # Detection power and -log10(p) should be higher at m=40 than m=3
        power_small = df.loc[df["chunk_size"] == 3, "power_at_alpha"].iloc[0]
        power_large = df.loc[df["chunk_size"] == 40, "power_at_alpha"].iloc[0]
        assert power_large >= power_small
        assert power_large > 0.8

    def test_end_to_end_evaluator_writer_and_visualizer_with_groups(
        self, tmp_path: Path
    ):
        csv_path = tmp_path / "grouped.csv"
        pd.DataFrame(
            {
                "text": [f"sample text {i}" for i in range(8)],
                "label": [1, 1, 1, 1, 0, 0, 0, 0],
                "doc_id": [
                    "suspect_1",
                    "suspect_1",
                    "suspect_2",
                    "suspect_2",
                    "ref",
                    "ref",
                    "ref",
                    "ref",
                ],
            }
        ).to_csv(csv_path, index=False)

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            f"""
model:
  model_id: "gpt2"
data:
  data_path: "{csv_path}"
  format: "csv"
  text_column: "text"
  label_column: "label"
  group_column: "doc_id"
  text_length: 10
methods:
  - type: "loss"
dataset_inference:
  enabled: true
  n_permutations: 200
  chunk_sizes: [2, 4]
  n_bootstrap: 20
"""
        )
        config = Config(cfg_path)
        loader = DataLoader(
            data_path=csv_path,
            data_format="csv",
            text_column="text",
            label_column="label",
            group_column="doc_id",
        )
        mock_model_loader = mock.MagicMock()
        method = mock.MagicMock()
        method.method_name = "loss"
        method.requires_labels = False
        method.requires_tokenizer = False
        method.requires_sampling_params = True
        method.run.return_value = [0.95, 0.91, 0.88, 0.92, 0.12, 0.15, 0.10, 0.18]

        evaluator = Evaluator(loader, mock_model_loader, [method])
        eval_res = evaluator.evaluate(config)

        assert not eval_res.dataset_inference_df.empty
        assert not eval_res.group_inference_df.empty
        assert not eval_res.chunk_scaling_df.empty
        assert eval_res.data_stats["num_groups"] == 3

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        writer = ResultWriter(run_dir, config, datetime.now())
        saved = writer.save_all(
            results_df=eval_res.results_df,
            results=eval_res.detailed_results,
            labels=eval_res.labels,
            cache_stats=eval_res.cache_stats,
            data_stats=eval_res.data_stats,
            end_time=datetime.now(),
            dataset_inference_df=eval_res.dataset_inference_df,
            group_inference_df=eval_res.group_inference_df,
            chunk_scaling_df=eval_res.chunk_scaling_df,
        )
        assert saved["dataset_inference"].exists()
        assert saved["group_inference"].exists()
        assert saved["chunk_scaling"].exists()

        meta = json.loads(saved["metadata"].read_text())
        assert "dataset_inference" in meta
        report = saved["report"].read_text()
        assert "DATASET INFERENCE" in report
        assert "GROUP-LEVEL DATASET INFERENCE" in report

        vis = Visualizer(run_dir)
        figs = vis.generate_all_plots(
            eval_res.detailed_results,
            eval_res.labels,
            chunk_scaling_df=eval_res.chunk_scaling_df,
        )
        assert "chunk_scaling" in figs
        assert figs["chunk_scaling"].exists()

    def test_disabled_dataset_inference_returns_empty_dfs(self):
        di_df, grp_df, scl_df = evaluate_dataset_inference(
            [{"method_name": "loss", "scores": [0.9, 0.1]}],
            labels=[1, 0],
            di_config={"enabled": False},
        )
        assert di_df.empty
        assert grp_df.empty
        assert scl_df.empty
