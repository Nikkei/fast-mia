from unittest import mock

import pytest

from src.methods.blind import BlindMethod
from src.utils import get_metrics


class TestBlindInit:
    def test_defaults(self):
        method = BlindMethod()
        assert method.method_name == "blind_tfidf"
        assert method.features == "tfidf"
        assert method.requires_labels is True
        assert method.requires_sampling_params is False

    def test_surface_feature_mode_does_not_require_labels(self):
        method = BlindMethod({"features": "digit_ratio"})
        assert method.method_name == "blind_digit_ratio"
        assert method.requires_labels is False

    @pytest.mark.parametrize(
        "bad_params",
        [
            {"features": "unknown"},
            {"n_splits": 1},
            {"max_features": 0},
            {"c": 0.0},
        ],
    )
    def test_invalid_params_raise(self, bad_params):
        with pytest.raises(ValueError):
            BlindMethod(bad_params)


class TestBlindScoring:
    def test_tfidf_detects_temporal_distribution_shift_with_zero_model_calls(self):
        texts = [
            "Archived encyclopedia entry published in 2019 about classical physics.",
            "Historical biography from 2018 describing early steam locomotives.",
            "Reference article written in 2019 covering medieval architecture.",
            "2018 encyclopedia summary of continental plate tectonics.",
            "Breaking report in 2025 announcing new post-cutoff election results.",
            "2025 match recap and updated tournament standings from March 2025.",
            "Recent 2025 software release notes and 2025 security advisories.",
            "2025 quarterly economic forecast published in February 2025.",
        ]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        model = mock.MagicMock()

        method = BlindMethod({"features": "tfidf", "n_splits": 4, "seed": 42})
        scores = method.run(texts=texts, labels=labels, model=model)

        assert len(scores) == len(texts)
        model.generate.assert_not_called()
        auroc, _, _ = get_metrics(scores, labels)
        assert auroc > 0.9

    def test_surface_features_and_non_space_delimited_normalization(self):
        method = BlindMethod({"features": "length"})
        scores = method.run(
            texts=["a b c", "d e"],
            data_config={"space_delimited_language": False},
        )
        assert scores == [3.0, 2.0]

        digit_method = BlindMethod({"features": "digit_ratio"})
        assert digit_method.run(texts=["2025", "abcd", ""]) == pytest.approx(
            [1.0, 0.0, 0.0]
        )

    def test_process_output_raises_for_tfidf_and_works_for_surface_mode(self):
        out = mock.MagicMock()
        out.prompt = "hello 12"

        with pytest.raises(NotImplementedError):
            BlindMethod({"features": "tfidf"}).process_output(out)

        assert BlindMethod({"features": "length"}).process_output(out) == 8.0

    def test_tfidf_missing_or_insufficient_labels_raises(self):
        method = BlindMethod({"features": "tfidf"})
        with pytest.raises(ValueError, match="requires 'labels'"):
            method.run(texts=["a", "b", "c", "d"], labels=None)

        with pytest.raises(ValueError, match="at least 2 samples"):
            method.run(texts=["a", "b"], labels=[1, 0])

    def test_benchmark_config_loads_cleanly(self):
        from pathlib import Path

        from src.config import Config
        from src.methods.factory import MethodFactory

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "benchmark_blind.yaml"
        config = Config(cfg_path)
        methods = MethodFactory.create_methods(config.methods)
        assert any(isinstance(m, BlindMethod) for m in methods)

