import math

import pandas as pd

from src.utils import fix_seed, format_results_df, get_metrics


class TestFormatResultsDf:
    def test_formats_metrics_as_percentages(self):
        df = pd.DataFrame(
            {
                "method": ["loss"],
                "auroc": [0.853],
                "fpr95": [0.5],
                "tpr05": [float("nan")],
            }
        )
        formatted = format_results_df(df)
        assert formatted.loc[0, "auroc"] == "85.3%"
        assert formatted.loc[0, "fpr95"] == "50.0%"
        assert formatted.loc[0, "tpr05"] == "nan%"
        assert formatted.loc[0, "method"] == "loss"

    def test_original_df_not_modified(self):
        df = pd.DataFrame({"method": ["loss"], "auroc": [0.5]})
        format_results_df(df)
        assert df.loc[0, "auroc"] == 0.5


class TestGetMetrics:
    def test_perfectly_separable_scores(self):
        # Members (label 1) all score higher than non-members (label 0).
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        auroc, fpr95, tpr05 = get_metrics(scores, labels)
        assert auroc == 1.0
        assert fpr95 == 0.0
        assert tpr05 == 1.0

    def test_single_class_returns_nan(self):
        # Metrics are undefined when only one label is present.
        auroc, fpr95, tpr05 = get_metrics([0.1, 0.9], [1, 1])
        assert math.isnan(auroc)
        assert math.isnan(fpr95)
        assert math.isnan(tpr05)

    def test_returns_three_finite_metrics_for_mixed_scores(self):
        scores = [0.1, 0.6, 0.4, 0.9]
        labels = [0, 1, 0, 1]
        auroc, fpr95, tpr05 = get_metrics(scores, labels)
        assert 0.0 <= auroc <= 1.0
        assert 0.0 <= fpr95 <= 1.0
        assert 0.0 <= tpr05 <= 1.0


class TestFixSeed:
    def test_reproducible_numpy_draw(self):
        import numpy as np

        fix_seed(123)
        first = np.random.rand(3).tolist()
        fix_seed(123)
        second = np.random.rand(3).tolist()
        assert first == second
