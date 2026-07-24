import pandas as pd

from src.utils import format_results_df


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
