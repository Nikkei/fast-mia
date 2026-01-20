import pytest
from unittest import mock
import pandas as pd
import math

from src.evaluator import Evaluator, EvaluationResult
from src.config import Config
from src.methods.factory import MethodFactory
from src.utils import get_metrics

class TestEvaluator:
    @pytest.fixture
    def config(self):
        path = __import__("pathlib").Path(__file__).parent.parent / "fixtures" / "valid_config.yaml"
        return Config(path)

    @pytest.fixture
    def mock_data_loader(self):
        mock_loader = mock.MagicMock()
        # Sample data (4 items)
        texts = [
            "This is a sample text.",
            "This is another sample text.",
            "Here is also a sample text.",
            "This is a longer sample text. There are many words. This is used for split test."
        ]
        labels = [1, 0, 1, 0]
        mock_loader.get_data.return_value = (texts, labels)
        return mock_loader

    @pytest.fixture
    def mock_model_loader(self):
        mock_loader = mock.MagicMock()
        mock_loader.model = mock.MagicMock()
        mock_loader.tokenizer = mock.MagicMock()
        mock_loader.get_sampling_params.return_value = mock.MagicMock()
        mock_loader.get_lora_request.return_value = None
        return mock_loader

    @pytest.fixture
    def mock_methods(self):
        # Prepare two dummy methods
        method1 = mock.MagicMock()
        method1.method_name = "loss"
        method1.run.return_value = [0.9, 0.1, 0.8, 0.2]
        method2 = mock.MagicMock()
        method2.method_name = "zlib"
        method2.run.return_value = [0.8, 0.2, 0.7, 0.3]
        return [method1, method2]

    @pytest.fixture
    def mock_methods_all(self):
        # Prepare dummy methods for each attack type
        method_loss = mock.MagicMock()
        method_loss.method_name = "loss"
        method_loss.run.return_value = [0.9, 0.1, 0.8, 0.2]

        method_lower = mock.MagicMock()
        method_lower.method_name = "lower"
        method_lower.run.return_value = [0.8, 0.2, 0.7, 0.3]

        method_zlib = mock.MagicMock()
        method_zlib.method_name = "zlib"
        method_zlib.run.return_value = [0.7, 0.3, 0.6, 0.4]

        method_mink = mock.MagicMock()
        method_mink.method_name = "mink_0.1"
        method_mink.run.return_value = [0.6, 0.4, 0.5, 0.5]

        method_pac = mock.MagicMock()
        method_pac.method_name = "pac"
        method_pac.run.return_value = [0.5, 0.5, 0.4, 0.6]

        method_recall = mock.MagicMock()
        method_recall.method_name = "recall"
        # Recall also has different run arguments in Evaluator, but run.return_value is fine for mock
        method_recall.run.return_value = [0.2, 0.8, 0.1, 0.9]

        return [
            method_loss, method_lower, method_zlib, method_mink,
            method_pac, method_recall
        ]

    def test_evaluate_normal(self, config, mock_data_loader, mock_model_loader, mock_methods):
        evaluator = Evaluator(mock_data_loader, mock_model_loader, mock_methods)
        result = evaluator.evaluate(config)
        # Check EvaluationResult structure
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.results_df, pd.DataFrame)
        assert set(result.results_df.columns) == {"method", "auroc", "fpr95", "tpr05"}
        assert list(result.results_df["method"]) == ["loss", "zlib"]
        # Check that metric values are in percentage format
        for col in ["auroc", "fpr95", "tpr05"]:
            for val in result.results_df[col]:
                assert isinstance(val, str) and val.endswith("%")
        # Check detailed_results
        assert len(result.detailed_results) == 2
        assert result.detailed_results[0]["method_name"] == "loss"
        # Check data_stats
        assert result.data_stats["num_samples"] == 4

    def test_evaluate_metrics_correctness(self, config, mock_data_loader, mock_model_loader):
        # Check that when one method has perfect score and label match
        method = mock.MagicMock()
        method.method_name = "loss"
        method.run.return_value = [1, 0, 1, 0]  # 完全にラベルと一致
        evaluator = Evaluator(mock_data_loader, mock_model_loader, [method])
        result = evaluator.evaluate(config)
        # AUROCが100%になること
        assert result.results_df.loc[0, "auroc"] == "100.0%"

    def test_evaluate_multiple_methods(self, config, mock_data_loader, mock_model_loader):
        # Check that multiple method results are correctly aggregated
        method1 = mock.MagicMock()
        method1.method_name = "loss"
        method1.run.return_value = [0.9, 0.1, 0.8, 0.2]
        method2 = mock.MagicMock()
        method2.method_name = "zlib"
        method2.run.return_value = [0.8, 0.2, 0.7, 0.3]
        evaluator = Evaluator(mock_data_loader, mock_model_loader, [method1, method2])
        result = evaluator.evaluate(config)
        assert list(result.results_df["method"]) == ["loss", "zlib"]

    def test_evaluate_error_handling(self, config, mock_data_loader, mock_model_loader):
        # Check that when a method throws an exception
        method = mock.MagicMock()
        method.method_name = "loss"
        method.run.side_effect = RuntimeError("method error")
        evaluator = Evaluator(mock_data_loader, mock_model_loader, [method])
        with pytest.raises(RuntimeError, match="method error"):
            evaluator.evaluate(config)

    def test_evaluate_all_methods(self, config, mock_data_loader, mock_model_loader, mock_methods_all):
        evaluator = Evaluator(mock_data_loader, mock_model_loader, mock_methods_all)
        result = evaluator.evaluate(config)
        # Check that all method_name are included in the result
        expected_methods = [
            "loss", "lower", "zlib", "mink_0.1", "pac", "recall"
        ]
        assert set(result.results_df["method"]) == set(expected_methods)
        # Check that metric values are in percentage format
        for col in ["auroc", "fpr95", "tpr05"]:
            for val in result.results_df[col]:
                assert isinstance(val, str) and val.endswith("%")

class TestMetrics:
    def test_get_metrics_perfect(self):
        # Check that when all scores are perfectly separated
        scores = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        auroc, fpr95, tpr05 = get_metrics(scores, labels)
        assert auroc == 1.0
        # Since all scores are perfectly separated, FPR95=0, TPR05=1
        assert fpr95 == 0.0
        assert tpr05 == 1.0

    def test_get_metrics_random(self):
        # Check that when scores are random
        scores = [0.1, 0.4, 0.35, 0.8]
        labels = [0, 0, 1, 1]
        auroc, fpr95, tpr05 = get_metrics(scores, labels)
        # AUROC is 0.75 (same as sklearn example)
        assert abs(auroc - 0.75) < 1e-6
        # FPR95, TPR05 are in the range 0<=x<=1
        assert 0.0 <= fpr95 <= 1.0
        assert 0.0 <= tpr05 <= 1.0

    def test_get_metrics_all_same_label(self):
        # Check that when all labels are the same (nan)
        scores = [0.1, 0.2, 0.3, 0.4]
        labels = [1, 1, 1, 1]
        auroc, fpr95, tpr05 = get_metrics(scores, labels)
        assert math.isnan(auroc)
        assert math.isnan(fpr95)
        assert math.isnan(tpr05)
