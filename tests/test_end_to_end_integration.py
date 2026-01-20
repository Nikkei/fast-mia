import tempfile
from pathlib import Path
import shutil
import pandas as pd
import pytest
from unittest import mock

from src.config import Config
from src.data_loader import DataLoader
from src.model_loader import ModelLoader
from src.methods.factory import MethodFactory
from src.evaluator import Evaluator, EvaluationResult

class TestEndToEndIntegration:
    @mock.patch("src.model_loader.SamplingParams")
    @mock.patch("src.model_loader.LLM")
    def test_end_to_end_all(self, mock_llm, mock_sampling_params):
        """
        End-to-end test: Verify the entire process from config file loading, data loading, model loading, method creation, evaluation, to result output and saving
        - Verify config values, number of methods, and contents
        - Run evaluation and check DataFrame
        - Verify result file saving, contents, and overwrite
        """
        # Mock setup for vllm
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.eos_token = "</s>"
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.get_tokenizer.return_value = mock_tokenizer
        
        # Mock sampling params
        mock_sampling_params_instance = mock.MagicMock()
        mock_sampling_params.return_value = mock_sampling_params_instance
        
        # Mock generate method to return dummy outputs with proper structure
        def create_mock_output(text):
            mock_output = mock.MagicMock()
            mock_output.prompt = text
            # Create mock logprobs structure
            mock_logprob = mock.MagicMock()
            mock_logprob.logprob = -0.5
            mock_output.prompt_logprobs = [
                {0: mock_logprob} for _ in range(len(text.split()))
            ]
            return mock_output
        
        # Mock generate to return appropriate outputs based on input
        def mock_generate(texts, *args, **kwargs):
            return [create_mock_output(text) for text in texts]
        
        mock_llm_instance.generate.side_effect = mock_generate
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare config and data files
            config_src = Path(__file__).parent / "fixtures" / "valid_config.yaml"
            config_dst = Path(temp_dir) / "config.yaml"
            shutil.copy(config_src, config_dst)

            data_src = Path(__file__).parent / "fixtures" / "test_data.csv"
            data_dst = Path(temp_dir) / "test_data.csv"
            shutil.copy(data_src, data_dst)

            # Rewrite data_path/output_dir in config file
            with open(config_dst, "r") as f:
                config_text = f.read()
            config_text = config_text.replace("./data/test_data.csv", str(data_dst))
            config_text = config_text.replace("./results", str(temp_dir))
            with open(config_dst, "w") as f:
                f.write(config_text)

            # Load config
            config = Config(config_dst)

            # --- Verify config values, number of methods, and contents ---
            assert config.model["model_id"] == "facebook/opt-125m"
            assert config.sampling_parameters["temperature"] == 0.0
            assert config.sampling_parameters["top_p"] == 1.0
            assert config.sampling_parameters["prompt_logprobs"] == 0
            assert config.sampling_parameters["max_tokens"] == 1
            assert config.data["text_column"] == "text"
            assert config.data["label_column"] == "label"
            assert len(config.methods) >= 2
            method_names = [m["type"] for m in config.methods]

            # --- DataLoader, ModelLoader, Method creation ---
            loader = DataLoader(
                data_path=config.data["data_path"],
                data_format=config.data["format"],
                text_column=config.data["text_column"],
                label_column=config.data["label_column"]
            )
            model_loader = ModelLoader(config.model)
            methods = [MethodFactory.create_method(m) for m in config.methods]

            # --- Run evaluation ---
            evaluator = Evaluator(loader, model_loader, methods)
            result = evaluator.evaluate(config)

            # --- Check EvaluationResult ---
            assert isinstance(result, EvaluationResult)
            assert isinstance(result.results_df, pd.DataFrame)
            assert set(result.results_df.columns) >= {"method", "auroc", "fpr95", "tpr05"}
            # All method names should be included in the result
            result_methods = set(result.results_df["method"])
            for name in method_names:
                assert any(name in m for m in result_methods)
            # Number of data
            assert len(result.results_df) == len(config.methods)
            # Metric values should be in percent format
            for col in ["auroc", "fpr95", "tpr05"]:
                for val in result.results_df[col]:
                    assert isinstance(val, str) and val.endswith("%")
            # Check detailed_results
            assert len(result.detailed_results) == len(config.methods)
            # Check data_stats
            assert result.data_stats["num_samples"] == 4

            # --- Verify result file saving and contents ---
            output_path = Path(temp_dir) / "result.csv"
            result.results_df.to_csv(output_path, index=False)
            assert output_path.exists()
            df = pd.read_csv(output_path)
            assert set(df.columns) >= {"method", "auroc", "fpr95", "tpr05"}
            assert df["method"].dtype == object
            assert len(df) == len(config.methods)
            for col in ["auroc", "fpr95", "tpr05"]:
                for val in df[col]:
                    assert isinstance(val, str) and val.endswith("%")
