import tempfile
from pathlib import Path
import shutil
import os
import pandas as pd
import pytest

from src.config import Config
from src.data_loader import DataLoader
from src.model_loader import ModelLoader
from src.methods.factory import MethodFactory
from src.evaluator import Evaluator

class TestEndToEndIntegration:
    def test_end_to_end_all(self):
        """
        End-to-end test: Verify the entire process from config file loading, data loading, model loading, method creation, evaluation, to result output and saving
        - Verify config values, number of methods, and contents
        - Run evaluation and check DataFrame
        - Verify result file saving, contents, and overwrite
        """
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

            # --- Check DataFrame ---
            assert isinstance(result, pd.DataFrame)
            assert set(result.columns) >= {"method", "auroc", "fpr95", "tpr05"}
            # All method names should be included in the result
            result_methods = set(result["method"])
            for name in method_names:
                assert any(name in m for m in result_methods)
            # Number of data
            assert len(result) == len(config.methods)
            # Metric values should be in percent format
            for col in ["auroc", "fpr95", "tpr05"]:
                for val in result[col]:
                    assert isinstance(val, str) and val.endswith("%")

            # --- Verify result file saving and contents ---
            output_path = Path(temp_dir) / "result.csv"
            result.to_csv(output_path, index=False)
            assert output_path.exists()
            df = pd.read_csv(output_path)
            assert set(df.columns) >= {"method", "auroc", "fpr95", "tpr05"}
            assert df["method"].dtype == object
            assert len(df) == len(config.methods)
            for col in ["auroc", "fpr95", "tpr05"]:
                for val in df[col]:
                    assert isinstance(val, str) and val.endswith("%")
