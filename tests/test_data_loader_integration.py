"""Integration test for DataLoader"""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from src.config import Config
from src.data_loader import DataLoader


class TestDataLoaderIntegration:
    """Integration test class for DataLoader"""

    @pytest.fixture
    def sample_config_path(self):
        """Fixture to get the path of the test config file"""
        return Path(__file__).parent / "fixtures" / "valid_config.yaml"

    @pytest.fixture
    def sample_csv_path(self):
        """Fixture to get the path of the CSV sample data"""
        return Path(__file__).parent / "fixtures" / "test_data.csv"

    def test_dataloader_from_config(self, sample_config_path, sample_csv_path):
        """Test initializing DataLoader from config file"""
        # Load config
        config = Config(sample_config_path)

        # Temporarily change the data path to the sample CSV path
        original_data_path = config.data.get("data_path")
        config.config["data"]["data_path"] = str(sample_csv_path)

        # Initialize data loader
        data_loader = DataLoader(
            data_path=config.data.get("data_path"),
            data_format=config.data.get("format", "csv"),
            text_column=config.data.get("text_column", "text"),
            label_column=config.data.get("label_column", "label"),
        )

        # Get and verify data
        texts, labels = data_loader.get_data()
        assert len(texts) == 4
        assert len(labels) == 4

        # Reset config to original path (to avoid affecting other tests)
        config.config["data"]["data_path"] = original_data_path

    @mock.patch("src.data_loader.load_dataset")
    def test_wikimia_integration_with_main(self, mock_load_dataset, monkeypatch):
        """Integration test for WikiMIA dataset (with main script)"""
        # Temporarily set environment variable (if needed)
        monkeypatch.setenv("PYTHONPATH", ".")

        # Create mock dataset without using Dataset.from_dict()
        mock_dataset = mock.MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame(
            {
                "input": [
                    "This is a WikiMIA integration test sample.",
                    "Another WikiMIA sample sentence.",
                ],
                "label": [1, 0],
            }
        )
        mock_load_dataset.return_value = mock_dataset

        # Load WikiMIA dataset
        text_length = 64
        data_loader = DataLoader.load_wikimia(text_length)

        # Get and verify data
        texts, labels = data_loader.get_data()
        assert len(texts) == 2
        assert len(labels) == 2
        assert texts[0] == "This is a WikiMIA integration test sample."
        assert labels[0] == 1

        # Verify correct parameters were called
        mock_load_dataset.assert_called_once_with(
            "swj0419/WikiMIA", split=f"WikiMIA_length{text_length}"
        )

    def test_huggingface_integration_not_supported(self):
        """Integration test to ensure unsupported Hugging Face datasets fail fast"""
        with pytest.raises(ValueError, match="Generic Hugging Face datasets are not supported"):
            DataLoader(
                data_path="dummy/dataset",
                data_format="huggingface",
                text_column="text",
                label_column="label",
            )

    def test_text_length_splitting_integration(self, sample_csv_path):
        """Integration test for text splitting with specified text length"""
        # データローダーを作成
        loader = DataLoader(
            data_path=sample_csv_path,
            data_format="csv",
            text_column="text",
            label_column="label",
        )

        # Test with different text lengths
        for text_length in [1, 2, 3, 5]:
            texts, labels = loader.get_data(text_length=text_length)

            # Verify all texts are less than or equal to text_length
            for text in texts:
                assert len(text.split()) <= text_length

            # Verify labels remain unchanged
            assert labels == [1, 0, 1, 0]

    @mock.patch("src.data_loader.load_dataset")
    def test_mimir_integration(self, mock_load_dataset):
        """Integration test for Mimir dataset"""
        # Create mock dataset
        mock_dataset = {
            "member": ["This is a member text sample."],
            "nonmember": ["This is a nonmember text sample."]
        }
        mock_load_dataset.return_value = mock_dataset

        # Load Mimir dataset
        data_path = "iamgroot42/mimir_pc_702"
        token = "test_token"
        data_loader = DataLoader.load_mimir(data_path, token)

        # Get and verify data
        texts, labels = data_loader.get_data()
        assert len(texts) == 2
        assert len(labels) == 2
        assert texts[0] == "This is a member text sample."
        assert texts[1] == "This is a nonmember text sample."
        assert labels[0] == 1  # member
        assert labels[1] == 0  # nonmember

        # Verify correct parameters were called
        mock_load_dataset.assert_called_once_with(
            "iamgroot42/mimir", "pile_cc", split="ngram_7_0.2", token=token, trust_remote_code=True
        )

    def test_mimir_invalid_path_format(self):
        """Test Mimir dataset with invalid path format"""
        # Test with incorrect number of elements
        invalid_paths = [
            "iamgroot42_pc",  # Only 2 elements
            "iamgroot42_pc_702_extra",  # 4 elements
            "single_element",  # Only 1 element
        ]

        for invalid_path in invalid_paths:
            with pytest.raises(ValueError, match="Mimir dataset path must be in the format"):
                DataLoader.load_mimir(invalid_path, "test_token")

    def test_mimir_invalid_domain(self):
        """Test Mimir dataset with invalid domain"""
        invalid_domain = "iamgroot42/mimir_invalid_702"
        
        with pytest.raises(ValueError, match="Invalid Mimir domain code"):
            DataLoader.load_mimir(invalid_domain, "test_token")

    def test_mimir_invalid_ngram(self):
        """Test Mimir dataset with invalid ngram"""
        invalid_ngram = "iamgroot42/mimir_pc_invalid"
        
        with pytest.raises(ValueError, match="Invalid Mimir ngram code"):
            DataLoader.load_mimir(invalid_ngram, "test_token")

    def test_mimir_missing_token(self):
        """Test that loading Mimir without a token fails fast"""
        data_path = "iamgroot42/mimir_pc_702"

        with pytest.raises(ValueError, match="Hugging Face token is required"):
            DataLoader.load_mimir(data_path, token="")

    def test_wikimia_invalid_text_length(self):
        """Ensure WikiMIA loader validates supported text lengths"""
        with pytest.raises(ValueError, match="WikiMIA dataset supports text_length"):
            DataLoader.load_wikimia(text_length=999)

    def test_missing_local_data_file(self, sample_csv_path):
        """Ensure local file loads show a descriptive error when the file is missing"""
        missing_path = sample_csv_path.parent / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError, match="Data file"):
            DataLoader(
                data_path=missing_path,
                data_format="csv",
                text_column="text",
                label_column="label",
            )

    @pytest.mark.parametrize(
        ("text_column", "label_column", "missing_name"),
        [
            ("missing_text", "label", "missing_text"),
            ("text", "missing_label", "missing_label"),
            ("missing_text", "missing_label", "'missing_label', 'missing_text'"),
            
        ],
    )
    def test_missing_columns_in_get_data(
        self, sample_csv_path, text_column, label_column, missing_name
    ):
        """Ensure missing text/label columns produce descriptive errors"""
        loader = DataLoader(
            data_path=sample_csv_path,
            data_format="csv",
            text_column=text_column,
            label_column=label_column,
        )

        with pytest.raises(ValueError, match=missing_name):
            loader.get_data()
