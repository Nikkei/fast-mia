import os
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from datasets import Dataset

from src.data_loader import DataLoader


class TestDataLoader:
    """Test class for DataLoader"""

    @pytest.fixture
    def sample_csv_path(self):
        """CSV sample data path fixture"""
        return Path(__file__).parent.parent / "fixtures" / "test_data.csv"

    @pytest.fixture
    def sample_json_path(self):
        """JSON sample data path fixture"""
        return Path(__file__).parent.parent / "fixtures" / "test_data.json"

    @pytest.fixture
    def sample_jsonl_path(self):
        """JSONL sample data path fixture"""
        return Path(__file__).parent.parent / "fixtures" / "test_data.jsonl"
    
    @pytest.fixture
    def sample_parquet_path(self):
        """PARQUET sample data path fixture"""
        return Path(__file__).parent.parent / "fixtures" / "test_data.parquet"

    def test_load_csv(self, sample_csv_path):
        """Test loading data from CSV file"""
        # Create a data loader for the CSV file
        loader = DataLoader(
            data_path=sample_csv_path,
            data_format="csv",
            text_column="text",
            label_column="label",
        )

        # Get and verify data
        texts, labels = loader.get_data()
        assert len(texts) == 4
        assert len(labels) == 4
        assert texts[0] == "This is a sample text."
        assert labels[0] == 1

    def test_load_json(self, sample_json_path):
        """Test loading data from JSON file"""
        # Create a data loader for the JSON file
        loader = DataLoader(
            data_path=sample_json_path,
            data_format="json",
            text_column="text",
            label_column="label",
        )

        # Get and verify data
        texts, labels = loader.get_data()
        assert len(texts) == 3
        assert len(labels) == 3
        assert texts[0] == "This is a JSON sample text."
        assert labels[0] == 1

    def test_load_jsonl(self, sample_jsonl_path):
        """Test loading data from JSONL file"""
        # Create a data loader for the JSONL file
        loader = DataLoader(
            data_path=sample_jsonl_path,
            data_format="jsonl",
            text_column="text",
            label_column="label",
        )

        # Get and verify data
        texts, labels = loader.get_data()
        assert len(texts) == 3
        assert len(labels) == 3
        assert texts[0] == "This is a JSONL sample text."
        assert labels[0] == 1
        
    def test_load_parquet(self, sample_parquet_path):
        """Test loading data from PARQUET file"""
        # Create a data loader for the PARQUET file
        loader = DataLoader(
            data_path=sample_parquet_path,
            data_format="parquet",
            text_column="text",
            label_column="label",
        )

        # Get and verify data
        texts, labels = loader.get_data()
        assert len(texts) == 4
        assert len(labels) == 4
        assert texts[0] == "This is a sample text."
        assert labels[0] == 1

    def test_text_splitting(self, sample_csv_path):
        """Test text splitting functionality"""
        # Create a data loader
        loader = DataLoader(
            data_path=sample_csv_path,
            data_format="csv",
            text_column="text",
            label_column="label",
        )

        # Get data with specified token length
        token_length = 4
        texts, labels = loader.get_data(token_length=token_length)

        # Verify the length of split texts
        for text in texts:
            # Ensure the number of words separated by spaces is less than or equal to token_length
            assert len(text.split()) <= token_length

        # The last text is long, so verify it is split
        assert texts[3] == "This is a longer"

        # Verify labels remain unchanged
        assert labels == [1, 0, 1, 0]

    def test_error_handling_no_data(self):
        """Test error handling when data is not loaded"""
        loader = DataLoader()

        # Verify that calling get_data without loading data raises an error
        with pytest.raises(ValueError, match="Data is not loaded"):
            loader.get_data()

    def test_error_handling_invalid_format(self, sample_csv_path):
        """Test invalid data format"""
        with pytest.raises(ValueError, match="Unsupported data format"):
            DataLoader(data_path=sample_csv_path, data_format="unknown")

    def test_error_handling_missing_file(self):
        """Test non-existent file"""
        with pytest.raises(FileNotFoundError):
            DataLoader(data_path="nonexistent_file.csv", data_format="csv")

    @mock.patch("src.data_loader.load_dataset")
    def test_load_huggingface(self, mock_load_dataset):
        """Test loading data from Hugging Face Datasets"""
        # Create a mock dataset without using Dataset.from_dict()
        mock_train_split = mock.MagicMock()
        mock_train_split.to_pandas.return_value = pd.DataFrame(
            {"text": ["This is a Hugging Face sample text."], "label": [1]}
        )
        mock_dataset = {"train": mock_train_split}
        mock_load_dataset.return_value = mock_dataset

        # Create a data loader
        loader = DataLoader(
            data_path="dummy/dataset",
            data_format="huggingface",
            text_column="text",
            label_column="label",
        )

        # Verify that the mock is called correctly
        mock_load_dataset.assert_called_once_with("dummy/dataset")

        # Get and verify data
        texts, labels = loader.get_data()
        assert len(texts) == 1
        assert len(labels) == 1
        assert texts[0] == "This is a Hugging Face sample text."
        assert labels[0] == 1

    @mock.patch("src.data_loader.load_dataset")
    def test_load_wikimia(self, mock_load_dataset):
        """Test loading data from WikiMIA dataset"""
        # Create a mock dataset without using Dataset.from_dict()
        mock_dataset = mock.MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame(
            {"input": ["This is a WikiMIA sample text."], "label": [1]}
        )
        mock_load_dataset.return_value = mock_dataset

        # Load WikiMIA dataset
        token_length = 32
        loader = DataLoader.load_wikimia(token_length)

        # Verify that the mock is called correctly
        mock_load_dataset.assert_called_once_with(
            "swj0419/WikiMIA", split=f"WikiMIA_length{token_length}"
        )

        # Get and verify data
        texts, labels = loader.get_data()
        assert len(texts) == 1
        assert len(labels) == 1
        assert texts[0] == "This is a WikiMIA sample text."
        assert labels[0] == 1

        # Verify column names are set correctly for WikiMIA
        assert loader.text_column == "input"
        assert loader.label_column == "label"
    
    @mock.patch("src.data_loader.load_dataset")
    def test_load_mimir(self, mock_load_dataset):
        """Test loading data from Mimir dataset"""
        # Create a mock dataset without using Dataset.from_dict()
        # Mimir dataset uses direct dictionary access: dataset["member"] and dataset["nonmember"]
        mock_dataset = {
            "member": ["This is a Mimir sample text."],
            "nonmember": ["This is a Mimir sample text."],
        }
        mock_load_dataset.return_value = mock_dataset
        
        # Load Mimir dataset
        data_path = "iamgroot42/mimir_pc_702"
        token = "test_token"
        loader = DataLoader.load_mimir(data_path, token)
        
        # Verify that the mock is called correctly
        mock_load_dataset.assert_called_once_with(
            "iamgroot42/mimir", "pile_cc", split="ngram_7_0.2", token=token, trust_remote_code=True
        )
        
        
        # Get and verify data
        texts, labels = loader.get_data()
        assert len(texts) == 2
        assert len(labels) == 2
        assert texts[0] == "This is a Mimir sample text."
        assert texts[1] == "This is a Mimir sample text."
        assert labels[0] == 1
        assert labels[1] == 0
        
        # Verify column names are set correctly for Mimir
        assert loader.text_column == "input"
        assert loader.label_column == "label"

    def test_parquet_format(self):
        """Test loading data from Parquet file"""
        # Create a pandas DataFrame
        df = pd.DataFrame(
            {"text": ["This is a Parquet sample text."], "label": [1]}
        )

        # Create a temporary directory and save parquet file
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = os.path.join(tmp_dir, "sample.parquet")
            df.to_parquet(parquet_path)

            # Create a data loader for the parquet file
            loader = DataLoader(
                data_path=parquet_path,
                data_format="parquet",
                text_column="text",
                label_column="label",
            )

            # Get and verify data
            texts, labels = loader.get_data()
            assert len(texts) == 1
            assert len(labels) == 1
            assert texts[0] == "This is a Parquet sample text."
            assert labels[0] == 1

    def test_file_extension_validation(self):
        """Test file extension validation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test allowed extensions
            allowed_extensions = [".csv", ".json", ".jsonl", ".parquet"]
            for ext in allowed_extensions:
                test_file = os.path.join(tmp_dir, f"test{ext}")
                with open(test_file, "w") as f:
                    if ext == ".csv":
                        f.write("text,label\nsample,1")
                    elif ext == ".json":
                        f.write('[{"text": "sample", "label": 1}]')
                    elif ext == ".jsonl":
                        f.write('{"text": "sample", "label": 1}')
                    else:  # parquet
                        df = pd.DataFrame({"text": ["sample"], "label": [1]})
                        df.to_parquet(test_file)
                
                # Verify that the file can be loaded correctly
                loader = DataLoader(
                    data_path=test_file,
                    data_format=ext[1:],  # Remove the dot from the extension
                    text_column="text",
                    label_column="label",
                )
                texts, labels = loader.get_data()
                assert len(texts) == 1
                assert texts[0] == "sample"
                assert labels[0] == 1

    def test_unsupported_file_extension(self):
        """Test unsupported file extension validation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a file with an unsupported extension
            test_file = os.path.join(tmp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("sample text")
            
            # Verify that an error is raised for an unsupported extension
            with pytest.raises(ValueError, match="Unsupported file extension"):
                DataLoader(
                    data_path=test_file,
                    data_format="csv",  # The extension does not match the format
                    text_column="text",
                    label_column="label",
                )

