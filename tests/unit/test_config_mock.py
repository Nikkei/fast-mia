from unittest.mock import mock_open, patch

import pytest
import yaml

from src.config import Config


class TestConfigWithMock:
    """Test for config file loading and parsing functionality using mock"""

    @patch(
        "pathlib.Path.open",
        new_callable=mock_open,
        read_data="""
model:
  model_id: "facebook/opt-125m"
  temperature: 0.0
data:
  data_path: "./data/mock_data.csv"
  format: "csv"
methods:
  - type: "loss"
    params: {}
output_dir: "./results"
""",
    )
    def test_config_load_with_mock(self, mock_file):
        """Test loading config file using mock"""
        config = Config("mock_config.yaml")

        # Check that the file is opened
        mock_file.assert_called_once_with()

        # Check that the settings are loaded correctly
        assert config.model["model_id"] == "facebook/opt-125m"
        assert config.model["temperature"] == 0.0
        assert config.data["data_path"] == "./data/mock_data.csv"
        assert config.data["format"] == "csv"
        assert len(config.methods) == 1
        assert config.methods[0]["type"] == "loss"
        assert config.config["output_dir"] == "./results"
        assert config.sampling_parameters["prompt_logprobs"] == 0
        assert config.sampling_parameters["max_tokens"] == 1
        assert config.sampling_parameters["temperature"] == 0.0
        assert config.sampling_parameters["top_p"] == 1.0

    @patch(
        "pathlib.Path.open", new_callable=mock_open, read_data="invalid: yaml: file:"
    )
    @patch("yaml.safe_load")
    def test_yaml_error_with_mock(self, mock_yaml_load, _):
        """Test YAML parse error using mock"""
        # Mock yaml.safe_load to raise YAMLError
        mock_yaml_load.side_effect = yaml.YAMLError("Mock YAML Error")

        # Check that YAMLError is raised
        with pytest.raises(yaml.YAMLError):
            Config("mock_config.yaml")

    @patch(
        "pathlib.Path.open",
        new_callable=mock_open,
        read_data="""
# LoRA section included in config
model:
  model_id: "facebook/opt-125m"
lora:
  name: "test_adapter"
  id: 1
  path: "/path/to/adapter"
data:
  data_path: "./data/mock_data.csv"
sampling_parameters:
  temperature: 0.7
  top_p: 0.9
""",
    )
    def test_optional_sections_with_mock(self, _):
        """Test for optional sections (lora, sampling_parameters)"""
        config = Config("mock_config.yaml")

        # Check that optional sections are loaded correctly
        assert config.lora is not None
        assert config.lora["name"] == "test_adapter"
        assert config.lora["id"] == 1
        assert config.lora["path"] == "/path/to/adapter"

        assert config.sampling_parameters is not None
        assert config.sampling_parameters["temperature"] == 0.7
        assert config.sampling_parameters["top_p"] == 0.9
        assert config.sampling_parameters["prompt_logprobs"] == 0
        assert config.sampling_parameters["max_tokens"] == 1

        # Check that an empty list is returned if methods section is missing
        assert config.methods == []
