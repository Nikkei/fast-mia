import os
from pathlib import Path

import pytest
import yaml

from src.config import Config


# Function to get the fixture path for tests
def get_fixture_path(filename):
    """Get the fixture file path for tests"""
    current_dir = Path(__file__).parent.parent  # tests/unit -> tests
    return current_dir / "fixtures" / filename


class TestConfig:
    """Test for config file loading and parsing functionality"""

    def test_valid_config_load(self):
        """Test to confirm valid config file is loaded correctly"""
        config_path = get_fixture_path("valid_config.yaml")
        config = Config(config_path)

        # Check that each section's values are loaded correctly
        assert config.model["model_id"] == "facebook/opt-125m"
        # Check sampling_parameters values
        assert config.sampling_parameters["temperature"] == 0.0
        assert config.sampling_parameters["top_p"] == 1.0

        assert config.data["data_path"] == "./data/test_data.csv"
        assert config.data["format"] == "csv"
        assert config.data["text_column"] == "text"
        assert config.data["label_column"] == "label"
        assert config.data["text_length"] == 64

        assert len(config.methods) == 3
        assert config.methods[0]["type"] == "loss"
        assert config.methods[1]["type"] == "zlib"
        assert config.methods[2]["type"] == "mink"
        assert config.methods[2]["params"]["k_percent"] == 10

        assert config.config["output_dir"] == "./results"

    def test_missing_model_section(self):
        """Test to confirm config file missing required section is handled correctly"""
        config_path = get_fixture_path("invalid_config.yaml")
        config = Config(config_path)

        # Check that an empty dict is returned if model section is missing
        assert config.model == {}
        # Check that other sections are loaded correctly
        assert config.data["data_path"] == "./data/test_data.csv"
        assert len(config.methods) == 1

    def test_nonexistent_config_file(self):
        """Test to confirm error is raised when nonexistent file is specified"""
        config_path = get_fixture_path("nonexistent_file.yaml")
        with pytest.raises(FileNotFoundError):
            Config(config_path)

    def test_malformed_yaml(self):
        """Test to confirm error is raised for malformed YAML file"""
        config_path = get_fixture_path("malformed_config.yaml")
        with pytest.raises(yaml.YAMLError):
            Config(config_path)

    def test_property_methods(self):
        """Test to confirm each property method works correctly"""
        config_path = get_fixture_path("valid_config.yaml")
        config = Config(config_path)

        # Check return values of each property method
        assert isinstance(config.model, dict)
        assert isinstance(config.data, dict)
        assert isinstance(config.methods, list)
        assert config.lora is None  # 設定ファイルにLoRAセクションがない
        # Check that sampling_parameters are loaded correctly
        assert config.sampling_parameters["temperature"] == 0.0
        assert config.sampling_parameters["top_p"] == 1.0

    def test_empty_sections(self):
        """Test for empty sections"""
        # Create a temporary empty config file
        empty_config_path = get_fixture_path("empty_config.yaml")
        with open(empty_config_path, "w") as f:
            f.write("# Empty config file\n")

        try:
            config = Config(empty_config_path)

            # Check that each section is initialized as empty dict or list
            assert config.model == {}
            assert config.data == {}
            assert config.methods == []
            assert config.lora is None
            assert config.sampling_parameters == {}
        finally:
            # Delete file after test
            if os.path.exists(empty_config_path):
                os.remove(empty_config_path)
