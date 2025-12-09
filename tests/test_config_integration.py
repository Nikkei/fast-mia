import tempfile
from pathlib import Path

from src.config import Config


class TestConfigIntegration:
    """Integration test for config file functionality"""

    def test_config_with_temp_file(self):
        """Test using a temporary file"""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary config file
            temp_config_path = Path(temp_dir) / "temp_config.yaml"
            with open(temp_config_path, "w") as f:
                f.write("""
# Temporary config file
model:
  model_id: "facebook/opt-125m"
  temperature: 0.0
  top_p: 0.95

data:
  data_path: "./data/test_data.csv"
  format: "csv"
  text_column: "input_text"
  label_column: "is_member"
  text_length: 128

methods:
  - type: "loss"
    params: {}
  - type: "zlib"
    params:
      window_size: 32
  - type: "mink"
    params:
      k_percent: 5

output_dir: "./temp_results"
                """)

            # Read config
            config = Config(temp_config_path)

            # Check config values
            assert config.model["model_id"] == "facebook/opt-125m"
            assert config.model["temperature"] == 0.0
            assert config.data["text_column"] == "input_text"
            assert config.data["text_length"] == 128
            assert len(config.methods) == 3
            assert config.methods[1]["params"]["window_size"] == 32
            assert config.config["output_dir"] == "./temp_results"

    def test_config_with_real_use_case(self):
        """Test simulating a real use case"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test config file
            config_path = Path(temp_dir) / "test_config.yaml"
            with open(config_path, "w") as f:
                f.write("""
model:
  model_id: "facebook/opt-125m"
  
lora:
  name: "adapter"
  id: 1
  path: "./lora_path"

data:
  data_path: "swj0419/WikiMIA"
  format: "huggingface"
  text_column: "input"
  label_column: "label"
  text_length: 64
  
methods:
  - type: "loss"
    params: {}
  - type: "zlib"
    params: {}
  - type: "mink"
    params:
      k_percent: 10
  - type: "recall"
    params:
      num_samples: 10
      
sampling_parameters:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 20
  
output_dir: "./results/wikimia_test"
                """)

            # Read config
            config = Config(config_path)

            # Check WikiMIA config
            assert config.data["data_path"] == "swj0419/WikiMIA"
            assert config.data["format"] == "huggingface"
            assert config.data["text_length"] == 64

            # Check multiple methods
            assert len(config.methods) == 4
            assert config.methods[0]["type"] == "loss"
            assert config.methods[2]["params"]["k_percent"] == 10
            assert config.methods[3]["type"] == "recall"

            # Check sampling parameters
            assert config.sampling_parameters["temperature"] == 0.7
            assert config.sampling_parameters["top_p"] == 0.9
            assert config.sampling_parameters["max_tokens"] == 20

            # Check lora config
            assert config.lora["name"] == "adapter"
            assert config.lora["id"] == 1
            assert config.lora["path"] == "./lora_path"
