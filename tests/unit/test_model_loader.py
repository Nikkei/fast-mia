from pathlib import Path
from unittest import mock

import pytest

from src.config import Config
from src.model_loader import ModelLoader


class TestModelLoader:
    """Test class for model loader functionality"""

    @pytest.fixture
    def valid_config_path(self):
        """Fixture to get the path of a valid config file"""
        return Path(__file__).parent.parent / "fixtures" / "valid_config.yaml"

    @pytest.fixture
    def config(self, valid_config_path):
        """Fixture to get the config object"""
        return Config(valid_config_path)

    @mock.patch("src.model_loader.LLM")
    def test_load_model_normal(self, mock_llm):
        """Test for normal model loading"""
        # Mock setup
        mock_tokenizer = mock.MagicMock()
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.get_tokenizer.return_value = mock_tokenizer

        # Model config
        model_config = {
            "model_id": "facebook/opt-125m",
            "trust_remote_code": True,
        }

        # Model loader instance
        loader = ModelLoader(model_config)

        # Check if model is loaded correctly
        mock_llm.assert_called_once_with(
            model="facebook/opt-125m",
            trust_remote_code=True,
        )

        # Check if tokenizer setup is correct
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token
        assert mock_tokenizer.padding_side == "left"

    @mock.patch("src.model_loader.LLM")
    def test_missing_model_id(self, mock_llm):
        """Test when model_id is missing"""
        # Model config without model_id
        model_config = {
            "temperature": 0.0,
            "top_p": 1.0,
        }
        # Check that ValueError is raised
        with pytest.raises(ValueError, match="model_id is required"):
            ModelLoader(model_config)

        # Check that mock is not called
        mock_llm.assert_not_called()

    @mock.patch("src.model_loader.LLM")
    def test_get_sampling_params(self, mock_llm):
        """Test for generating sampling parameters"""
        # Mock setup
        mock_tokenizer = mock.MagicMock()
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.get_tokenizer.return_value = mock_tokenizer

        # Model loader instance
        model_config = {"model_id": "facebook/opt-125m"}
        loader = ModelLoader(model_config)

        # Sampling parameter setup
        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "stop": ["END"],
        }

        # Get sampling parameters
        with mock.patch("src.model_loader.SamplingParams") as mock_sampling_params:
            loader.get_sampling_params(sampling_params)

            # Check if SamplingParams is called with correct parameters
            mock_sampling_params.assert_called_once_with(
                temperature=0.7,
                top_p=0.9,
                max_tokens=100,
                stop=["END"],
            )

    @mock.patch("src.model_loader.LLM")
    def test_get_lora_request(self, mock_llm):
        """Test for generating LoRA request"""
        # Mock setup
        mock_tokenizer = mock.MagicMock()
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.get_tokenizer.return_value = mock_tokenizer

        # Model loader instance
        model_config = {"model_id": "facebook/opt-125m"}
        loader = ModelLoader(model_config)

        # LoRA config
        lora_config = {
            "lora_id": "example-lora",
            "lora_int8": False,
            "lora_modules": ["q_proj", "v_proj"],
        }

        # Get LoRA request
        with mock.patch("src.model_loader.LoRARequest") as mock_lora_request:
            loader.get_lora_request(lora_config)

            # Check if LoRARequest is called with correct parameters
            mock_lora_request.assert_called_once_with(
                lora_id="example-lora",
                lora_int8=False,
                lora_modules=["q_proj", "v_proj"],
            )

    @mock.patch("src.model_loader.LLM")
    def test_model_load_error(self, mock_llm):
        """Test for model loading error"""
        mock_llm.side_effect = ValueError("Model not found")
        model_config = {"model_id": "non-existent-model"}
        with pytest.raises(ValueError, match="Model not found"):
            ModelLoader(model_config)

    @mock.patch("src.model_loader.LLM")
    def test_invalid_sampling_params(self, mock_llm):
        """Test for invalid sampling parameters"""
        # Mock setup
        mock_tokenizer = mock.MagicMock()
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.get_tokenizer.return_value = mock_tokenizer

        # Model loader instance
        model_config = {"model_id": "facebook/opt-125m"}
        loader = ModelLoader(model_config)

        # Invalid parameters in sampling parameters
        invalid_params = {
            "temperature": "invalid",  # Not a number but a string
            "top_p": 0.9,
        }

        # Mock SamplingParams to raise TypeError
        with mock.patch(
            "src.model_loader.SamplingParams",
            side_effect=TypeError("temperature must be a number"),
        ):
            # Check if TypeError is raised
            with pytest.raises(
                TypeError, match="temperature must be a number"
            ):
                loader.get_sampling_params(invalid_params)

    @mock.patch("src.model_loader.LLM")
    def test_invalid_lora_config(self, mock_llm):
        """Test for invalid LoRA config"""
        # Mock setup
        mock_tokenizer = mock.MagicMock()
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.get_tokenizer.return_value = mock_tokenizer

        # Model loader instance
        model_config = {"model_id": "facebook/opt-125m"}
        loader = ModelLoader(model_config)

        # Invalid LoRA config
        invalid_lora_config = {
            # lora_id is missing
            "lora_int8": False,
        }

        # Mock LoRARequest to raise ValueError
        with mock.patch(
            "src.model_loader.LoRARequest", side_effect=ValueError("lora_id is required")
        ):
            # Check if ValueError is raised
            with pytest.raises(ValueError, match="lora_id is required"):
                loader.get_lora_request(invalid_lora_config)

    @mock.patch("src.model_loader.LLM")
    def test_from_config(self, mock_llm, config):
        """Test for creating model loader from config object"""
        # Mock setup
        mock_tokenizer = mock.MagicMock()
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.get_tokenizer.return_value = mock_tokenizer

        # Create model loader from config object
        loader = ModelLoader(config.model)

        # Check if model is loaded correctly
        mock_llm.assert_called_once_with(
            model="facebook/opt-125m"
        )

    @mock.patch("src.model_loader.LLM")
    def test_complex_sampling_params(self, mock_llm):
        """Test for complex sampling parameters"""
        # Mock setup
        mock_tokenizer = mock.MagicMock()
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.get_tokenizer.return_value = mock_tokenizer

        # Model loader instance
        model_config = {"model_id": "facebook/opt-125m"}
        loader = ModelLoader(model_config)

        # Complex sampling parameters
        complex_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "stop": ["END", "STOP", "."],
            "top_k": 40,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.2,
            "beam_width": 4,
            "length_penalty": 1.5,
        }

        # Get sampling parameters
        with mock.patch("src.model_loader.SamplingParams") as mock_sampling_params:
            loader.get_sampling_params(complex_params)

            # Check if SamplingParams is called with correct parameters
            mock_sampling_params.assert_called_once_with(**complex_params)
