# Copyright (c) 2025 Nikkei Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import difflib
import logging
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SAMPLING_PARAMETERS: dict[str, Any] = {
    "prompt_logprobs": 0,
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
}

KNOWN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "lora",
        "data",
        "sampling_parameters",
        "methods",
        "output_dir",
    }
)

KNOWN_DATA_KEYS: frozenset[str] = frozenset(
    {
        "data_path",
        "format",
        "text_column",
        "label_column",
        "text_length",
        "space_delimited_language",
    }
)

KNOWN_METHOD_ENTRY_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "params",
        "name",
    }
)

COMMON_TYPO_SUGGESTIONS: dict[str, str] = {
    "token_length": "text_length",
    "max_length": "text_length",
    "max_token_length": "text_length",
    "path": "data_path",
    "data_format": "format",
    "sampling_params": "sampling_parameters",
    "method": "type",
    "parameters": "params",
}


def _suggest_key(unknown_key: str, valid_keys: frozenset[str]) -> str | None:
    """Suggest the closest valid key for a mistyped configuration key."""
    alias = COMMON_TYPO_SUGGESTIONS.get(unknown_key)
    if alias in valid_keys:
        return alias
    matches = difflib.get_close_matches(
        unknown_key, sorted(valid_keys), n=1, cutoff=0.6
    )
    return matches[0] if matches else None


class Config:
    """Class for loading config files"""

    def __init__(
        self, config_path: str | Path, validate_required: bool = False
    ) -> None:
        """Load config file

        Args:
            config_path: Path to config file
            validate_required: If True, validate required sections immediately
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._warn_unknown_keys()
        if validate_required:
            self.validate()

    def _load_config(self) -> dict[str, Any]:
        """Load config file"""
        with self.config_path.open() as f:
            config = yaml.safe_load(f)
        if config is None:
            return {}
        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration root in '{self.config_path}' must be a YAML mapping, "
                f"got {type(config).__name__}."
            )
        return config

    def _warn_unknown_keys(self) -> None:
        """Log warnings for unrecognized keys or probable typos."""
        for key in self.config:
            if key not in KNOWN_TOP_LEVEL_KEYS:
                suggestion = _suggest_key(str(key), KNOWN_TOP_LEVEL_KEYS)
                hint = f" Did you mean '{suggestion}'?" if suggestion else ""
                logging.warning(
                    f"Unknown top-level config key '{key}' in '{self.config_path}'.{hint}"
                )

        data_cfg = self.config.get("data")
        if isinstance(data_cfg, dict):
            for key in data_cfg:
                if key not in KNOWN_DATA_KEYS:
                    suggestion = _suggest_key(str(key), KNOWN_DATA_KEYS)
                    hint = f" Did you mean '{suggestion}'?" if suggestion else ""
                    logging.warning(
                        f"Unknown key '{key}' in 'data' config of '{self.config_path}'.{hint}"
                    )

        methods_cfg = self.config.get("methods")
        if isinstance(methods_cfg, list):
            for idx, entry in enumerate(methods_cfg):
                if not isinstance(entry, dict):
                    continue
                for key in entry:
                    if key not in KNOWN_METHOD_ENTRY_KEYS:
                        suggestion = _suggest_key(str(key), KNOWN_METHOD_ENTRY_KEYS)
                        hint = f" Did you mean '{suggestion}'?" if suggestion else ""
                        logging.warning(
                            f"Unknown key '{key}' in 'methods[{idx}]' config of "
                            f"'{self.config_path}'.{hint}"
                        )

    def validate(self) -> None:
        """Validate required configuration sections and field types.

        Raises:
            ValueError: If required fields are missing or malformed
        """
        model_cfg = self.config.get("model")
        if not isinstance(model_cfg, dict) or not model_cfg.get("model_id"):
            raise ValueError(
                "Missing required config 'model.model_id'. "
                "Specify a valid Hugging Face model ID or path under 'model:'."
            )

        data_cfg = self.config.get("data")
        if not isinstance(data_cfg, dict) or not data_cfg.get("data_path"):
            raise ValueError(
                "Missing required config 'data.data_path'. "
                "Specify a dataset path or name under 'data:'."
            )

        text_length = data_cfg.get("text_length")
        if text_length is not None and (
            isinstance(text_length, bool)
            or not isinstance(text_length, int)
            or text_length <= 0
        ):
            raise ValueError(
                f"'data.text_length' must be a positive integer, got {text_length!r}."
            )

        methods_cfg = self.config.get("methods")
        if not isinstance(methods_cfg, list) or len(methods_cfg) == 0:
            raise ValueError(
                "Missing required config 'methods'. "
                "Specify a non-empty list of MIA methods to evaluate."
            )

        for idx, entry in enumerate(methods_cfg):
            if not isinstance(entry, dict) or not entry.get("type"):
                raise ValueError(
                    f"Method entry 'methods[{idx}]' must be a mapping with a non-empty 'type'."
                )
            params = entry.get("params")
            if params is not None and not isinstance(params, dict):
                raise ValueError(
                    f"'methods[{idx}].params' must be a mapping, got {type(params).__name__}."
                )
            name = entry.get("name")
            if name is not None and (not isinstance(name, str) or not name.strip()):
                raise ValueError(
                    f"'methods[{idx}].name' must be a non-empty string when provided."
                )

    @property
    def model(self) -> dict[str, Any]:
        """Get model config"""
        return self.config.get("model", {})

    @property
    def lora(self) -> dict[str, Any] | None:
        """Get LoRA config"""
        return self.config.get("lora")

    @property
    def data(self) -> dict[str, Any]:
        """Get data config"""
        return self.config.get("data", {})

    @property
    def methods(self) -> list[dict[str, Any]]:
        """Get method configs"""
        return self.config.get("methods", [])

    @property
    def sampling_parameters(self) -> dict[str, Any]:
        """Get sampling_parameters config"""
        sampling_parameters = (self.config.get("sampling_parameters") or {}).copy()
        return {**DEFAULT_SAMPLING_PARAMETERS, **sampling_parameters}
