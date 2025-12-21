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

from pathlib import Path
from typing import Any

import yaml

DEFAULT_SAMPLING_PARAMETERS: dict[str, Any] = {
    "prompt_logprobs": 0,
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
}


class Config:
    """Class for loading config files"""

    def __init__(self, config_path: str | Path) -> None:
        """Load config file

        Args:
            config_path: Path to config file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load config file"""
        with self.config_path.open() as f:
            config = yaml.safe_load(f)
        return config or {}

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
