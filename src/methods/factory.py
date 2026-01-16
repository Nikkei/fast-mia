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

from typing import Any

from .base import BaseMethod
from .conrecall import CONReCaLLMethod
from .dcpdd import DCPDDMethod
from .loss import LossMethod
from .lower import LowerMethod
from .mink import MinKMethod
from .pac import PACMethod
from .recall import ReCaLLMethod
from .samia import SaMIAMethod
from .zlib import ZlibMethod

METHOD_BUILDERS = {
    "loss": LossMethod,
    "lower": LowerMethod,
    "zlib": ZlibMethod,
    "mink": MinKMethod,
    "pac": PACMethod,
    "recall": ReCaLLMethod,
    "conrecall": CONReCaLLMethod,
    "samia": SaMIAMethod,
    "dcpdd": DCPDDMethod,
}


class MethodFactory:
    """Method factory class"""

    @staticmethod
    def create_method(method_config: dict[str, Any]) -> BaseMethod:
        """Create a method

        Args:
            method_config: Method configuration
                - type: Type of method ('loss', 'lower', 'zlib', 'mink', 'pac', 'recall', 'conrecall', 'samia', 'dcpdd')
                - params: Method-specific parameters

        Returns:
            Created method

        Raises:
            ValueError: If unknown method type is specified
        """
        method_type = method_config.get("type")
        method_params = method_config.get("params", {})

        if not method_type:
            supported = ", ".join(sorted(METHOD_BUILDERS))
            raise ValueError(
                f"Each method config must include a 'type'. Supported types: {supported}"
            )

        builder = METHOD_BUILDERS.get(method_type)
        if builder is None:
            supported = ", ".join(sorted(METHOD_BUILDERS))
            raise ValueError(
                f"Unknown method type '{method_type}'. Supported types: {supported}"
            )

        return builder(method_params)

    @staticmethod
    def create_methods(methods_config: list[dict[str, Any]]) -> list[BaseMethod]:
        """Create multiple methods

        Args:
            methods_config: List of method configurations

        Returns:
            List of created methods
        """
        methods = []
        for method_config in methods_config:
            method = MethodFactory.create_method(method_config)
            methods.append(method)
        return methods
