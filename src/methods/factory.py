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

import copy
from collections import Counter, defaultdict
from typing import Any

from .aeca import AECAMethod
from .base import BaseMethod
from .camia import CAMIAMethod
from .conrecall import CONReCaLLMethod
from .dcpdd import DCPDDMethod
from .loss import LossMethod
from .lower import LowerMethod
from .mink import MinKMethod
from .neighbour import NeighbourMethod
from .pac import PACMethod
from .recall import ReCaLLMethod
from .ref import RefMethod
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
    "ref": RefMethod,
    "neighbour": NeighbourMethod,
    "aeca": AECAMethod,
    "camia": CAMIAMethod,
}

_SCALAR_TYPES = (str, int, float, bool)


def _disambiguate_method_names(
    methods: list[BaseMethod], methods_config: list[dict[str, Any]]
) -> None:
    """Ensure every method in ``methods`` has a unique ``method_name``.

    First attempts to distinguish unnamed colliding methods using differing
    scalar parameter values (e.g. ``recall_num_shots=3`` vs
    ``recall_num_shots=10``), then appends ``_1``, ``_2``, ... for any
    remaining duplicates.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, method in enumerate(methods):
        groups[method.method_name].append(idx)

    for base_name, indices in groups.items():
        if len(indices) <= 1:
            continue
        # If none of the colliding entries set an explicit custom name, try
        # disambiguating via differing scalar params first.
        if all(not methods_config[i].get("name") for i in indices):
            param_dicts = [methods_config[i].get("params") or {} for i in indices]
            all_keys = list(
                dict.fromkeys(k for params in param_dicts for k in params.keys())
            )
            diff_keys = [
                k
                for k in all_keys
                if all(
                    isinstance(params.get(k), _SCALAR_TYPES) or params.get(k) is None
                    for params in param_dicts
                )
                and len({repr(params.get(k)) for params in param_dicts}) > 1
            ]
            if diff_keys:
                for i in indices:
                    params = methods_config[i].get("params") or {}
                    suffix = "_".join(
                        f"{k}={params.get(k, 'default')}" for k in diff_keys
                    )
                    methods[i].method_name = f"{base_name}_{suffix}"

    # Guarantee strict uniqueness for any remaining collisions.
    counts = Counter(method.method_name for method in methods)
    reserved = {name for name, count in counts.items() if count == 1}
    seen_counters: dict[str, int] = defaultdict(int)
    for method in methods:
        if counts[method.method_name] <= 1:
            continue
        stem = method.method_name
        while True:
            seen_counters[stem] += 1
            candidate = f"{stem}_{seen_counters[stem]}"
            if candidate not in reserved:
                method.method_name = candidate
                reserved.add(candidate)
                break


class MethodFactory:
    """Method factory class"""

    @staticmethod
    def create_method(method_config: dict[str, Any]) -> BaseMethod:
        """Create a method

        Args:
            method_config: Method configuration
                - type: Type of method
                - name: Optional custom display identifier
                - params: Method-specific parameters

        Returns:
            Created method

        Raises:
            ValueError: If unknown method type or invalid custom name is specified
        """
        method_type = method_config.get("type")
        method_params = copy.deepcopy(method_config.get("params") or {})
        custom_name = method_config.get("name")

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

        if custom_name is not None and (
            not isinstance(custom_name, str) or not custom_name.strip()
        ):
            raise ValueError(
                f"Method 'name' must be a non-empty string, got {custom_name!r}."
            )

        method = builder(method_params)
        if custom_name is not None:
            method.method_name = custom_name.strip()
        return method

    @staticmethod
    def create_methods(methods_config: list[dict[str, Any]]) -> list[BaseMethod]:
        """Create multiple methods with automatic name collision resolution

        Args:
            methods_config: List of method configurations

        Returns:
            List of created methods with unique names
        """
        method_types = [method_config.get("type") for method_config in methods_config]
        if "ref" in method_types and method_types.index("ref") != len(method_types) - 1:
            raise ValueError(
                "The 'ref' method destroys the global vLLM parallel state when "
                "releasing its reference model, which breaks inference for "
                "methods that run after it. Move 'ref' to the end of the "
                "methods list in your config."
            )

        methods = [
            MethodFactory.create_method(method_config)
            for method_config in methods_config
        ]
        _disambiguate_method_names(methods, methods_config)
        return methods
