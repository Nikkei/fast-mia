# Copyright (c) 2026 Nikkei Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import importlib
import logging
import os
from types import ModuleType

import torch


def _load_xla_model() -> ModuleType | None:
    """Import ``torch_xla.core.xla_model`` lazily when available."""
    try:
        return importlib.import_module("torch_xla.core.xla_model")
    except Exception:
        return None


def is_cuda_available() -> bool:
    """Return True only when a real or explicitly mocked CUDA runtime is active."""
    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is None:
        return False
    is_avail_fn = getattr(cuda_mod, "is_available", None)
    if not callable(is_avail_fn):
        return False
    try:
        return is_avail_fn() is True
    except Exception:
        return False


def is_tpu_available() -> bool:
    """Return True when a Cloud TPU / PyTorch-XLA runtime is active."""
    xm = _load_xla_model()
    if xm is not None:
        get_supported = getattr(xm, "get_xla_supported_devices", None)
        if callable(get_supported):
            try:
                devices = get_supported("TPU")
                if isinstance(devices, list | tuple) and len(devices) > 0:
                    return True
            except Exception:
                pass
        if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
            return True

    return False


def get_accelerator_type() -> str:
    """Detect active accelerator backend ('cuda', 'tpu', or 'cpu')."""
    if is_cuda_available():
        return "cuda"
    if is_tpu_available():
        return "tpu"
    return "cpu"


def get_torch_device(prefer_xla: bool = True) -> torch.device:
    """Return the appropriate ``torch.device`` for auxiliary PyTorch models.

    Args:
        prefer_xla: Return the XLA device when TPU is active; when False on TPU,
            fall back to CPU to avoid XLA graph compilation for tiny workloads.

    Returns:
        Device object suitable for ``model.to(device)`` and ``tensor.to(device)``
    """
    backend = get_accelerator_type()
    if backend == "cuda":
        return torch.device("cuda")
    if backend == "tpu" and prefer_xla:
        xm = _load_xla_model()
        if xm is not None and callable(getattr(xm, "xla_device", None)):
            try:
                return xm.xla_device()
            except Exception:
                logging.debug("Falling back to CPU device after xm.xla_device error.")
    return torch.device("cpu")


def synchronize_and_empty_cache() -> str:
    """Flush pending accelerator work and release cached device memory.

    Returns:
        Detected accelerator backend name ('cuda', 'tpu', or 'cpu')
    """
    gc.collect()
    backend = get_accelerator_type()

    if backend == "cuda":
        cuda_mod = getattr(torch, "cuda", None)
        sync_fn = getattr(cuda_mod, "synchronize", None)
        if callable(sync_fn):
            try:
                sync_fn()
            except Exception:
                pass
        empty_fn = getattr(cuda_mod, "empty_cache", None)
        if callable(empty_fn):
            try:
                empty_fn()
            except Exception:
                pass
    elif backend == "tpu":
        xm = _load_xla_model()
        if xm is not None:
            for hook_name in ("mark_step", "wait_device_ops"):
                hook = getattr(xm, hook_name, None)
                if callable(hook):
                    try:
                        hook()
                    except Exception:
                        pass

    return backend


def seed_accelerator(seed: int) -> None:
    """Seed PyTorch and active accelerator RNGs safely across CUDA/TPU/CPU."""
    manual_seed_fn = getattr(torch, "manual_seed", None)
    if callable(manual_seed_fn):
        manual_seed_fn(seed)

    if is_cuda_available():
        seed_all_fn = getattr(getattr(torch, "cuda", None), "manual_seed_all", None)
        if callable(seed_all_fn):
            try:
                seed_all_fn(seed)
            except Exception:
                pass
    elif is_tpu_available():
        xm = _load_xla_model()
        set_rng_fn = getattr(xm, "set_rng_state", None) if xm is not None else None
        if callable(set_rng_fn):
            try:
                set_rng_fn(seed)
            except Exception:
                pass
