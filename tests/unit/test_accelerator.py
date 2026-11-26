import os
from types import SimpleNamespace
from unittest import mock

from src import accelerator


class TestAcceleratorDetection:
    def test_defaults_to_cpu_when_mocked_torch_is_not_explicit_true(self):
        # MockModule("torch") returns MagicMock for torch.cuda.is_available();
        # accelerator helpers must NOT treat MagicMock truthiness as CUDA.
        with mock.patch.object(accelerator, "_load_xla_model", return_value=None):
            assert accelerator.is_cuda_available() is False
            assert accelerator.is_tpu_available() is False
            assert accelerator.get_accelerator_type() == "cpu"

    def test_detects_cuda_when_is_available_returns_true(self):
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=mock.MagicMock(),
        )
        with (
            mock.patch.object(accelerator.torch, "cuda", fake_cuda, create=True),
            mock.patch.object(
                accelerator.torch,
                "device",
                side_effect=lambda name: SimpleNamespace(type=name),
            ),
        ):
            assert accelerator.is_cuda_available() is True
            assert accelerator.get_accelerator_type() == "cuda"
            assert accelerator.get_torch_device().type == "cuda"
            accelerator.seed_accelerator(99)
            fake_cuda.manual_seed_all.assert_called_once_with(99)

    def test_detects_tpu_via_xla_supported_devices(self):
        fake_xm = SimpleNamespace(
            get_xla_supported_devices=lambda kind: ["xla:0"] if kind == "TPU" else [],
            xla_device=lambda: SimpleNamespace(type="xla"),
            mark_step=mock.MagicMock(),
            wait_device_ops=mock.MagicMock(),
            set_rng_state=mock.MagicMock(),
        )
        fake_cuda = SimpleNamespace(is_available=lambda: False)
        with (
            mock.patch.object(accelerator.torch, "cuda", fake_cuda, create=True),
            mock.patch.object(accelerator, "_load_xla_model", return_value=fake_xm),
        ):
            assert accelerator.is_tpu_available() is True
            assert accelerator.get_accelerator_type() == "tpu"
            dev = accelerator.get_torch_device()
            assert dev.type == "xla"

            backend = accelerator.synchronize_and_empty_cache()
            assert backend == "tpu"
            fake_xm.mark_step.assert_called_once()
            fake_xm.wait_device_ops.assert_called_once()

            accelerator.seed_accelerator(123)
            fake_xm.set_rng_state.assert_called_once_with(123)

    def test_detects_tpu_via_pjrt_env_when_xla_installed(self):
        fake_xm = SimpleNamespace(get_xla_supported_devices=lambda _kind: [])
        fake_cuda = SimpleNamespace(is_available=lambda: False)
        with (
            mock.patch.object(accelerator.torch, "cuda", fake_cuda, create=True),
            mock.patch.object(accelerator, "_load_xla_model", return_value=fake_xm),
            mock.patch.dict(os.environ, {"PJRT_DEVICE": "TPU"}),
        ):
            assert accelerator.is_tpu_available() is True
            assert accelerator.get_accelerator_type() == "tpu"

    def test_synchronize_and_empty_cache_calls_cuda_hooks(self):
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            synchronize=mock.MagicMock(),
            empty_cache=mock.MagicMock(),
        )
        with mock.patch.object(accelerator.torch, "cuda", fake_cuda, create=True):
            backend = accelerator.synchronize_and_empty_cache()
            assert backend == "cuda"
            fake_cuda.synchronize.assert_called_once()
            fake_cuda.empty_cache.assert_called_once()
