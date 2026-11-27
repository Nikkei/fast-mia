from pathlib import Path
import subprocess

from src.config import Config
from src.methods.factory import MethodFactory

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCloudTpuAssets:
    def test_tpu_sample_config_is_valid_and_instantiates_methods(self) -> None:
        tpu_cfg_path = REPO_ROOT / "config" / "tpu-sample.yaml"
        cfg = Config(tpu_cfg_path)
        assert cfg.model["dtype"] == "bfloat16"
        assert cfg.model["tensor_parallel_size"] == 1
        assert cfg.model["max_model_len"] == 512
        methods = MethodFactory.create_methods(cfg.methods)
        assert len(methods) == 4

    def test_tpu_shell_scripts_pass_bash_syntax_check(self) -> None:
        for script_name in ("submit_tpu_job.sh", "setup_tpu_instance.sh"):
            script_path = REPO_ROOT / "gcp" / script_name
            assert script_path.is_file(), f"Missing {script_path}"
            res = subprocess.run(
                ["bash", "-n", str(script_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, res.stderr
