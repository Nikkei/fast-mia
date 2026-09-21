"""Exercise the driver preflight without a GPU or any cloud operations."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("drivers", "query_status", "succeeds"),
    [
        ("580.65.06", 0, True),
        ("590.48.01\n590.48.01", 0, True),
        (" 580.126.20 ", 0, True),
        ("570.133.20", 0, False),
        ("580.65.06\n570.133.20", 0, False),
        ("N/A", 0, False),
        ("", 0, False),
        ("", 1, False),
    ],
)
def test_driver_preflight(tmp_path, drivers, query_status, succeeds):
    smi = tmp_path / "nvidia-smi"
    smi.write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_DRIVERS"\nexit "$TEST_STATUS"\n')
    smi.chmod(0o755)
    script = Path(__file__).resolve().parents[2] / "gcp" / "check_driver.sh"
    result = subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "TEST_DRIVERS": drivers,
            "TEST_STATUS": str(query_status),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is succeeds
    assert ("check passed" in result.stdout) is succeeds
    if not succeeds:
        assert "ERROR:" in result.stderr
