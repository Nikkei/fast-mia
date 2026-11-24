import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import yaml

from src.main import main


class TestMainAuditMode:
    """Tests for main() audit-mode startup validation."""

    def test_rejects_label_dependent_method_before_loading_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data_path = tmp_path / "audit.csv"
        pd.DataFrame({"text": ["sample one", "sample two"]}).to_csv(
            data_path, index=False
        )

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {"model_id": "facebook/opt-125m"},
                    "data": {
                        "data_path": str(data_path),
                        "format": "csv",
                        "text_column": "text",
                        "label_column": None,
                    },
                    "methods": [
                        {
                            "type": "conrecall",
                            "params": {"prefix_type": "non_member", "num_shots": 1},
                        }
                    ],
                }
            )
        )

        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        with mock.patch("src.main.ModelLoader") as mock_model_loader:
            with pytest.raises(ValueError, match="unlabeled audit mode"):
                main()
            mock_model_loader.assert_not_called()
