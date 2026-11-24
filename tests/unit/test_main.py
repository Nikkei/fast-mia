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

    @pytest.mark.parametrize("include_text", [False, True])
    def test_audit_mode_include_text_in_scores_opt_in(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        include_text: bool,
    ) -> None:
        data_path = tmp_path / "audit.csv"
        pd.DataFrame({"text": ["first doc", "second doc"]}).to_csv(
            data_path, index=False
        )
        output_dir = tmp_path / "results"

        data_cfg: dict[str, object] = {
            "data_path": str(data_path),
            "format": "csv",
            "text_column": "text",
            "label_column": None,
        }
        if include_text:
            data_cfg["include_text_in_scores"] = True

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {"model_id": "facebook/opt-125m"},
                    "data": data_cfg,
                    "methods": [{"type": "loss"}],
                    "output_dir": str(output_dir),
                }
            )
        )

        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        mock_outputs = []
        for text in ["first doc", "second doc"]:
            out = mock.MagicMock()
            out.prompt = text
            lp = mock.MagicMock()
            lp.logprob = -0.5
            out.prompt_logprobs = [{0: lp}, {0: lp}]
            out.prompt_token_ids = [0, 0]
            mock_outputs.append(out)

        with mock.patch("src.main.ModelLoader") as mock_model_loader:
            mock_loader_instance = mock_model_loader.return_value
            mock_loader_instance.model.generate.return_value = mock_outputs
            mock_loader_instance.get_sampling_params.return_value = mock.MagicMock()
            main()

        run_dirs = list(output_dir.iterdir())
        assert len(run_dirs) == 1
        scores_df = pd.read_csv(run_dirs[0] / "detailed_scores.csv")
        assert list(scores_df["sample_id"]) == [0, 1]
        if include_text:
            assert list(scores_df["text"]) == ["first doc", "second doc"]
        else:
            assert "text" not in scores_df.columns

