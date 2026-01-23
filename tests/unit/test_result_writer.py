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

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.config import Config
from src.result_writer import ResultWriter


def get_fixture_path(filename: str) -> Path:
    """Get the fixture file path for tests"""
    current_dir = Path(__file__).parent.parent
    return current_dir / "fixtures" / filename


@pytest.fixture
def config() -> Config:
    """Load valid config for testing"""
    return Config(get_fixture_path("valid_config.yaml"))


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    """Create sample results DataFrame"""
    return pd.DataFrame({
        "method_name": ["loss", "zlib", "mink_0.1"],
        "auroc": [0.75, 0.72, 0.78],
        "fpr95": [0.45, 0.50, 0.40],
        "tpr05": [0.30, 0.25, 0.35],
    })


@pytest.fixture
def sample_detailed_results() -> list[dict]:
    """Create sample detailed results"""
    return [
        {"method_name": "loss", "auroc": 0.75, "fpr95": 0.45, "tpr05": 0.30, "scores": [0.1, 0.9, 0.3, 0.8]},
        {"method_name": "zlib", "auroc": 0.72, "fpr95": 0.50, "tpr05": 0.25, "scores": [0.2, 0.85, 0.35, 0.75]},
        {"method_name": "mink_0.1", "auroc": 0.78, "fpr95": 0.40, "tpr05": 0.35, "scores": [0.15, 0.88, 0.32, 0.82]},
    ]


@pytest.fixture
def sample_labels() -> list[int]:
    """Create sample labels"""
    return [1, 1, 0, 0]


@pytest.fixture
def sample_data_stats() -> dict:
    """Create sample data statistics"""
    return {
        "num_samples": 4,
        "num_members": 2,
        "num_nonmembers": 2,
    }


@pytest.fixture
def sample_cache_stats() -> dict:
    """Create sample cache statistics"""
    return {
        "hits": 10,
        "misses": 5,
        "hit_rate": 0.67,
    }


class TestResultWriter:
    """Tests for ResultWriter class"""

    def test_initialization(self, tmp_path: Path, config: Config) -> None:
        """Test ResultWriter initialization"""
        start_time = datetime.now()
        writer = ResultWriter(tmp_path, config, start_time)

        assert writer.run_dir == tmp_path
        assert writer.config == config
        assert writer.start_time == start_time

    def test_copy_config(self, tmp_path: Path, config: Config) -> None:
        """Test config file copying"""
        writer = ResultWriter(tmp_path, config, datetime.now())
        copied_path = writer.copy_config()

        assert copied_path.exists()
        assert copied_path.name == "config.yaml"

        with copied_path.open() as f:
            copied_config = yaml.safe_load(f)
        assert copied_config["model"]["model_id"] == "facebook/opt-125m"

    def test_save_results_csv(
        self,
        tmp_path: Path,
        config: Config,
        sample_results_df: pd.DataFrame,
    ) -> None:
        """Test saving results CSV"""
        writer = ResultWriter(tmp_path, config, datetime.now())
        output_path = writer.save_results_csv(sample_results_df)

        assert output_path.exists()
        assert output_path.name == "results.csv"

        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["method_name", "auroc", "fpr95", "tpr05"]

    def test_save_detailed_scores(
        self,
        tmp_path: Path,
        config: Config,
        sample_detailed_results: list[dict],
        sample_labels: list[int],
    ) -> None:
        """Test saving detailed scores"""
        writer = ResultWriter(tmp_path, config, datetime.now())
        output_path = writer.save_detailed_scores(sample_detailed_results, sample_labels)

        assert output_path.exists()
        assert output_path.name == "detailed_scores.csv"

        loaded_df = pd.read_csv(output_path)
        assert "label" in loaded_df.columns
        assert "loss" in loaded_df.columns
        assert "zlib" in loaded_df.columns
        assert len(loaded_df) == 4

    def test_save_metadata(
        self,
        tmp_path: Path,
        config: Config,
        sample_cache_stats: dict,
        sample_data_stats: dict,
    ) -> None:
        """Test saving metadata"""
        start_time = datetime.now()
        writer = ResultWriter(tmp_path, config, start_time)
        end_time = datetime.now()

        json_path = writer.save_metadata(end_time, sample_cache_stats, sample_data_stats)

        assert json_path.exists()
        assert json_path.name == "metadata.json"

        yaml_path = tmp_path / "metadata.yaml"
        assert yaml_path.exists()

        with json_path.open() as f:
            metadata = json.load(f)

        assert "experiment" in metadata
        assert "environment" in metadata
        assert "git" in metadata
        assert "model" in metadata
        assert "data" in metadata
        assert "cache" in metadata

    def test_generate_summary_report(
        self,
        tmp_path: Path,
        config: Config,
        sample_results_df: pd.DataFrame,
        sample_detailed_results: list[dict],
        sample_data_stats: dict,
    ) -> None:
        """Test generating summary report"""
        writer = ResultWriter(tmp_path, config, datetime.now())
        report_path = writer.generate_summary_report(
            sample_results_df, sample_detailed_results, sample_data_stats
        )

        assert report_path.exists()
        assert report_path.name == "report.txt"

        content = report_path.read_text()
        assert "FAST-MIA EVALUATION REPORT" in content
        assert "MODEL CONFIGURATION" in content
        assert "DATA CONFIGURATION" in content
        assert "METHODS EVALUATED" in content
        assert "RESULTS SUMMARY" in content
        assert "BEST PERFORMERS" in content

    def test_save_default(
        self,
        tmp_path: Path,
        config: Config,
        sample_results_df: pd.DataFrame,
        sample_detailed_results: list[dict],
        sample_data_stats: dict,
    ) -> None:
        """Test save_default saves config, results.csv, and report.txt"""
        writer = ResultWriter(tmp_path, config, datetime.now())
        paths = writer.save_default(
            sample_results_df, sample_detailed_results, sample_data_stats
        )

        assert "config" in paths
        assert "results_csv" in paths
        assert "report" in paths
        assert len(paths) == 3

        assert paths["config"].exists()
        assert paths["results_csv"].exists()
        assert paths["report"].exists()

    def test_save_all(
        self,
        tmp_path: Path,
        config: Config,
        sample_results_df: pd.DataFrame,
        sample_detailed_results: list[dict],
        sample_labels: list[int],
        sample_cache_stats: dict,
        sample_data_stats: dict,
    ) -> None:
        """Test save_all saves all outputs"""
        start_time = datetime.now()
        writer = ResultWriter(tmp_path, config, start_time)
        end_time = datetime.now()

        paths = writer.save_all(
            results_df=sample_results_df,
            results=sample_detailed_results,
            labels=sample_labels,
            cache_stats=sample_cache_stats,
            data_stats=sample_data_stats,
            end_time=end_time,
        )

        assert "config" in paths
        assert "results_csv" in paths
        assert "detailed_scores" in paths
        assert "metadata" in paths
        assert "report" in paths
        assert len(paths) == 5

        for path in paths.values():
            assert path.exists()
