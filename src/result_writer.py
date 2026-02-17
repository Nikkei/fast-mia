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
import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import Config


def _get_git_info() -> dict[str, str]:
    """Get git repository information"""
    git_info = {}
    try:
        git_info["commit_hash"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        git_info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        git_info["is_dirty"] = (
            subprocess.call(
                ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
            )
            != 0
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_info["commit_hash"] = "unknown"
        git_info["branch"] = "unknown"
        git_info["is_dirty"] = False
    return git_info


class ResultWriter:
    """Writer for saving evaluation results with metadata"""

    def __init__(
        self,
        run_dir: Path,
        config: Config,
        start_time: datetime,
    ) -> None:
        """Initialize result writer

        Args:
            run_dir: Directory to save results (already created by main.py)
            config: Configuration object
            start_time: Experiment start time
        """
        self.run_dir = run_dir
        self.config = config
        self.start_time = start_time

    def _build_metadata(
        self,
        end_time: datetime,
        cache_stats: dict[str, Any],
        data_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Build metadata dictionary

        Args:
            end_time: Experiment end time
            cache_stats: Cache statistics from evaluator
            data_stats: Data statistics (num_samples, num_members, etc.)

        Returns:
            Metadata dictionary
        """
        duration = end_time - self.start_time
        git_info = _get_git_info()

        metadata = {
            "experiment": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "duration_formatted": str(duration).split(".")[0],
            },
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
            },
            "git": git_info,
            "model": {
                "model_id": self.config.model.get("model_id", "unknown"),
                "trust_remote_code": self.config.model.get("trust_remote_code", False),
                "enable_lora": self.config.model.get("enable_lora", False),
            },
            "data": {
                "data_path": self.config.data.get("data_path"),
                "format": self.config.data.get("format", "csv"),
                "text_length": self.config.data.get("text_length"),
                "text_column": self.config.data.get("text_column", "text"),
                "label_column": self.config.data.get("label_column", "label"),
                **data_stats,
            },
            "sampling_parameters": self.config.sampling_parameters,
            "methods": self.config.methods,
            "cache": cache_stats,
        }

        if self.config.lora:
            metadata["lora"] = self.config.lora

        return metadata

    def save_metadata(
        self,
        end_time: datetime,
        cache_stats: dict[str, Any],
        data_stats: dict[str, Any],
    ) -> Path:
        """Save experiment metadata as JSON and YAML

        Args:
            end_time: Experiment end time
            cache_stats: Cache statistics
            data_stats: Data statistics

        Returns:
            Path to saved metadata file
        """
        metadata = self._build_metadata(end_time, cache_stats, data_stats)

        json_path = self.run_dir / "metadata.json"
        with json_path.open("w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        yaml_path = self.run_dir / "metadata.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

        logging.info(f"Metadata saved to {json_path}")
        return json_path

    def save_results_csv(
        self,
        results_df: pd.DataFrame,
    ) -> Path:
        """Save results as CSV

        Args:
            results_df: DataFrame with evaluation results

        Returns:
            Path to saved CSV file
        """
        output_path = self.run_dir / "results.csv"
        results_df.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")
        return output_path

    def save_detailed_scores(
        self,
        results: list[dict[str, Any]],
        labels: list[int],
    ) -> Path:
        """Save detailed scores for each sample

        Args:
            results: List of method results with scores
            labels: Ground truth labels

        Returns:
            Path to saved scores file
        """
        scores_data = {"label": labels}

        for result in results:
            method_name = result["method_name"]
            scores_data[method_name] = result["scores"]

        scores_df = pd.DataFrame(scores_data)
        output_path = self.run_dir / "detailed_scores.csv"
        scores_df.to_csv(output_path, index=False)
        logging.info(f"Detailed scores saved to {output_path}")
        return output_path

    def copy_config(self) -> Path:
        """Copy config file to run directory

        Returns:
            Path to copied config file
        """
        config_copy_path = self.run_dir / "config.yaml"
        shutil.copy(self.config.config_path, config_copy_path)
        logging.info(f"Config copied to {config_copy_path}")
        return config_copy_path

    def generate_summary_report(
        self,
        results_df: pd.DataFrame,
        results: list[dict[str, Any]],
        data_stats: dict[str, Any],
    ) -> Path:
        """Generate a human-readable summary report

        Args:
            results_df: DataFrame with evaluation results
            results: List of method results with detailed info
            data_stats: Data statistics

        Returns:
            Path to saved report file
        """
        lines = [
            "=" * 60,
            "FAST-MIA EVALUATION REPORT",
            "=" * 60,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Run ID: {self.run_dir.name}",
            "",
            "-" * 60,
            "MODEL CONFIGURATION",
            "-" * 60,
            f"Model: {self.config.model.get('model_id', 'unknown')}",
            f"Trust Remote Code: {self.config.model.get('trust_remote_code', False)}",
            "",
            "-" * 60,
            "DATA CONFIGURATION",
            "-" * 60,
            f"Data Path: {self.config.data.get('data_path')}",
            f"Format: {self.config.data.get('format', 'csv')}",
            f"Text Length: {self.config.data.get('text_length')} words",
            f"Total Samples: {data_stats.get('num_samples', 'N/A')}",
            f"Members: {data_stats.get('num_members', 'N/A')}",
            f"Non-members: {data_stats.get('num_nonmembers', 'N/A')}",
            "",
            "-" * 60,
            "METHODS EVALUATED",
            "-" * 60,
        ]

        for method_config in self.config.methods:
            method_type = method_config.get("type", "unknown")
            params = method_config.get("params", {})
            params_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "default"
            lines.append(f"  - {method_type}: {params_str}")

        lines.extend([
            "",
            "-" * 60,
            "RESULTS SUMMARY",
            "-" * 60,
            "",
            results_df.to_string(index=False),
            "",
            "-" * 60,
            "BEST PERFORMERS",
            "-" * 60,
        ])

        valid_results = [r for r in results if not pd.isna(r["auroc"])]
        if valid_results:
            best_auroc = max(valid_results, key=lambda x: x["auroc"])
            lines.append(f"Best AUROC: {best_auroc['method_name']} ({best_auroc['auroc']:.3f})")

            best_fpr95 = min(
                [r for r in valid_results if not pd.isna(r["fpr95"])],
                key=lambda x: x["fpr95"],
                default=None,
            )
            if best_fpr95:
                lines.append(
                    f"Best FPR@95%TPR: {best_fpr95['method_name']} ({best_fpr95['fpr95']:.3f})"
                )

            best_tpr05 = max(
                [r for r in valid_results if not pd.isna(r["tpr05"])],
                key=lambda x: x["tpr05"],
                default=None,
            )
            if best_tpr05:
                lines.append(
                    f"Best TPR@5%FPR: {best_tpr05['method_name']} ({best_tpr05['tpr05']:.3f})"
                )

        lines.extend([
            "",
            "-" * 60,
            "OUTPUT FILES",
            "-" * 60,
            f"Results: {self.run_dir}/results.csv",
            f"Detailed Scores: {self.run_dir}/detailed_scores.csv",
            f"Metadata: {self.run_dir}/metadata.json",
            f"Config: {self.run_dir}/config.yaml",
            f"Figures: {self.run_dir}/figures/",
            "",
            "=" * 60,
        ])

        report_path = self.run_dir / "report.txt"
        with report_path.open("w") as f:
            f.write("\n".join(lines))

        logging.info(f"Summary report saved to {report_path}")
        return report_path

    def save_default(
        self,
        results_df: pd.DataFrame,
        results: list[dict[str, Any]],
        data_stats: dict[str, Any],
    ) -> dict[str, Path]:
        """Save default outputs (config, results.csv, report.txt)

        Args:
            results_df: DataFrame with evaluation results
            results: List of method results with detailed info
            data_stats: Data statistics

        Returns:
            Dictionary mapping output names to file paths
        """
        paths = {}

        paths["config"] = self.copy_config()
        paths["results_csv"] = self.save_results_csv(results_df)
        paths["report"] = self.generate_summary_report(results_df, results, data_stats)

        return paths

    def save_all(
        self,
        results_df: pd.DataFrame,
        results: list[dict[str, Any]],
        labels: list[int],
        cache_stats: dict[str, Any],
        data_stats: dict[str, Any],
        end_time: datetime,
    ) -> dict[str, Path]:
        """Save all outputs

        Args:
            results_df: DataFrame with evaluation results
            results: List of method results with detailed info
            labels: Ground truth labels
            cache_stats: Cache statistics
            data_stats: Data statistics
            end_time: Experiment end time

        Returns:
            Dictionary mapping output names to file paths
        """
        paths = {}

        paths["config"] = self.copy_config()
        paths["results_csv"] = self.save_results_csv(results_df)
        paths["detailed_scores"] = self.save_detailed_scores(results, labels)
        paths["metadata"] = self.save_metadata(end_time, cache_stats, data_stats)
        paths["report"] = self.generate_summary_report(results_df, results, data_stats)

        return paths
