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

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


class Visualizer:
    """Visualizer for MIA evaluation results"""

    def __init__(self, output_dir: Path) -> None:
        """Initialize visualizer

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_roc_curves(
        self,
        results: list[dict[str, Any]],
        labels: list[int],
    ) -> Path:
        """Plot ROC curves for all methods

        Args:
            results: List of method results containing scores
            labels: Ground truth labels

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for i, result in enumerate(results):
            method_name = result["method_name"]
            scores = result["scores"]

            if len(set(labels)) < 2:
                continue

            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = result["auroc"]

            ax.plot(
                fpr,
                tpr,
                color=colors[i],
                lw=2,
                label=f"{method_name} (AUC={auroc:.3f})",
            )

        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves for MIA Methods", fontsize=14)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        output_path = self.figures_dir / "roc_curves.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logging.info(f"ROC curves saved to {output_path}")
        return output_path

    def plot_score_distributions(
        self,
        results: list[dict[str, Any]],
        labels: list[int],
    ) -> Path:
        """Plot score distributions for each method

        Args:
            results: List of method results containing scores
            labels: Ground truth labels

        Returns:
            Path to saved figure
        """
        n_methods = len(results)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_methods == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        labels_arr = np.array(labels)

        for i, result in enumerate(results):
            ax = axes[i]
            method_name = result["method_name"]
            scores = np.array(result["scores"])

            member_scores = scores[labels_arr == 1]
            nonmember_scores = scores[labels_arr == 0]

            ax.hist(
                member_scores,
                bins=30,
                alpha=0.6,
                label="Member",
                color="steelblue",
                density=True,
            )
            ax.hist(
                nonmember_scores,
                bins=30,
                alpha=0.6,
                label="Non-member",
                color="coral",
                density=True,
            )

            ax.set_xlabel("Score", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"{method_name}\nAUROC={result['auroc']:.3f}", fontsize=11)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()

        output_path = self.figures_dir / "score_distributions.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logging.info(f"Score distributions saved to {output_path}")
        return output_path

    def plot_metrics_comparison(
        self,
        results: list[dict[str, Any]],
    ) -> Path:
        """Plot bar chart comparing metrics across methods

        Args:
            results: List of method results containing metrics

        Returns:
            Path to saved figure
        """
        methods = [r["method_name"] for r in results]
        aurocs = [r["auroc"] for r in results]
        fpr95s = [r["fpr95"] for r in results]
        tpr05s = [r["tpr05"] for r in results]

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(12, len(methods) * 1.2), 6))

        bars1 = ax.bar(x - width, aurocs, width, label="AUROC", color="steelblue")
        bars2 = ax.bar(x, fpr95s, width, label="FPR@95%TPR", color="coral")
        bars3 = ax.bar(x + width, tpr05s, width, label="TPR@5%FPR", color="seagreen")

        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Metrics Comparison Across MIA Methods", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        def add_labels(bars: Any) -> None:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)

        fig.tight_layout()

        output_path = self.figures_dir / "metrics_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logging.info(f"Metrics comparison saved to {output_path}")
        return output_path

    def generate_all_plots(
        self,
        results: list[dict[str, Any]],
        labels: list[int],
    ) -> dict[str, Path]:
        """Generate all visualization plots

        Args:
            results: List of method results
            labels: Ground truth labels

        Returns:
            Dictionary mapping plot names to file paths
        """
        paths = {}

        paths["roc_curves"] = self.plot_roc_curves(results, labels)
        paths["score_distributions"] = self.plot_score_distributions(results, labels)
        paths["metrics_comparison"] = self.plot_metrics_comparison(results)

        return paths
