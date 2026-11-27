from pathlib import Path

from src.visualizer import Visualizer


class TestVisualizer:
    def test_generate_all_plots_includes_log_roc(self, tmp_path: Path):
        visualizer = Visualizer(tmp_path)
        results = [
            {
                "method_name": "loss",
                "scores": [0.95, 0.85, 0.2, 0.1],
                "auroc": 1.0,
                "fpr95": 0.0,
                "tpr05": 1.0,
                "tpr01": 1.0,
            },
            {
                "method_name": "zlib",
                "scores": [0.7, 0.6, 0.5, 0.2],
                "auroc": 0.75,
                "fpr95": 0.5,
                "tpr05": 0.5,
                "tpr01": 0.5,
            },
        ]
        labels = [1, 1, 0, 0]

        paths = visualizer.generate_all_plots(results, labels)
        assert set(paths.keys()) == {
            "roc_curves",
            "roc_curves_log",
            "score_distributions",
            "metrics_comparison",
        }
        assert paths["roc_curves_log"].name == "roc_curves_log.png"
        for path in paths.values():
            assert path.exists()
            assert path.stat().st_size > 0
