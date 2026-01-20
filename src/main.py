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

import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Import required modules
from .config import Config
from .data_loader import DataLoader
from .evaluator import Evaluator
from .methods.factory import MethodFactory
from .model_loader import ModelLoader
from .utils import fix_seed


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Evaluation of membership inference attacks using vLLM"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-cache-size", type=int, default=1000, help="Maximum cache size"
    )
    parser.add_argument(
        "--detailed-report",
        action="store_true",
        help="Generate detailed report with metadata, scores, and visualizations",
    )
    args = parser.parse_args()
    start_time = datetime.now()

    # Log settings
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load environment variables
    load_dotenv()

    # Fix random seed
    fix_seed(args.seed)

    # Load config
    config = Config(args.config)

    # Get text length (word count)
    text_length = config.data.get("text_length", 32)

    # If using WikiMIA dataset
    data_path = config.data.get("data_path")
    if data_path == "swj0419/WikiMIA":
        logging.info(f"WikiMIA dataset will be used with text length {text_length}")
        data_loader = DataLoader.load_wikimia(text_length)
    elif data_path is not None and data_path.startswith("iamgroot42/mimir"):
        logging.info(f"Mimir dataset will be used with text length {text_length}")
        if text_length != 200:
            raise ValueError(
                f"Mimir dataset only supports text length 200, but '{text_length}' "
                "was provided. Update data.text_length in your config."
            )

        token = os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError(
                "Environment variable HUGGINGFACE_TOKEN is not set. "
                "Create a .env file in the project root (see docs/how-to-use.md) "
                "with your Hugging Face token before using the Mimir dataset."
            )

        data_loader = DataLoader.load_mimir(
            data_path=config.data.get("data_path"),
            token=token,
        )
    else:
        # Initialize normal data loader
        data_config = config.data
        data_loader = DataLoader(
            data_path=data_config.get("data_path"),
            data_format=data_config.get("format", "csv"),
            text_column=data_config.get("text_column", "text"),
            label_column=data_config.get("label_column", "label"),
        )

    # Initialize model loader
    model_loader = ModelLoader(config.model)

    # Initialize methods
    methods = MethodFactory.create_methods(config.methods)

    # Initialize and run evaluator
    evaluator = Evaluator(
        data_loader, model_loader, methods, max_cache_size=args.max_cache_size
    )

    eval_result = evaluator.evaluate(config=config)
    end_time = datetime.now()

    # Show results
    logging.info(
        "\nEvaluation Results:\n"
        + "=" * 50
        + "\n"
        + eval_result.results_df.to_string(index=False)
        + "\n"
        + "=" * 50
    )

    # Save results
    output_dir = Path(config.config.get("output_dir", "./results"))
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / start_time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    if args.detailed_report:
        # Detailed report mode: metadata, scores, and visualizations
        from .result_writer import ResultWriter
        from .visualizer import Visualizer

        result_writer = ResultWriter(run_dir, config, start_time)

        saved_paths = result_writer.save_all(
            results_df=eval_result.results_df,
            results=eval_result.detailed_results,
            labels=eval_result.labels,
            cache_stats=eval_result.cache_stats,
            data_stats=eval_result.data_stats,
            end_time=end_time,
        )

        # Generate visualizations
        visualizer = Visualizer(run_dir)
        figure_paths = visualizer.generate_all_plots(
            eval_result.detailed_results,
            eval_result.labels,
        )
        saved_paths.update(figure_paths)

        logging.info(f"\nAll outputs saved to: {run_dir}")
        logging.info("Output files:")
        for name, path in saved_paths.items():
            logging.info(f"  - {name}: {path}")
    else:
        # Default mode: simple output (results.csv and config.yaml only)
        output_path = run_dir / "results.csv"
        eval_result.results_df.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")

        config_copy_path = run_dir / "config.yaml"
        shutil.copy(config.config_path, config_copy_path)
        logging.info(f"Config copied to {config_copy_path}")


if __name__ == "__main__":
    main()
