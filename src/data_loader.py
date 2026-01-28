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

import pandas as pd
from datasets import load_dataset


class DataLoader:
    """Data loader class"""

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet"}

    def __init__(
        self,
        data_path: str | Path | None = None,
        data_format: str = "csv",
        text_column: str = "text",
        label_column: str = "label",
    ) -> None:
        """Initialize the data loader

        Args:
            data_path: Path to the data (file or directory, or dataset name for huggingface format)
            data_format: Data format ("csv", "jsonl", "json", "parquet", "huggingface")
            text_column: Name of the text column
            label_column: Name of the label column
        """
        self.text_column = text_column
        self.label_column = label_column
        self.data = None

        # Load data (only if data_path is specified)
        if data_path is not None:
            logging.info(
                f"Loading data (data_path={data_path}, data_format={data_format})..."
            )
            self.data = self._load_data(data_path, data_format)

    def _load_data(
        self, data_path: str | Path | None, data_format: str
    ) -> pd.DataFrame:
        """Load data

        Args:
            data_path: Path to the data (dataset name for huggingface format)
            data_format: Data format

        Returns:
            DataFrame
        """
        if data_format == "huggingface":
            raise ValueError(
                "Generic Hugging Face datasets are not supported. "
                "Set data_path to 'swj0419/WikiMIA' or 'iamgroot42/mimir_{domain}_{ngram}' "
                "to use the dedicated loaders."
            )

        if not data_path:
            raise ValueError("data_path must be provided")

        data_path = Path(data_path)

        # Validate file extension
        self._validate_file_extension(data_path)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file '{data_path}' was not found. Check data.data_path in your config."
            )

        if data_format == "csv":
            return pd.read_csv(data_path)
        elif data_format == "jsonl":
            return pd.read_json(data_path, lines=True)
        elif data_format == "json":
            return pd.read_json(data_path)
        elif data_format == "parquet":
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

    def get_data(self, text_length: int | None = None) -> tuple[list[str], list[int]]:
        """Get data

        Args:
            text_length: Number of words to split (if None, no split)

        Returns:
            texts: List of texts
            labels: List of labels
        """
        if self.data is None:
            raise ValueError(
                "Data is not loaded. Please call load_data() or load_wikimia()."
            )

        required_columns = {self.text_column, self.label_column}
        missing_columns = sorted(
            [col for col in required_columns if col not in self.data.columns]
        )
        if missing_columns:
            column_list = "', '".join(missing_columns)
            column_label = "Column" if len(missing_columns) == 1 else "Columns"
            raise ValueError(
                f"{column_label} '{column_list}' not found in data. "
                "Check data.text_column and data.label_column in your config."
            )

        if text_length:
            # Split by number of words
            texts = []
            labels = []

            for _, row in self.data.iterrows():
                text = row[self.text_column]
                label = row[self.label_column]

                texts.append(" ".join(text.split()[:text_length]))
                labels.append(label)

            return texts, labels

        else:
            # No split
            texts = self.data[self.text_column].tolist()
            labels = self.data[self.label_column].tolist()

            return texts, labels

    @staticmethod
    def load_wikimia(text_length: int) -> "DataLoader":
        """Load WikiMIA dataset with specified text length

        Args:
            text_length: Text length (one of 32, 64, 128, 256)

        Returns:
            DataLoader instance
        """
        allowed_lengths = {32, 64, 128, 256}
        if text_length not in allowed_lengths:
            raise ValueError(
                "WikiMIA dataset supports text_length values of "
                "32, 64, 128, or 256. "
                f"Received '{text_length}'. Update data.text_length in your config."
            )

        logging.info(f"Loading WikiMIA dataset (text_length={text_length})...")
        dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{text_length}")
        df = dataset.to_pandas()

        # Initialize DataLoader without loading data
        loader = DataLoader(data_path=None)
        loader.data = df
        loader.text_column = "input"
        loader.label_column = "label"

        return loader

    @staticmethod
    def load_mimir(data_path: str, token: str) -> "DataLoader":
        """Load Mimir dataset with fixed text length constraints

        Args:
            data_path: Path to the data (dataset name for huggingface format)
            token: Hugging Face token

        Returns:
            DataLoader instance
        """
        if not token:
            raise ValueError(
                "Hugging Face token is required to load the Mimir dataset. "
                "Set the HUGGINGFACE_TOKEN environment variable (see docs/how-to-use.md)."
            )

        mimir_domains = {
            "ax": "arxiv",
            "dm": "dm_mathematics",
            "gh": "github",
            "hn": "hackernews",
            "pc": "pile_cc",
            "pm": "pubmed_central",
            "we": "wikipedia_(en)",
            "fp": "full_pile",
            "c4": "c4",
            "ta": "temporal_arxiv",
            "tw": "temporal_wiki",
        }
        mimir_ngrams = {
            "702": "ngram_7_0.2",
            "1302": "ngram_13_0.2",
            "1308": "ngram_13_0.8",
            "none": "none",
        }

        elements = data_path.split("_")
        if len(elements) != 3:
            raise ValueError(
                "Mimir dataset path must be in the format 'iamgroot42/mimir_{domain}_{ngram}'. "
                f"Received '{data_path}'."
            )

        dataset_name, domain_key, ngram_key = elements

        if domain_key not in mimir_domains:
            raise ValueError(
                f"Invalid Mimir domain code '{domain_key}'. Supported domains: "
                + ", ".join(sorted(mimir_domains.keys()))
            )

        if ngram_key not in mimir_ngrams:
            raise ValueError(
                f"Invalid Mimir ngram code '{ngram_key}'. Supported ngrams: "
                + ", ".join(sorted(mimir_ngrams.keys()))
            )

        domain = mimir_domains[domain_key]
        ngram = mimir_ngrams[ngram_key]

        logging.info(
            f"Loading Mimir dataset (dataset_name={dataset_name}, domain={domain}, ngram={ngram})..."
        )
        dataset = load_dataset(
            dataset_name, domain, split=ngram, token=token, trust_remote_code=True
        )
        member_texts = dataset["member"]
        nonmember_texts = dataset["nonmember"]
        df = pd.DataFrame(
            {
                "input": member_texts + nonmember_texts,
                "label": [1] * len(member_texts) + [0] * len(nonmember_texts),
            }
        )

        # Initialize DataLoader without loading data
        loader = DataLoader(data_path=None)
        loader.data = df
        loader.text_column = "input"
        loader.label_column = "label"

        return loader

    def _validate_file_extension(self, data_path: Path) -> None:
        """Validate file extension

        Args:
            data_path: Path to the file to validate

        Raises:
            ValueError: Unsupported file extension
        """
        file_extension = data_path.suffix.lower()
        if file_extension not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. "
                f"Allowed extensions are: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
            )
