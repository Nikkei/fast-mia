# Copyright (c) 2026 Nikkei Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import zlib
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion
from vllm import LLM
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .base import BaseMethod

_SUPPORTED_FEATURES = ("tfidf", "length", "zlib", "digit_ratio")


class BlindMethod(BaseMethod):
    """Model-free blind baseline for detecting dataset distribution shift.

    Computes membership scores purely from input text statistics or cross-
    validated bag-of-ngrams classification without querying the target LLM
    (Das, Zhang & Tramèr, IEEE S&P 2025; arXiv:2406.16201). High blind AUROC
    warns that member and non-member splits are separable by surface/temporal
    artifacts rather than genuine model memorization.
    """

    requires_labels: bool = True
    requires_sampling_params: bool = False

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize Blind baseline method.

        Args:
            method_config: Method configuration
                - features: Feature mode ('tfidf', 'length', 'zlib',
                  'digit_ratio'; default 'tfidf')
                - n_splits: Number of stratified CV folds when features='tfidf'
                  (default 5)
                - max_features: Max TF-IDF features per analyzer (default 5000)
                - c: Inverse L2 regularization strength for LogisticRegression
                  (default 1.0)
                - seed: Random seed for CV fold shuffling (default 42)

        Raises:
            ValueError: If 'features', 'n_splits', 'max_features', or 'c' is invalid
        """
        config = dict(method_config) if method_config is not None else {}
        features = str(config.get("features", "tfidf"))
        if features not in _SUPPORTED_FEATURES:
            supported = ", ".join(_SUPPORTED_FEATURES)
            raise ValueError(
                f"Unsupported BlindMethod features '{features}'. "
                f"Supported features: {supported}"
            )

        n_splits = int(config.get("n_splits", 5))
        max_features = int(config.get("max_features", 5000))
        reg_c = float(config.get("c", 1.0))
        seed = int(config.get("seed", 42))

        if n_splits < 2:
            raise ValueError("BlindMethod requires 'n_splits' >= 2.")
        if max_features < 1:
            raise ValueError("BlindMethod requires 'max_features' >= 1.")
        if reg_c <= 0.0:
            raise ValueError("BlindMethod requires positive 'c'.")

        method_name = f"blind_{features}"
        super().__init__(method_name, config)
        self.features = features
        self.n_splits = n_splits
        self.max_features = max_features
        self.c = reg_c
        self.seed = seed
        self.requires_labels = features == "tfidf"

    @staticmethod
    def _surface_score(text: str, features: str) -> float:
        """Compute unsupervised surface heuristic score for a single text."""
        if features == "length":
            return float(len(text))
        if features == "zlib":
            return float(len(zlib.compress(text.encode("utf-8"))))
        if features == "digit_ratio":
            if not text:
                return 0.0
            digits = sum(ch.isdigit() for ch in text)
            return float(digits) / float(len(text))
        raise ValueError(f"Unsupported surface feature '{features}'.")

    def process_output(self, output: RequestOutput) -> float:
        """Calculate single-text surface score from ``output.prompt``.

        Note: ``features='tfidf'`` requires cross-validation across the dataset
        via ``run()`` and is not supported in single-output mode.

        Args:
            output: Model output carrying ``prompt``

        Returns:
            Surface feature score

        Raises:
            NotImplementedError: When ``features == 'tfidf'``
        """
        if self.features == "tfidf":
            raise NotImplementedError(
                "process_output is not supported in BlindMethod when "
                "features='tfidf'. Use run() instead."
            )
        prompt = getattr(output, "prompt", "") or ""
        return self._surface_score(str(prompt), self.features)

    def _build_vectorizer(self) -> FeatureUnion:
        """Build combined word + character n-gram TF-IDF extractor."""
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=self.max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=self.max_features,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

    def _score_tfidf_cv(self, texts: list[str], labels: list[int]) -> list[float]:
        """Compute out-of-fold predicted membership probabilities."""
        y = np.asarray(labels, dtype=int)
        if y.shape[0] != len(texts):
            raise ValueError(
                "BlindMethod received mismatched lengths for 'texts' and 'labels'."
            )

        unique_classes, counts = np.unique(y, return_counts=True)
        if unique_classes.size < 2 or int(counts.min()) < 2:
            raise ValueError(
                "BlindMethod with features='tfidf' requires at least 2 samples "
                "in both member (1) and non-member (0) classes."
            )

        effective_splits = min(self.n_splits, int(counts.min()))
        splitter = StratifiedKFold(
            n_splits=effective_splits, shuffle=True, random_state=self.seed
        )
        oof_scores = np.zeros(len(texts), dtype=np.float64)

        for train_idx, test_idx in splitter.split(texts, y):
            train_texts = [texts[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            y_train = y[train_idx]

            vectorizer = self._build_vectorizer()
            try:
                x_train = vectorizer.fit_transform(train_texts)
                x_test = vectorizer.transform(test_texts)
            except ValueError:
                oof_scores[test_idx] = 0.5
                continue

            clf = LogisticRegression(
                C=self.c,
                max_iter=1000,
                random_state=self.seed,
            )
            clf.fit(x_train, y_train)
            pos_col = int(np.flatnonzero(clf.classes_ == 1)[0])
            oof_scores[test_idx] = clf.predict_proba(x_test)[:, pos_col]

        return oof_scores.tolist()

    def run(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        model: LLM = None,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """Run model-free blind scoring over input texts without calling the LLM.

        Args:
            texts: List of input texts
            labels: Membership labels (required when features='tfidf')
            model: Unused LLM reference (accepted for Evaluator compatibility)
            lora_request: Unused LoRA request (accepted for Evaluator compatibility)
            data_config: Optional data configuration

        Returns:
            List of blind baseline scores
        """
        del model, lora_request
        if not texts:
            return []

        if data_config and not data_config.get("space_delimited_language", True):
            texts = [text.replace(" ", "") for text in texts]

        if self.features != "tfidf":
            return [self._surface_score(text, self.features) for text in texts]

        if labels is None:
            raise ValueError("BlindMethod with features='tfidf' requires 'labels'.")

        return self._score_tfidf_cv(texts, labels)
