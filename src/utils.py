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

import random

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from transformers import set_seed


def fix_seed(seed: int = 0) -> None:
    """Fix random seed

    Args:
        seed: Seed value to fix
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def get_metrics(scores: list[float], labels: list[int]) -> tuple[float, float, float]:
    """Calculate evaluation metrics

    Args:
        scores: List of scores
        labels: List of labels (1: membership, 0: non-membership)

    Returns:
        auroc: AUROC
        fpr95: FPR when TPR is 95%
        tpr05: TPR when FPR is 5%
    """
    # If only one class is present in labels, metrics are undefined
    if len(set(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    try:
        fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    except IndexError:
        fpr95 = float("nan")
    try:
        tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    except IndexError:
        tpr05 = float("nan")
    return auroc, fpr95, tpr05
