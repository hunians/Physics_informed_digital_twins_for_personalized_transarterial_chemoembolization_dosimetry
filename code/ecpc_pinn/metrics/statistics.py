from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def auc_score(target: FloatArray, score: FloatArray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    positive = target == 1
    count_positive = int(positive.sum())
    count_negative = len(target) - count_positive
    if count_positive == 0 or count_negative == 0:
        raise ValueError("AUC requires both classes")
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - count_positive * (count_positive + 1) / 2) / (
        count_positive * count_negative
    )


@dataclass(frozen=True)
class ClassificationMetrics:
    auc: float
    sensitivity: float
    specificity: float
    f1: float


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    rmse: float
    mae: float


def classification_metrics(
    target: FloatArray, score: FloatArray, threshold: float = 0.5
) -> ClassificationMetrics:
    predicted = score >= threshold
    positive = target == 1
    negative = ~positive
    sensitivity = float((predicted & positive).sum() / max(positive.sum(), 1))
    specificity = float((~predicted & negative).sum() / max(negative.sum(), 1))
    return ClassificationMetrics(
        auc_score(target, score),
        sensitivity,
        specificity,
        float(
            2.0
            * (predicted & positive).sum()
            / max(
                2 * (predicted & positive).sum()
                + (predicted & negative).sum()
                + (~predicted & positive).sum(),
                1,
            )
        ),
    )


def regression_metrics(target: FloatArray, predicted: FloatArray) -> RegressionMetrics:
    residual = target - predicted
    denominator = np.square(target - target.mean()).sum()
    r2 = 1.0 - np.square(residual).sum() / denominator
    return RegressionMetrics(
        float(r2),
        float(np.sqrt(np.square(residual).mean())),
        float(np.abs(residual).mean()),
    )


def relative_l2(target: FloatArray, predicted: FloatArray) -> float:
    return float(np.linalg.norm(target - predicted) / np.linalg.norm(target))


def bootstrap_auc(
    target: FloatArray, score: FloatArray, resamples: int = 1000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, len(target), len(target))
        if np.unique(target[indices]).size == 2:
            values.append(auc_score(target[indices], score[indices]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
