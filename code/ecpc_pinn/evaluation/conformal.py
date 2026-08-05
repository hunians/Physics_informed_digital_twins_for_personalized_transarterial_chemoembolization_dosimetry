import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PredictionInterval:
    lower: FloatArray
    upper: FloatArray


class PhysicsAwareConformal:
    def __init__(self, alpha: float = 0.10, sensitivity: float = 0.42) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        self.alpha = alpha
        self.sensitivity = sensitivity
        self.quantile: float | None = None

    def fit(
        self, target: FloatArray, prediction: FloatArray, residual: FloatArray
    ) -> "PhysicsAwareConformal":
        scores = np.abs(target - prediction) * (1.0 + self.sensitivity * np.abs(residual))
        rank = min(math.ceil((1.0 - self.alpha) * (len(scores) + 1)), len(scores))
        self.quantile = float(np.partition(scores, rank - 1)[rank - 1])
        return self

    def predict(self, prediction: FloatArray, residual: FloatArray) -> PredictionInterval:
        if self.quantile is None:
            raise RuntimeError("calibrator has not been fitted")
        width = self.quantile * (1.0 + self.sensitivity * np.abs(residual))
        return PredictionInterval(prediction - width, prediction + width)

    def coverage(self, target: FloatArray, interval: PredictionInterval) -> float:
        return float(np.mean((target >= interval.lower) & (target <= interval.upper)))
