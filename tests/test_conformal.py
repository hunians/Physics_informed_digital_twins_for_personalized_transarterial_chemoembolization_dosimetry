import numpy as np

from ecpc_pinn.evaluation.conformal import PhysicsAwareConformal


def test_interval_expands_with_residual() -> None:
    target = np.array([0.0, 1.0, 2.0])
    prediction = np.array([0.1, 0.8, 2.2])
    residual = np.array([0.0, 0.1, 0.2])
    model = PhysicsAwareConformal().fit(target, prediction, residual)
    interval = model.predict(np.array([1.0, 1.0]), np.array([0.0, 2.0]))
    widths = interval.upper - interval.lower
    assert widths[1] > widths[0]
