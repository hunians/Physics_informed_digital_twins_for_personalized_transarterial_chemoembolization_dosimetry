import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecpc_pinn.losses.objectives import LossTerms, coupled_loss
from ecpc_pinn.models.system import ECPCPINN
from ecpc_pinn.types import PatientBatch, PhysiologicalParameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingState:
    stage: str
    epoch: int
    global_step: int
    seed: int
    best_value: float


class CurriculumTrainer:
    def __init__(
        self,
        model: ECPCPINN,
        parameters: PhysiologicalParameters,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.parameters = parameters
        self.seed = seed
        self.state = TrainingState("hemodynamics", 0, 0, seed, float("inf"))
        set_seed(seed)

    def optimizer(self, stage: str) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters_for_stage(stage), lr=self.learning_rate)

    @property
    def learning_rate(self) -> float:
        return 1e-3

    def step(self, batch: PatientBatch, optimizer: torch.optim.Optimizer) -> LossTerms:
        optimizer.zero_grad(set_to_none=True)
        output = self.model(batch.coordinates, batch.time, batch.embolization)
        losses = coupled_loss(output, batch, self.parameters)
        losses.total.backward()
        optimizer.step()
        self.state.global_step += 1
        return losses

    def train_stage(self, batch: PatientBatch, stage: str, epochs: int) -> list[float]:
        self.state.stage = stage
        optimizer = self.optimizer(stage)
        history: list[float] = []
        for epoch in range(epochs):
            self.state.epoch = epoch
            losses = self.step(batch, optimizer)
            value = float(losses.total.detach())
            self.state.best_value = min(self.state.best_value, value)
            history.append(value)
        return history


def atomic_checkpoint(path: Path, model: ECPCPINN, state: TrainingState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "state": asdict(state),
            "seed": state.seed,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        temporary,
    )
    os.replace(temporary, path)


def restore_checkpoint(path: Path, model: ECPCPINN) -> TrainingState:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    state = TrainingState(**payload["state"])
    set_seed(state.seed)
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return state


def write_metrics(path: Path, values: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
