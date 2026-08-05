import torch

from ecpc_pinn.data.dataset import synthetic_patient
from ecpc_pinn.models.system import ECPCPINN
from ecpc_pinn.training.engine import CurriculumTrainer
from ecpc_pinn.types import PhysiologicalParameters


def test_training_updates_parameters() -> None:
    model = ECPCPINN(16, 2)
    batch = synthetic_patient(16, 4)
    before = [value.detach().clone() for value in model.parameters()]
    trainer = CurriculumTrainer(model, PhysiologicalParameters(), seed=4)
    trainer.train_stage(batch, "joint", 2)
    assert any(not torch.equal(left, right) for left, right in zip(before, model.parameters()))
