import argparse
from pathlib import Path

import yaml

from ecpc_pinn.data.dataset import synthetic_patient
from ecpc_pinn.models.system import ECPCPINN
from ecpc_pinn.training.engine import CurriculumTrainer, atomic_checkpoint
from ecpc_pinn.types import PhysiologicalParameters


def main() -> None:
    parser = argparse.ArgumentParser(prog="ecpc-train")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/main.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model.pt"))
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    model = ECPCPINN(config["hidden_width"], config["hidden_layers"])
    batch = synthetic_patient(config["collocation_points"], config["seed"])
    trainer = CurriculumTrainer(
        model, PhysiologicalParameters(), config["learning_rate"], config["seed"]
    )
    for stage, epochs in config["epochs"].items():
        trainer.train_stage(batch, stage, epochs)
    atomic_checkpoint(arguments.output, model, trainer.state)


if __name__ == "__main__":
    main()
