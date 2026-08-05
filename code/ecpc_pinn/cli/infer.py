import argparse
from pathlib import Path

import torch

from ecpc_pinn.data.dataset import synthetic_patient
from ecpc_pinn.models.system import ECPCPINN
from ecpc_pinn.training.engine import restore_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(prog="ecpc-infer")
    parser.add_argument("model", type=Path)
    arguments = parser.parse_args()
    network = ECPCPINN()
    restore_checkpoint(arguments.model, network)
    batch = synthetic_patient()
    with torch.inference_mode():
        output = network(batch.coordinates, batch.time, batch.embolization)
    print(float(torch.sigmoid(output.response_logit)))


if __name__ == "__main__":
    main()
