import argparse
from pathlib import Path

import numpy as np

from ecpc_pinn.metrics.statistics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(prog="ecpc-evaluate")
    parser.add_argument("predictions", type=Path)
    arguments = parser.parse_args()
    values = np.loadtxt(arguments.predictions, delimiter=",", skiprows=1)
    result = classification_metrics(values[:, 0], values[:, 1])
    print(result)


if __name__ == "__main__":
    main()
