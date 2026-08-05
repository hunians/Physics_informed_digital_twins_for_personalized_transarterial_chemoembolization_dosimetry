#!/usr/bin/env bash
set -euo pipefail
python -m ecpc_pinn.cli.train --config configs/experiment/main.yaml --output outputs/main.pt

