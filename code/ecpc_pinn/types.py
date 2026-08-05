from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class VesselSegment:
    identifier: int
    parent: int
    length_m: float
    radius_m: float
    terminal_resistance: float


@dataclass(frozen=True)
class PhysiologicalParameters:
    density: float = 1060.0
    viscosity: float = 3.5e-3
    diffusion_um2_s: float = 287.0
    elimination_per_hour: float = 0.04
    uptake_um_s: float = 9.0e-4
    release_per_hour: float = 1.0 / 1500.0
    proliferation_per_day: float = 0.03
    cell_motility: float = 0.01
    carrying_capacity: float = 1.0
    maximum_kill_per_hour: float = 0.3
    ec50_ng_ml: float = 50.0
    hill_coefficient: float = 2.0
    habr_gain: float = 1.5
    transition_sharpness: float = 10.0
    map_mmhg: float = 95.0


@dataclass
class PatientBatch:
    coordinates: Tensor
    time: Tensor
    velocity: Tensor
    pressure: Tensor
    concentration: Tensor
    density: Tensor
    embolization: Tensor
    response: Tensor

    def to(self, device: torch.device) -> "PatientBatch":
        return PatientBatch(*(value.to(device) for value in self.__dict__.values()))


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    metrics: Path
