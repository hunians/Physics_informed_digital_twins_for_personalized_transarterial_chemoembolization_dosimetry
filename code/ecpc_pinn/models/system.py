from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ecpc_pinn.models.networks import NormalizedMLP
from ecpc_pinn.models.physics import post_embolization_velocity
from ecpc_pinn.types import PhysiologicalParameters


@dataclass
class TwinOutput:
    pre_velocity: Tensor
    pressure: Tensor
    post_velocity: Tensor
    concentration: Tensor
    cell_density: Tensor
    response_logit: Tensor


class ECPCPINN(nn.Module):
    def __init__(self, width: int = 256, layers: int = 4) -> None:
        super().__init__()
        self.hemodynamics = NormalizedMLP(2, 2, width, layers)
        self.transport = NormalizedMLP(3, 1, width, layers)
        self.oncology = NormalizedMLP(3, 1, width, layers)
        self.response = nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, 1))
        self.habr_raw = nn.Parameter(torch.tensor(0.0))

    def habr_gain(self) -> Tensor:
        return 1.1 + 1.4 * torch.sigmoid(self.habr_raw)

    def forward(
        self,
        position: Tensor,
        time: Tensor,
        embolization: Tensor,
        tace_time: float = 0.5,
    ) -> TwinOutput:
        hemodynamics = self.hemodynamics(torch.cat([position, time], dim=-1))
        pre_velocity = hemodynamics[:, :1]
        pressure = hemodynamics[:, 1:2]
        post_velocity = post_embolization_velocity(
            pre_velocity,
            embolization,
            time,
            tace_time,
            self.habr_gain(),
        )
        concentration = torch.nn.functional.softplus(
            self.transport(torch.cat([position, time, post_velocity], dim=-1))
        )
        cell_density = torch.sigmoid(
            self.oncology(torch.cat([position, time, concentration], dim=-1))
        )
        summary = torch.stack(
            [
                concentration.mean(),
                cell_density.mean(),
                post_velocity.abs().mean(),
            ]
        ).reshape(1, 3)
        response_logit = self.response(summary)
        return TwinOutput(
            pre_velocity,
            pressure,
            post_velocity,
            concentration,
            cell_density,
            response_logit,
        )

    def parameters_for_stage(self, stage: str) -> list[nn.Parameter]:
        groups = {
            "hemodynamics": self.hemodynamics,
            "transport": self.transport,
            "oncology": self.oncology,
            "joint": self,
        }
        if stage not in groups:
            raise ValueError(f"unknown curriculum stage: {stage}")
        return list(groups[stage].parameters())


class PatientParameterEstimator(nn.Module):
    def __init__(self, initial: PhysiologicalParameters) -> None:
        super().__init__()
        self.diffusion_raw = nn.Parameter(torch.tensor(initial.diffusion_um2_s))
        self.elimination_raw = nn.Parameter(torch.tensor(initial.elimination_per_hour))
        self.uptake_raw = nn.Parameter(torch.tensor(initial.uptake_um_s))
        self.proliferation_raw = nn.Parameter(torch.tensor(initial.proliferation_per_day))
        self.kill_raw = nn.Parameter(torch.tensor(initial.maximum_kill_per_hour))

    def constrained(self) -> dict[str, Tensor]:
        return {
            "diffusion_um2_s": self.diffusion_raw.clamp(100.0, 800.0),
            "elimination_per_hour": self.elimination_raw.clamp(0.0, 1.0),
            "uptake_um_s": self.uptake_raw.clamp(0.0, 0.01),
            "proliferation_per_day": self.proliferation_raw.clamp(0.01, 0.10),
            "maximum_kill_per_hour": self.kill_raw.clamp(0.0, 2.0),
        }
