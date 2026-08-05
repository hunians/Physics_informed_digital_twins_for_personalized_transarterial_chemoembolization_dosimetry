from dataclasses import dataclass

import torch
from torch import Tensor

from ecpc_pinn.models.physics import (
    advection_diffusion_reaction_residual,
    deb_source,
    navier_stokes_residual,
    tumor_residual,
)
from ecpc_pinn.models.system import TwinOutput
from ecpc_pinn.types import PatientBatch, PhysiologicalParameters


@dataclass(frozen=True)
class LossWeights:
    data: float = 0.1
    physics: float = 1.0
    transport: float = 0.8
    oncology: float = 0.6


@dataclass
class LossTerms:
    total: Tensor
    hemodynamics: Tensor
    transport: Tensor
    oncology: Tensor
    response: Tensor


def mean_square(value: Tensor) -> Tensor:
    return value.pow(2).mean()


def coupled_loss(
    output: TwinOutput,
    batch: PatientBatch,
    parameters: PhysiologicalParameters,
    weights: LossWeights = LossWeights(),
) -> LossTerms:
    momentum, continuity = navier_stokes_residual(
        output.pre_velocity,
        output.pressure,
        batch.coordinates,
        batch.time,
        parameters,
    )
    source = deb_source(
        batch.coordinates,
        batch.time,
        batch.concentration,
        0.5,
        parameters,
    )
    transport_residual = advection_diffusion_reaction_residual(
        output.concentration,
        output.post_velocity,
        batch.coordinates,
        batch.time,
        source,
        parameters,
    )
    oncology_residual = tumor_residual(
        output.cell_density,
        output.concentration,
        batch.coordinates,
        batch.time,
        parameters,
    )
    hemodynamics = mean_square(momentum) + mean_square(continuity)
    hemodynamics = hemodynamics + weights.data * (
        mean_square(output.pre_velocity - batch.velocity)
        + mean_square(output.pressure - batch.pressure)
    )
    transport = mean_square(transport_residual)
    transport = transport + weights.data * mean_square(output.concentration - batch.concentration)
    oncology = mean_square(oncology_residual)
    oncology = oncology + weights.data * mean_square(output.cell_density - batch.density)
    response = torch.nn.functional.binary_cross_entropy_with_logits(
        output.response_logit.flatten(),
        batch.response.flatten(),
    )
    total = (
        weights.physics * hemodynamics
        + weights.transport * transport
        + weights.oncology * oncology
        + response
    )
    return LossTerms(total, hemodynamics, transport, oncology, response)


def inverse_objective(
    data_error: Tensor, adr_residual: Tensor, tumor_pde_residual: Tensor, eta: float = 0.01
) -> Tensor:
    return (
        mean_square(data_error)
        + eta * mean_square(adr_residual)
        + eta * mean_square(tumor_pde_residual)
    )
