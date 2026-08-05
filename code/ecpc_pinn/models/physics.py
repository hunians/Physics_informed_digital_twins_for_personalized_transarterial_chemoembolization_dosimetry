import math

import torch
from torch import Tensor

from ecpc_pinn.models.networks import derivatives
from ecpc_pinn.types import PhysiologicalParameters


def navier_stokes_residual(
    velocity: Tensor,
    pressure: Tensor,
    position: Tensor,
    time: Tensor,
    parameters: PhysiologicalParameters,
) -> tuple[Tensor, Tensor]:
    velocity_time = derivatives(velocity, time)
    velocity_space = derivatives(velocity, position)
    pressure_space = derivatives(pressure, position)
    velocity_space_two = derivatives(velocity_space, position)
    momentum = (
        parameters.density * velocity_time
        + parameters.density * velocity * velocity_space
        + pressure_space
        - parameters.viscosity * velocity_space_two
    )
    return momentum, velocity_space


def advection_diffusion_reaction_residual(
    concentration: Tensor,
    velocity: Tensor,
    position: Tensor,
    time: Tensor,
    source: Tensor,
    parameters: PhysiologicalParameters,
) -> Tensor:
    concentration_time = derivatives(concentration, time)
    concentration_space = derivatives(concentration, position)
    concentration_space_two = derivatives(concentration_space, position)
    diffusion = parameters.diffusion_um2_s * 1e-12
    removal = parameters.elimination_per_hour + parameters.uptake_um_s
    return (
        concentration_time
        + velocity * concentration_space
        - diffusion * concentration_space_two
        - source
        + removal * concentration
    )


def tumor_residual(
    cells: Tensor,
    concentration: Tensor,
    position: Tensor,
    time: Tensor,
    parameters: PhysiologicalParameters,
) -> Tensor:
    cells_time = derivatives(cells, time)
    cells_space_two = derivatives(cells, position, 2)
    kill = hill_kill(concentration, parameters)
    growth = parameters.proliferation_per_day * cells * (1.0 - cells / parameters.carrying_capacity)
    return cells_time - parameters.cell_motility * cells_space_two - growth + kill * cells


def hill_kill(concentration: Tensor, parameters: PhysiologicalParameters) -> Tensor:
    powered = concentration.clamp_min(0).pow(parameters.hill_coefficient)
    half = parameters.ec50_ng_ml**parameters.hill_coefficient
    return parameters.maximum_kill_per_hour * powered / (half + powered).clamp_min(1e-8)


def deb_source(
    position: Tensor,
    time: Tensor,
    loading: Tensor,
    tace_time: float,
    parameters: PhysiologicalParameters,
) -> Tensor:
    elapsed = time - tace_time
    active = (elapsed >= 0).to(time.dtype)
    return loading * torch.exp(-parameters.release_per_hour * elapsed.clamp_min(0)) * active


def permeability(extent: Tensor, time: Tensor, tace_time: float, sharpness: float) -> Tensor:
    return extent * (1.0 - torch.sigmoid(sharpness * (time - tace_time)))


def post_embolization_velocity(
    pre_velocity: Tensor,
    extent: Tensor,
    time: Tensor,
    tace_time: float,
    habr_gain: Tensor,
    sharpness: float = 10.0,
) -> Tensor:
    sigma = permeability(extent, time, tace_time, sharpness)
    gain = habr_gain.clamp(1.1, 2.5)
    compensatory = gain * pre_velocity
    return sigma * pre_velocity + (1.0 - sigma) * compensatory


def poiseuille_resistance(length_m: Tensor, radius_m: Tensor, viscosity: float = 3.5e-3) -> Tensor:
    return 8.0 * viscosity * length_m / (math.pi * radius_m.pow(4).clamp_min(1e-16))


def bifurcation_loss(
    parent_pressure: Tensor,
    daughter_pressure: Tensor,
    parent_area: Tensor,
    parent_velocity: Tensor,
    daughter_area: Tensor,
    daughter_velocity: Tensor,
) -> Tensor:
    pressure_error = (daughter_pressure - parent_pressure).pow(2).mean()
    parent_flow = parent_area * parent_velocity
    daughter_flow = (daughter_area * daughter_velocity).sum()
    return pressure_error + (parent_flow - daughter_flow).pow(2).mean()
