import torch

from ecpc_pinn.models.physics import permeability, post_embolization_velocity


def test_permeability_changes_at_treatment() -> None:
    time = torch.tensor([[0.0], [1.0]])
    extent = torch.ones_like(time)
    value = permeability(extent, time, 0.5, 10.0)
    assert value[0] > value[1]


def test_habr_gain_is_bounded() -> None:
    velocity = torch.ones(2, 1)
    time = torch.ones(2, 1)
    extent = torch.zeros(2, 1)
    output = post_embolization_velocity(velocity, extent, time, 0.5, torch.tensor(4.0))
    assert torch.all(output <= 2.5)
