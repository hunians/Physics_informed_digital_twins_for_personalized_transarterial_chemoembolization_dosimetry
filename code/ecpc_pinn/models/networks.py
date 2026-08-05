from collections.abc import Sequence

import torch
from torch import Tensor, nn


class NormalizedMLP(nn.Module):
    def __init__(self, inputs: int, outputs: int, width: int = 256, layers: int = 4) -> None:
        super().__init__()
        blocks: list[nn.Module] = [nn.Linear(inputs, width), nn.Tanh()]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(width, width), nn.Tanh()])
        blocks.append(nn.Linear(width, outputs))
        self.network = nn.Sequential(*blocks)
        self.register_buffer("lower", torch.zeros(inputs))
        self.register_buffer("upper", torch.ones(inputs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, values: Tensor) -> Tensor:
        normalized = 2.0 * (values - self.lower) / (self.upper - self.lower).clamp_min(1e-8) - 1.0
        return self.network(normalized)


class ResidualMLP(nn.Module):
    def __init__(self, inputs: int, outputs: int, width: int = 256, layers: int = 4) -> None:
        super().__init__()
        self.entry = nn.Linear(inputs, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.output = nn.Linear(width, outputs)

    def forward(self, values: Tensor) -> Tensor:
        hidden = torch.tanh(self.entry(values))
        for block in self.blocks:
            hidden = hidden + torch.tanh(block(hidden))
        return self.output(hidden)


def derivatives(output: Tensor, inputs: Tensor, order: int = 1) -> Tensor:
    current = output
    for _ in range(order):
        gradient = torch.autograd.grad(
            current,
            inputs,
            grad_outputs=torch.ones_like(current),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        current = gradient
    return current


def concatenate(values: Sequence[Tensor]) -> Tensor:
    return torch.cat(tuple(values), dim=-1)
