from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DNNConfig:
    input_size: int = 11
    hidden_layers: tuple[int, ...] = (128, 64, 32, 16)
    output_size: int = 3
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3


class CFSTConnectionDNN(nn.Module):
    def __init__(self, config: DNNConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_features = config.input_size
        for index, hidden in enumerate(config.hidden_layers):
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if index < 2:
                layers.append(nn.Dropout(config.dropout_rate))
            in_features = hidden
        layers.append(nn.Linear(in_features, config.output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def vectorize_model(model: nn.Module) -> np.ndarray:
    chunks = [parameter.detach().cpu().numpy().reshape(-1) for parameter in model.parameters()]
    return np.concatenate(chunks)


def load_vector_into_model(model: nn.Module, vector: np.ndarray) -> None:
    offset = 0
    for parameter in model.parameters():
        size = parameter.numel()
        values = vector[offset : offset + size]
        parameter.data.copy_(torch.tensor(values.reshape(parameter.shape), dtype=parameter.dtype))
        offset += size


def create_random_model(config: DNNConfig, rng: np.random.Generator | None = None) -> CFSTConnectionDNN:
    model = CFSTConnectionDNN(config)
    if rng is not None:
        vector = rng.normal(0.0, 0.05, size=count_trainable_parameters(model))
        load_vector_into_model(model, vector)
    return model
