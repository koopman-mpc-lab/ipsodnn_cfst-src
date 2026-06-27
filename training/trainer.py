from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from src.data.preprocessing import inverse_min_max_scale
from src.models.dnn import CFSTConnectionDNN, DNNConfig, count_trainable_parameters, load_vector_into_model


@dataclass
class TrainingConfig:
    batch_size: int = 16
    max_epochs: int = 500
    early_stopping_patience: int = 30
    fitness_epochs: int = 50


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32)


def train_dnn(
    model: CFSTConnectionDNN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    learning_rate: float,
) -> tuple[CFSTConnectionDNN, list[float]]:
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience = 0
    history: list[float] = []

    x_train_t = _to_tensor(x_train)
    y_train_t = _to_tensor(y_train)
    x_val_t = _to_tensor(x_val).to(device)
    y_val_t = _to_tensor(y_val).to(device)
    n_samples = x_train.shape[0]

    for _ in range(config.max_epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        for start in range(0, n_samples, config.batch_size):
            batch_idx = permutation[start : start + config.batch_size]
            batch_x = x_train_t[batch_idx].to(device)
            batch_y = y_train_t[batch_idx].to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t).item())
        history.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model: CFSTConnectionDNN, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        predictions = model(_to_tensor(features))
    return predictions.cpu().numpy()


def cross_validation_fitness(
    vector: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    dnn_config: DNNConfig,
    training_config: TrainingConfig,
    n_splits: int = 5,
) -> float:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    losses: list[float] = []
    for train_idx, val_idx in kfold.split(x_data):
        model = CFSTConnectionDNN(dnn_config)
        load_vector_into_model(model, vector[: count_trainable_parameters(model)])
        model = CFSTConnectionDNN(
            DNNConfig(
                input_size=dnn_config.input_size,
                hidden_layers=dnn_config.hidden_layers,
                output_size=dnn_config.output_size,
                dropout_rate=float(vector[-1]),
                learning_rate=float(vector[-2]),
            )
        )
        load_vector_into_model(model, vector[: count_trainable_parameters(model)])
        _, history = train_dnn(
            model,
            x_data[train_idx],
            y_data[train_idx],
            x_data[val_idx],
            y_data[val_idx],
            TrainingConfig(
                batch_size=training_config.batch_size,
                max_epochs=training_config.fitness_epochs,
                early_stopping_patience=training_config.early_stopping_patience,
                fitness_epochs=training_config.fitness_epochs,
            ),
            learning_rate=float(vector[-2]),
        )
        losses.append(min(history) if history else float("inf"))
    return float(np.mean(losses))


def decode_candidate_vector(
    vector: np.ndarray,
    weight_dim: int,
    lr_bounds: tuple[float, float],
    dropout_bounds: tuple[float, float],
) -> tuple[np.ndarray, float, float]:
    weights = vector[:weight_dim]
    learning_rate = float(np.clip(vector[-2], lr_bounds[0], lr_bounds[1]))
    dropout_rate = float(np.clip(vector[-1], dropout_bounds[0], dropout_bounds[1]))
    return weights, learning_rate, dropout_rate


def denormalize_predictions(
    predictions: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    return inverse_min_max_scale(predictions, y_min, y_max)
