from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.constants import TARGET_COLUMNS
from src.data.preprocessing import DataSplit, inverse_min_max_scale
from src.metrics import evaluate_targets, r2_score
from src.models.benchmarks import BenchmarkConfig, train_bp_model, train_rf_model, train_svr_model
from src.models.dnn import CFSTConnectionDNN, DNNConfig, count_trainable_parameters, load_vector_into_model
from src.models.ipso import IPSOConfig, IPSOOptimizer
from src.training.trainer import TrainingConfig, denormalize_predictions, predict, train_dnn


def run_ipso_dnn(
    split: DataSplit,
    config: dict,
    output_dir: Path,
) -> dict:
    dnn_cfg = DNNConfig(
        hidden_layers=tuple(config["dnn"]["hidden_layers"]),
        dropout_rate=config["dnn"]["dropout_rate"],
        learning_rate=config["dnn"]["learning_rate"],
    )
    train_cfg = TrainingConfig(
        batch_size=config["dnn"]["batch_size"],
        max_epochs=config["dnn"]["max_epochs"],
        early_stopping_patience=config["dnn"]["early_stopping_patience"],
        fitness_epochs=config["ipso"]["fitness_epochs"],
    )
    ipso_cfg = IPSOConfig(
        population_size=config["ipso"]["population_size"],
        max_iterations=config["ipso"]["max_iterations"],
        c1=config["ipso"]["c1"],
        c2=config["ipso"]["c2"],
        w_max=config["ipso"]["w_max"],
        w_min=config["ipso"]["w_min"],
        alpha=config["ipso"]["alpha"],
        stagnation_threshold=config["ipso"]["stagnation_threshold"],
        levy_lambda=config["ipso"]["levy_lambda"],
        levy_beta0=config["ipso"]["levy_beta0"],
    )

    template = CFSTConnectionDNN(dnn_cfg)
    weight_dim = count_trainable_parameters(template)
    x_train_val = np.vstack([split.x_train, split.x_val])
    y_train_val = np.vstack([split.y_train, split.y_val])
    lr_bounds = tuple(config["ipso"]["learning_rate_bounds"])
    dropout_bounds = tuple(config["ipso"]["dropout_bounds"])

    def fitness_fn(candidate: np.ndarray) -> float:
        weights = candidate[:weight_dim]
        learning_rate = float(np.clip(candidate[-2], lr_bounds[0], lr_bounds[1]))
        dropout_rate = float(np.clip(candidate[-1], dropout_bounds[0], dropout_bounds[1]))
        model = CFSTConnectionDNN(
            DNNConfig(
                hidden_layers=dnn_cfg.hidden_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )
        )
        load_vector_into_model(model, weights)
        _, history = train_dnn(
            model,
            split.x_train,
            split.y_train,
            split.x_val,
            split.y_val,
            TrainingConfig(
                batch_size=train_cfg.batch_size,
                max_epochs=train_cfg.fitness_epochs,
                early_stopping_patience=train_cfg.early_stopping_patience,
                fitness_epochs=train_cfg.fitness_epochs,
            ),
            learning_rate=learning_rate,
        )
        return min(history) if history else float("inf")

    bounds_low = np.concatenate(
        [
            np.full(weight_dim, -0.5),
            np.array([lr_bounds[0], dropout_bounds[0]]),
        ]
    )
    bounds_high = np.concatenate(
        [
            np.full(weight_dim, 0.5),
            np.array([lr_bounds[1], dropout_bounds[1]]),
        ]
    )
    optimizer = IPSOOptimizer(bounds_low, bounds_high, fitness_fn, ipso_cfg, random_seed=config["project"]["random_seed"])
    ipso_result = optimizer.optimize()

    best_weights = ipso_result.best_position[:weight_dim]
    best_lr = float(np.clip(ipso_result.best_position[-2], lr_bounds[0], lr_bounds[1]))
    best_dropout = float(np.clip(ipso_result.best_position[-1], dropout_bounds[0], dropout_bounds[1]))
    final_model = CFSTConnectionDNN(
        DNNConfig(
            hidden_layers=dnn_cfg.hidden_layers,
            dropout_rate=best_dropout,
            learning_rate=best_lr,
        )
    )
    load_vector_into_model(final_model, best_weights)
    final_model, _ = train_dnn(
        final_model,
        x_train_val,
        y_train_val,
        split.x_test,
        split.y_test,
        train_cfg,
        learning_rate=best_lr,
    )

    y_pred_norm = predict(final_model, split.x_test)
    y_true = denormalize_predictions(split.y_test, split.y_min, split.y_max)
    y_pred = denormalize_predictions(y_pred_norm, split.y_min, split.y_max)
    metrics = evaluate_targets(y_true, y_pred, TARGET_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch_path = output_dir / "ipso_dnn.pt"
    import torch

    torch.save(final_model.state_dict(), torch_path)
    with (output_dir / "ipso_history.json").open("w", encoding="utf-8") as handle:
        json.dump({"fitness_history": ipso_result.history, "metrics": metrics}, handle, indent=2)

    return {
        "model_path": str(torch_path),
        "metrics": metrics,
        "history": ipso_result.history,
        "hyperparameters": {"learning_rate": best_lr, "dropout_rate": best_dropout},
    }


def run_benchmarks(split: DataSplit, config: dict) -> pd.DataFrame:
    benchmark_cfg = BenchmarkConfig(
        bp_hidden_size=config["benchmarks"]["bp_hidden_size"],
        svr_c_grid=tuple(config["benchmarks"]["svr_c_grid"]),
        svr_gamma_grid=tuple(config["benchmarks"]["svr_gamma_grid"]),
        rf_n_estimators=config["benchmarks"]["rf_n_estimators"],
        rf_max_depth=config["benchmarks"]["rf_max_depth"],
        rf_min_samples_leaf=config["benchmarks"]["rf_min_samples_leaf"],
    )
    x_train_val = np.vstack([split.x_train, split.x_val])
    y_train_val = np.vstack([split.y_train, split.y_val])
    y_true = denormalize_predictions(split.y_test, split.y_min, split.y_max)

    models = {
        "BP": train_bp_model(x_train_val, y_train_val, benchmark_cfg),
        "SVR": train_svr_model(x_train_val, y_train_val, benchmark_cfg),
        "RF": train_rf_model(x_train_val, y_train_val, benchmark_cfg),
    }

    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        y_pred_norm = model.predict(split.x_test)
        y_pred = denormalize_predictions(y_pred_norm, split.y_min, split.y_max)
        metrics = evaluate_targets(y_true, y_pred, TARGET_COLUMNS)
        for target in TARGET_COLUMNS:
            rows.append(
                {
                    "model": name,
                    "target": target,
                    **metrics[target],
                }
            )
    return pd.DataFrame(rows)


def run_cross_validation(dataframe, config: dict) -> pd.DataFrame:
    from src.data.preprocessing import min_max_scale

    x = dataframe[[col for col in config["data"]["input_columns"]]].to_numpy(dtype=np.float64)
    y = dataframe[[col for col in config["data"]["target_columns"]]].to_numpy(dtype=np.float64)
    kfold = KFold(n_splits=5, shuffle=True, random_state=config["project"]["random_seed"])
    rows: list[dict[str, float | str]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(x), start=1):
        x_train_raw, x_test_raw = x[train_idx], x[test_idx]
        y_train_raw, y_test_raw = y[train_idx], y[test_idx]
        x_min = x_train_raw.min(axis=0)
        x_max = x_train_raw.max(axis=0)
        y_min = y_train_raw.min(axis=0)
        y_max = y_train_raw.max(axis=0)
        x_train = min_max_scale(x_train_raw, x_min, x_max)
        x_test = min_max_scale(x_test_raw, x_min, x_max)
        y_train = min_max_scale(y_train_raw, y_min, y_max)
        y_test = min_max_scale(y_test_raw, y_min, y_max)

        model = CFSTConnectionDNN(DNNConfig(hidden_layers=tuple(config["dnn"]["hidden_layers"])))
        model, _ = train_dnn(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            TrainingConfig(
                batch_size=config["dnn"]["batch_size"],
                max_epochs=120,
                early_stopping_patience=20,
            ),
            learning_rate=config["dnn"]["learning_rate"],
        )
        y_pred = denormalize_predictions(predict(model, x_test), y_min, y_max)
        y_true = denormalize_predictions(y_test, y_min, y_max)
        for target_idx, target in enumerate(TARGET_COLUMNS):
            rows.append(
                {
                    "fold": fold_idx,
                    "target": target,
                    "R2": r2_score(y_true[:, target_idx], y_pred[:, target_idx]),
                }
            )
    return pd.DataFrame(rows)
