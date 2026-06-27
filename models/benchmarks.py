from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass
class BenchmarkConfig:
    bp_hidden_size: int = 20
    svr_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    svr_gamma_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_leaf: int = 3


def train_bp_model(x_train: np.ndarray, y_train: np.ndarray, config: BenchmarkConfig) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=(config.bp_hidden_size,),
                    activation="relu",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_svr_model(x_train: np.ndarray, y_train: np.ndarray, config: BenchmarkConfig) -> MultiOutputRegressor:
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf")),
        ]
    )
    grid = GridSearchCV(
        base,
        param_grid={
            "svr__C": list(config.svr_c_grid),
            "svr__gamma": list(config.svr_gamma_grid),
        },
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    wrapped = MultiOutputRegressor(grid)
    wrapped.fit(x_train, y_train)
    return wrapped


def train_rf_model(x_train: np.ndarray, y_train: np.ndarray, config: BenchmarkConfig) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_leaf=config.rf_min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model
