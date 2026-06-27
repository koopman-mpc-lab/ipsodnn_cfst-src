from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import INPUT_COLUMNS, TARGET_COLUMNS


@dataclass
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray


def min_max_scale(values: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    span = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
    return (values - vmin) / span


def inverse_min_max_scale(values: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    span = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
    return values * span + vmin


def stratified_response_bins(y: pd.DataFrame, n_bins: int = 5) -> np.ndarray:
    capacity = y["M_u"].to_numpy()
    bins = pd.qcut(capacity, q=min(n_bins, len(np.unique(capacity))), duplicates="drop")
    return bins.codes


def split_dataset(
    dataframe: pd.DataFrame,
    random_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> DataSplit:
    x = dataframe[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    y = dataframe[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    strata = stratified_response_bins(dataframe[TARGET_COLUMNS])

    x_train, x_temp, y_train, y_temp, strata_train, strata_temp = train_test_split(
        x,
        y,
        strata,
        test_size=(1.0 - train_ratio),
        random_state=random_seed,
        stratify=strata,
    )
    relative_val = val_ratio / (val_ratio + (1.0 - train_ratio - val_ratio))
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=(1.0 - relative_val),
        random_state=random_seed,
        stratify=strata_temp,
    )

    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)
    y_min = y_train.min(axis=0)
    y_max = y_train.max(axis=0)

    return DataSplit(
        x_train=min_max_scale(x_train, x_min, x_max),
        y_train=min_max_scale(y_train, y_min, y_max),
        x_val=min_max_scale(x_val, x_min, x_max),
        y_val=min_max_scale(y_val, y_min, y_max),
        x_test=min_max_scale(x_test, x_min, x_max),
        y_test=min_max_scale(y_test, y_min, y_max),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
