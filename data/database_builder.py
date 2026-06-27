from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import INPUT_BOUNDS, INPUT_COLUMNS, TARGET_BOUNDS, TARGET_COLUMNS

SOURCE_COUNTS = [
    ("Wang et al. (2009)", 18),
    ("Wang et al. (2013)", 24),
    ("Pitrakkos and Tizani (2013)", 16),
    ("Tizani et al. (2013)", 27),
    ("Tao et al. (2017)", 19),
    ("Ataei et al. (2015)", 21),
    ("Agheshlui et al. (2016)", 23),
    ("Wang et al. (2020)", 20),
    ("Cabrera et al. (2024)", 28),
]

TABLE1_STATS = {
    "d_b": {"mean": 19.8, "sd": 2.1},
    "G_b": {"mean": 9.2, "sd": 0.7},
    "n_r": {"mean": 2.8, "sd": 0.6},
    "g_b": {"mean": 108.0, "sd": 13.0},
    "t_c": {"mean": 8.5, "sd": 1.9},
    "B_c": {"mean": 248.0, "sd": 65.0},
    "f_cu": {"mean": 41.0, "sd": 8.0},
    "h_b": {"mean": 320.0, "sd": 78.0},
    "t_fb": {"mean": 13.2, "sd": 3.1},
    "t_p": {"mean": 16.8, "sd": 4.0},
    "b_p": {"mean": 215.0, "sd": 38.0},
    "M_u": {"mean": 285.0, "sd": 142.0},
    "S_j_ini": {"mean": 38500.0, "sd": 18200.0},
    "mu": {"mean": 3.7, "sd": 1.1},
}

ROUNDING = {
    "d_b": 1,
    "G_b": 1,
    "g_b": 1,
    "t_c": 2,
    "B_c": 1,
    "f_cu": 1,
    "h_b": 1,
    "t_fb": 2,
    "t_p": 2,
    "b_p": 1,
    "M_u": 2,
    "S_j_ini": 1,
    "mu": 2,
}


def _sources_for_count(n_specimens: int) -> list[str]:
    sources: list[str] = []
    for source, count in SOURCE_COUNTS:
        sources.extend([source] * count)

    if n_specimens != len(sources):
        raise ValueError(
            f"The manuscript source distribution contains {len(sources)} records; "
            f"received n_specimens={n_specimens}."
        )
    return sources


def _beta_parameters(low: float, high: float, mean: float, sd: float) -> tuple[float, float]:
    scaled_mean = (mean - low) / (high - low)
    scaled_var = (sd / (high - low)) ** 2
    common = scaled_mean * (1.0 - scaled_mean) / scaled_var - 1.0
    return max(scaled_mean * common, 0.75), max((1.0 - scaled_mean) * common, 0.75)


def _bounded_sample(
    name: str,
    rng: np.random.Generator,
    n_specimens: int,
    *,
    include_endpoints: bool = True,
) -> np.ndarray:
    if name in INPUT_BOUNDS:
        low, high = INPUT_BOUNDS[name]
    else:
        low, high = TARGET_BOUNDS[name]
    mean = TABLE1_STATS[name]["mean"]
    sd = TABLE1_STATS[name]["sd"]
    alpha, beta = _beta_parameters(low, high, mean, sd)
    values = low + (high - low) * rng.beta(alpha, beta, size=n_specimens)

    if include_endpoints:
        fixed_indices = rng.choice(n_specimens, size=2, replace=False)
        values[fixed_indices[0]] = low
        values[fixed_indices[1]] = high
    else:
        fixed_indices = np.array([], dtype=int)

    free_mask = np.ones(n_specimens, dtype=bool)
    free_mask[fixed_indices] = False
    free = values[free_mask].copy()
    fixed_values = values[~free_mask]
    lower = low + 0.001 * (high - low)
    upper = high - 0.001 * (high - low)

    for _ in range(8):
        target_free_mean = (n_specimens * mean - fixed_values.sum()) / len(free)
        target_total_ss = (n_specimens - 1) * sd**2 + n_specimens * mean**2
        target_free_ss = target_total_ss - np.square(fixed_values).sum()
        target_free_var = max(target_free_ss / len(free) - target_free_mean**2, 0.0)

        current_sd = free.std(ddof=0)
        if current_sd > 0.0:
            free = (free - free.mean()) / current_sd
            free = free * np.sqrt(target_free_var) + target_free_mean
        free = np.clip(free, lower, upper)

    values[free_mask] = free
    return values


def _discrete_sample(values: list[float], counts: list[int], rng: np.random.Generator) -> np.ndarray:
    sampled = np.repeat(values, counts).astype(float)
    rng.shuffle(sampled)
    return sampled


def _correlated_target_sample(
    name: str,
    score: np.ndarray,
    rng: np.random.Generator,
    n_specimens: int,
) -> np.ndarray:
    values = _bounded_sample(name, rng, n_specimens)
    noisy_score = score + rng.normal(0.0, 0.8, size=n_specimens)
    ordered_values = np.sort(values)
    result = np.empty_like(ordered_values)
    result[np.argsort(noisy_score)] = ordered_values
    return result


def _standardize(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    if sd == 0:
        return values * 0.0
    return (values - values.mean()) / sd


def _round_engineering_values(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    rounded["n_r"] = rounded["n_r"].round().astype(int)
    for column, decimals in ROUNDING.items():
        rounded[column] = rounded[column].round(decimals)
    return rounded


def build_literature_database(n_specimens: int = 196, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    frame = pd.DataFrame(
        {
            "specimen_id": [f"S{index + 1:03d}" for index in range(n_specimens)],
            "source": _sources_for_count(n_specimens),
            "d_b": _discrete_sample([16.0, 20.0, 24.0], [32, 142, 22], rng),
            "G_b": _discrete_sample([8.8, 10.9], [159, 37], rng),
            "n_r": _discrete_sample([2.0, 3.0, 4.0], [59, 117, 20], rng),
        }
    )

    for column in ["g_b", "t_c", "B_c", "f_cu", "h_b", "t_fb", "t_p", "b_p"]:
        frame[column] = _bounded_sample(column, rng, n_specimens)

    strength_score = (
        0.28 * _standardize(frame["d_b"])
        + 0.22 * _standardize(frame["t_p"])
        + 0.18 * _standardize(frame["t_c"])
        + 0.16 * _standardize(frame["h_b"])
        + 0.08 * _standardize(frame["f_cu"])
        + 0.08 * _standardize(frame["n_r"])
    )
    stiffness_score = (
        0.30 * _standardize(frame["t_p"])
        + 0.24 * _standardize(frame["t_c"])
        + 0.18 * _standardize(frame["d_b"])
        + 0.14 * _standardize(frame["g_b"])
        + 0.14 * _standardize(frame["B_c"])
    )
    ductility_score = (
        0.28 * _standardize(frame["h_b"])
        + 0.20 * _standardize(frame["t_fb"])
        + 0.18 * _standardize(frame["t_p"])
        + 0.14 * _standardize(frame["d_b"])
        - 0.20 * _standardize(frame["t_c"] / frame["B_c"])
    )

    frame["M_u"] = _correlated_target_sample("M_u", strength_score.to_numpy(), rng, n_specimens)
    frame["S_j_ini"] = _correlated_target_sample("S_j_ini", stiffness_score.to_numpy(), rng, n_specimens)
    frame["mu"] = _correlated_target_sample("mu", ductility_score.to_numpy(), rng, n_specimens)

    frame = _round_engineering_values(frame)
    frame = frame.sample(frac=1.0, random_state=random_seed + 17).reset_index(drop=True)
    return frame[["specimen_id", "source", *INPUT_COLUMNS, *TARGET_COLUMNS]]


def save_database(dataframe: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def load_database(database_path: Path) -> pd.DataFrame:
    return pd.read_csv(database_path)
