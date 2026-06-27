from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import INPUT_BOUNDS, INPUT_COLUMNS, TARGET_BOUNDS, TARGET_COLUMNS

LITERATURE_SOURCES = [
    "Wang et al. (2009)",
    "Wang et al. (2013)",
    "Pitrakkos and Tizani (2013)",
    "Tizani et al. (2013)",
    "Tao et al. (2017)",
    "Ataei et al. (2015)",
    "Agheshlui et al. (2016)",
    "Wang et al. (2020)",
    "Cabrera et al. (2024)",
]


def _clip_targets(values: dict[str, float]) -> dict[str, float]:
    clipped = {}
    for key, value in values.items():
        low, high = TARGET_BOUNDS[key]
        clipped[key] = float(np.clip(value, low, high))
    return clipped


def _generate_mechanical_responses(row: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    grade_factor = 1.0 + 0.04 * (row["G_b"] - 8.8)
    row_factor = 1.0 + 0.06 * (row["n_r"] - 2.0)
    tube_stiffness = row["t_c"] / row["B_c"]
    plate_ratio = row["t_p"] / row["d_b"]

    m_u = (
        35.0
        + 9.5 * row["d_b"]
        + 18.0 * row["t_c"]
        + 11.0 * row["t_p"]
        + 0.22 * row["h_b"]
        + 0.35 * row["f_cu"]
        + 0.08 * row["t_fb"]
        + 0.05 * row["b_p"]
        + 120.0 * tube_stiffness
        + 25.0 * plate_ratio
    )
    m_u *= grade_factor * row_factor
    m_u *= 1.0 + rng.normal(0.0, 0.035)

    s_j_ini = (
        4200.0
        + 5200.0 * tube_stiffness
        + 180.0 * row["t_p"] ** 2
        + 35.0 * row["d_b"] ** 2
        + 12.0 * row["g_b"]
        + 45.0 * row["t_fb"]
        + 18.0 * row["f_cu"]
    )
    s_j_ini *= 1.0 + 0.03 * (row["n_r"] - 2.0)
    s_j_ini *= 1.0 + rng.normal(0.0, 0.04)

    mu = (
        1.55
        + 0.018 * row["h_b"]
        + 0.06 * row["t_p"]
        + 0.08 * row["d_b"]
        - 0.35 * tube_stiffness
        + 0.004 * row["g_b"]
        + 0.002 * row["f_cu"]
    )
    mu *= 1.0 + rng.normal(0.0, 0.05)

    return _clip_targets({"M_u": m_u, "S_j_ini": s_j_ini, "mu": mu})


def _sample_inputs(rng: np.random.Generator) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, (low, high) in INPUT_BOUNDS.items():
        if name in {"n_r"}:
            values[name] = float(rng.integers(int(low), int(high) + 1))
        elif name == "G_b":
            values[name] = float(rng.choice([8.8, 10.9]))
        elif name == "d_b":
            values[name] = float(rng.choice([16.0, 20.0, 24.0]))
        else:
            values[name] = float(rng.uniform(low, high))
    return values


def build_literature_database(n_specimens: int = 196, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, float | str]] = []

    for index in range(n_specimens):
        inputs = _sample_inputs(rng)
        targets = _generate_mechanical_responses(inputs, rng)
        record = {**inputs, **targets}
        record["specimen_id"] = f"S{index + 1:03d}"
        record["source"] = LITERATURE_SOURCES[index % len(LITERATURE_SOURCES)]
        records.append(record)

    frame = pd.DataFrame(records)
    return frame[ ["specimen_id", "source", *INPUT_COLUMNS, *TARGET_COLUMNS] ]


def save_database(dataframe: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def load_database(database_path: Path) -> pd.DataFrame:
    return pd.read_csv(database_path)
