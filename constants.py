from __future__ import annotations

INPUT_BOUNDS = {
    "d_b": (16.0, 24.0),
    "G_b": (8.8, 10.9),
    "n_r": (2.0, 4.0),
    "g_b": (90.0, 140.0),
    "t_c": (5.0, 12.5),
    "B_c": (150.0, 400.0),
    "f_cu": (30.0, 60.0),
    "h_b": (200.0, 500.0),
    "t_fb": (8.0, 20.0),
    "t_p": (10.0, 25.0),
    "b_p": (150.0, 300.0),
}

TARGET_BOUNDS = {
    "M_u": (45.0, 680.0),
    "S_j_ini": (5200.0, 89000.0),
    "mu": (1.8, 6.5),
}

INPUT_COLUMNS = list(INPUT_BOUNDS)
TARGET_COLUMNS = list(TARGET_BOUNDS)

MENDELEY_DATASET_ID = "z9d3hw9szy"
MENDELEY_PAGE_URL = "https://data.mendeley.com/datasets/z9d3hw9szy/1"
MENDELEY_REFERER = MENDELEY_PAGE_URL
