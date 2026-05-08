"""Helpers for displaying global model importance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


IMPORTANCE_PATH = Path("rsf_perm_importance_final5_scaled_external.csv")
EXPECTED_RANKING = [
    "Age",
    "LVEF",
    "LAarea_rand",
    "SIGNIFICANTLEFTSIDEDVHD",
    "HFrEF",
]


def load_importance_table(path: Path = IMPORTANCE_PATH) -> pd.DataFrame:
    """Load and validate the saved global importance ranking."""
    table = pd.read_csv(path)

    required_columns = {"feature", "median_importance", "stability_pos_frac"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(
            f"Importance file is missing required columns: {sorted(missing_columns)}"
        )

    table = table.copy()
    table["feature"] = pd.Categorical(
        table["feature"],
        categories=EXPECTED_RANKING,
        ordered=True,
    )
    table = table.sort_values("feature").reset_index(drop=True)

    ranking = table["feature"].astype(str).tolist()
    if ranking != EXPECTED_RANKING:
        raise ValueError(
            "Importance file ranking does not match the expected external validation order."
        )

    return table
