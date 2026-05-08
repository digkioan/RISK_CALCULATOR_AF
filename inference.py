"""Inference helpers for the external RSF survival model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FEATURE_ORDER = [
    "Age",
    "LAarea_rand",
    "SIGNIFICANTLEFTSIDEDVHD",
    "HFrEF",
    "LVEF",
]

MODEL_PATH = Path("rsf_final5_scaled_external.pkl")
SCALER_PATH = Path("scaler_rsf_final5_scaled_external.pkl")

HORIZON_DAYS = {
    "1-year": 365.25,
    "3-year": 3 * 365.25,
    "5-year": 5 * 365.25,
    "10-year": 10 * 365.25,
}


@dataclass(frozen=True)
class PredictionResult:
    """Container for model outputs used by the Streamlit app."""

    raw_risk_score: float
    event_risks: dict[str, float]
    curve_years: np.ndarray
    curve_survival: np.ndarray


def load_model_and_scaler(model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
    """Load the persisted RSF model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def build_feature_frame(
    *,
    age: int,
    laarea_rand: float,
    significant_left_sided_vhd: int,
    hfrEF: int,
    lvef: int,
) -> pd.DataFrame:
    """Create a one-row DataFrame in the exact model feature order."""
    frame = pd.DataFrame(
        [
            {
                "Age": int(age),
                "LAarea_rand": float(laarea_rand),
                "SIGNIFICANTLEFTSIDEDVHD": int(significant_left_sided_vhd),
                "HFrEF": int(hfrEF),
                "LVEF": int(lvef),
            }
        ]
    )
    return frame.loc[:, FEATURE_ORDER]


def _clamp_probability(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _survival_at_horizon(survival_function, horizon_days: float) -> float:
    """Evaluate a scikit-survival step function at a fixed horizon."""
    survival_value = float(survival_function(horizon_days))
    return _clamp_probability(survival_value)


def predict_patient(model, scaler, patient_frame: pd.DataFrame) -> PredictionResult:
    """Run scaled inference and return the risk score, curve, and horizon risks."""
    scaled_values = scaler.transform(patient_frame)
    raw_risk_score = float(model.predict(scaled_values)[0])

    survival_function = model.predict_survival_function(scaled_values)[0]
    curve_days = np.asarray(survival_function.x, dtype=float)
    curve_survival = np.clip(np.asarray(survival_function.y, dtype=float), 0.0, 1.0)
    curve_years = curve_days / 365.25

    event_risks = {
        label: _clamp_probability(1.0 - _survival_at_horizon(survival_function, days))
        for label, days in HORIZON_DAYS.items()
    }

    return PredictionResult(
        raw_risk_score=raw_risk_score,
        event_risks=event_risks,
        curve_years=curve_years,
        curve_survival=curve_survival,
    )
