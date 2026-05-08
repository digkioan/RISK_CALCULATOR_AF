# AF Survival Risk Calculator

Streamlit app for inference with an external `RandomSurvivalForest` model for atrial fibrillation survival risk prediction using five clinical features.

## Overview

This project provides a clinician-friendly interface for running inference with a saved survival model. It loads a persisted external `RandomSurvivalForest` model and `StandardScaler`, accepts five patient inputs, and returns:

- 5-year event risk as the primary result
- Fixed-horizon event risks at 1, 3, 5, and 10 years
- A predicted survival curve
- Raw RSF risk score as a technical output
- Static global model importance based on saved external permutation importance

The app is for inference only. It does not retrain the model and does not require the original training dataset at runtime.

## Model Inputs

The model input feature order is fixed and must remain exactly:

1. `Age`
2. `LAarea_rand`
3. `SIGNIFICANTLEFTSIDEDVHD`
4. `HFrEF`
5. `LVEF`

## Project Files

- `app.py`: Streamlit user interface
- `inference.py`: model loading, preprocessing, and prediction helpers
- `importance.py`: static global importance loading and validation
- `requirements.txt`: pinned runtime dependencies
- `rsf_final5_scaled_external.pkl`: saved external RSF model
- `scaler_rsf_final5_scaled_external.pkl`: saved `StandardScaler`
- `rsf_perm_importance_final5_scaled_external.csv`: saved global permutation importance

## Requirements

Install the pinned dependencies from `requirements.txt`.

Important compatibility note:

- The saved scaler was created with `scikit-learn==1.7.2`
- The runtime should use `scikit-learn==1.7.2` to avoid pickle compatibility issues

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the App

From the project directory:

```bash
streamlit run app.py
```

If `streamlit` is not on your `PATH`, run:

```bash
python3 -m streamlit run app.py
```

By default, Streamlit serves the app locally at:

```text
http://localhost:8501
```

## Inference Logic

The app performs the following steps:

1. Loads the saved RSF model and scaler once using Streamlit caching
2. Builds a one-row pandas `DataFrame` in the exact required feature order
3. Applies the saved scaler
4. Generates:
   - Raw RSF score with `model.predict(...)`
   - Survival function with `model.predict_survival_function(...)`
   - Fixed-horizon event risks as `1 - S(t)` at:
     - 1 year = `365.25` days
     - 3 years = `3 * 365.25` days
     - 5 years = `5 * 365.25` days
     - 10 years = `10 * 365.25` days

Important:

- The raw RSF score is not a probability
- The primary patient-facing risk output is the 5-year event risk

## Global Importance

The app does not implement SHAP or patient-specific importance.

It uses the saved external permutation-importance file and presents global model importance only. This section is explicitly described as precomputed and not individualized to the current patient.

Expected importance order:

1. `Age`
2. `LVEF`
3. `LAarea_rand`
4. `SIGNIFICANTLEFTSIDEDVHD`
5. `HFrEF`

## Training Reference Values

The app exposes the following external model training-distribution reference values in the technical details section.

Raw RSF score:

- `P25`: `48.79`
- `P50`: `123.71`
- `P75`: `134.00`
- `P90`: `185.77`
- `P95`: `205.03`
- `P99`: `302.53`

5-year risk:

- `Median`: `0.1645`
- `P75`: `0.1760`
- `P90`: `0.2373`
- `P95`: `0.2507`

## Clinical Disclaimer

This application is intended for research or specialist interpretation only. Predictions should always be interpreted in full clinical context and should not be used in isolation for clinical decision-making.
