"""Streamlit inference app for the external RSF survival model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

from importance import load_importance_table
from inference import FEATURE_ORDER, build_feature_frame, load_model_and_scaler, predict_patient


st.set_page_config(
    page_title="AF Survival Risk Calculator",
    page_icon=":hospital:",
    layout="wide",
)


TRAINING_REFERENCE_RAW_SCORE = {
    "P25": 48.79,
    "P50": 123.71,
    "P75": 134.00,
    "P90": 185.77,
    "P95": 205.03,
    "P99": 302.53,
}

TRAINING_REFERENCE_5YR_RISK = {
    "Median": 0.1645,
    "P75": 0.1760,
    "P90": 0.2373,
    "P95": 0.2507,
}

RISK_BANDS = [
    ("Low", 0.0, 0.165, "#1f7a3a", "#eaf6ee"),
    ("Moderate", 0.165, 0.176, "#9a6700", "#fff4d6"),
    ("High", 0.176, 0.237, "#b54708", "#fff0e6"),
    ("Very high", 0.237, float("inf"), "#b42318", "#fdecec"),
]


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    """Load the persisted model artifacts once per process."""
    return load_model_and_scaler()


@st.cache_data(show_spinner=False)
def get_importance_table():
    """Load the saved global importance table once per process."""
    return load_importance_table()


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def risk_band(five_year_risk: float) -> tuple[str, str, str]:
    """Map 5-year event risk to the configured categorical band."""
    for label, lower, upper, text_color, background in RISK_BANDS:
        if lower <= five_year_risk < upper:
            return label, text_color, background
    return "Unclassified", "#475467", "#f2f4f7"

def render_metric_card(label: str, value: str, *, accent: str = "#124b7a", primary: bool = False) -> None:
    """Render a metric directly on the page background."""
    min_height = "108px" if primary else "88px"
    value_size = "2.4rem" if primary else "2rem"
    st.markdown(
        f"""
        <div style="
            min-height:{min_height};
            padding:0.25rem 0.1rem 0.35rem 0.1rem;
            background:transparent;
            border:none;
            box-shadow:none;
        ">
            <div style="font-size:0.92rem;font-weight:600;color:#98a2b3;margin-bottom:0.35rem;">
                {label}
            </div>
            <div style="font-size:{value_size};font-weight:700;color:{accent};line-height:1.1;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_gradient(five_year_risk: float) -> None:
    """Render a training-band gradient with a marker for the patient's 5-year risk."""
    label, _, _ = risk_band(five_year_risk)
    marker_position = max(0.0, min(100.0, five_year_risk / 0.30 * 100.0))
    st.markdown(
        f"""
        <div style="margin-top:0.5rem;">
            <div style="font-size:0.95rem;font-weight:700;color:#e5eef7;margin-bottom:0.65rem;">
                5-year risk category: {label}
            </div>
            <div style="
                position:relative;
                width:100%;
                height:52px;
                border-radius:14px;
                border:1px solid rgba(148,163,184,0.45);
                background:linear-gradient(90deg,
                    #dbeafe 0%,
                    #f8fafc 18%,
                    #fde68a 40%,
                    #84cc16 63%,
                    #f97316 82%,
                    #dc2626 100%);
                overflow:hidden;
            ">
                <div style="position:absolute;left:8%;top:50%;transform:translateY(-50%);font-size:0.76rem;color:#111827;">low</div>
                <div style="position:absolute;left:31%;top:50%;transform:translateY(-50%);font-size:0.76rem;color:#111827;">medium</div>
                <div style="position:absolute;left:58%;top:50%;transform:translateY(-50%);font-size:0.76rem;color:#111827;">high</div>
                <div style="position:absolute;left:84%;top:50%;transform:translate(-50%,-50%);font-size:0.76rem;color:#111827;">very high</div>
                <div style="
                    position:absolute;
                    left:calc({marker_position}% - 8px);
                    top:50%;
                    transform:translateY(-50%);
                    width:16px;
                    height:16px;
                    border-radius:50%;
                    background:#fbbf24;
                    border:2px solid #111827;
                    box-shadow:0 0 0 3px rgba(255,255,255,0.28);
                "></div>
            </div>
            <div style="font-size:0.82rem;color:#98a2b3;margin-top:0.45rem;">
                Marker position is based on the predicted 5-year event risk.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_survival_curve(prediction) -> None:
    """Render the survival curve with Plotly, or matplotlib as fallback."""
    if HAS_PLOTLY:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=prediction.curve_years,
                y=prediction.curve_survival,
                mode="lines",
                line={"color": "#124b7a", "width": 3},
                name="Survival probability",
            )
        )
        figure.update_layout(
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            height=220,
            xaxis_title="Years",
            yaxis_title="Survival probability",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        figure.update_yaxes(range=[0, 1], gridcolor="#e4e7ec")
        figure.update_xaxes(gridcolor="#e4e7ec")
        st.plotly_chart(figure, use_container_width=True)
        return

    figure, axis = plt.subplots(figsize=(5.8, 2.7))
    axis.plot(prediction.curve_years, prediction.curve_survival, color="#124b7a", linewidth=2.5)
    axis.set_xlabel("Years")
    axis.set_ylabel("Survival probability")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    st.pyplot(figure, clear_figure=True)


def render_technical_details(prediction, importance_table) -> None:
    """Expose technical details and training-distribution references."""
    with st.expander("Technical Details", expanded=False):
        st.markdown(
            "Inference uses the external saved `RandomSurvivalForest` model with a persisted "
            "`StandardScaler`. The primary patient-facing output is the **5-year event risk**. "
            "The raw RSF score below is a technical model output and is **not a probability**."
        )

        raw_col, feature_col = st.columns([1, 1.2])
        with raw_col:
            st.caption("External training distribution references for raw RSF score")
            st.table(
                pd.DataFrame(
                    {
                        "Statistic": list(TRAINING_REFERENCE_RAW_SCORE.keys()),
                        "Value": [f"{value:.2f}" for value in TRAINING_REFERENCE_RAW_SCORE.values()],
                    }
                )
            )
        with feature_col:
            st.caption("Exact model feature order")
            st.code("\n".join(FEATURE_ORDER), language="text")
            st.caption("Global model importance order")
            st.markdown(
                "\n".join(
                    [
                        f"{index}. {feature}"
                        for index, feature in enumerate(
                            importance_table["feature"].astype(str).tolist(),
                            start=1,
                        )
                    ]
                )
            )
            st.caption("External training distribution references for 5-year risk")
            st.table(
                pd.DataFrame(
                    {
                        "Statistic": list(TRAINING_REFERENCE_5YR_RISK.keys()),
                        "Value": [format_percent(value) for value in TRAINING_REFERENCE_5YR_RISK.values()],
                    }
                )
            )


st.title("AF Survival Risk Calculator")
st.caption(
    "Clinical decision support for inference with the saved external RSF model. "
    "For research or specialist interpretation only; outputs should be considered in clinical context."
)

try:
    model, scaler = get_model_bundle()
    importance_table = get_importance_table()
except Exception as exc:
    st.error(
        "The model artifacts could not be loaded. Confirm that the `.pkl` and `.csv` files are "
        "present and that the runtime matches the pinned package versions in `requirements.txt`."
    )
    st.exception(exc)
    st.stop()

input_col, output_col = st.columns([0.95, 1.35], gap="large")

with input_col:
    st.subheader("Patient Inputs")

    age = st.number_input(
        "Age (years)",
        min_value=8,
        max_value=103,
        value=68,
        step=1,
        help="Integer input. Typical observed range is roughly 23 to 93 years.",
    )

    laarea_rand = st.number_input(
        "Left atrial area",
        min_value=12.0,
        max_value=52.0,
        value=18.4,
        step=0.1,
        help="Continuous input, likely in cm². Typical observed range is roughly 16.0 to 39.4.",
    )

    significant_left_sided_vhd_label = st.radio(
        "Significant left-sided valvular heart disease",
        options=["No", "Yes"],
        horizontal=True,
    )

    hfrEF_label = st.radio(
        "Heart failure with reduced ejection fraction (HFrEF)",
        options=["No", "Yes"],
        horizontal=True,
    )

    lvef = st.slider(
        "LVEF (%)",
        min_value=10,
        max_value=88,
        value=55,
        step=1,
        help="Typical observed range is roughly 20 to 87.",
    )

    patient_frame = build_feature_frame(
        age=age,
        laarea_rand=laarea_rand,
        significant_left_sided_vhd=1 if significant_left_sided_vhd_label == "Yes" else 0,
        hfrEF=1 if hfrEF_label == "Yes" else 0,
        lvef=lvef,
    )

    st.caption("All fields are required. Inputs are scaled internally before inference.")

with output_col:
    prediction = predict_patient(model, scaler, patient_frame)
    five_year_risk = prediction.event_risks["5-year"]

    st.subheader("Predicted Risk")
    primary_col, category_col = st.columns([1.15, 1.0], gap="large")
    with primary_col:
        render_metric_card(
            "5-year event risk",
            format_percent(five_year_risk),
            accent="#0f766e",
            primary=True,
        )
    with category_col:
        render_risk_gradient(five_year_risk)

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("1-year event risk", format_percent(prediction.event_risks["1-year"]), accent="#124b7a")
    with metric_cols[1]:
        render_metric_card("3-year event risk", format_percent(prediction.event_risks["3-year"]), accent="#124b7a")
    with metric_cols[2]:
        render_metric_card("10-year event risk", format_percent(prediction.event_risks["10-year"]), accent="#124b7a")

    curve_col, technical_col = st.columns([0.78, 1.22], gap="large")
    with curve_col:
        st.subheader("Survival Curve")
        render_survival_curve(prediction)
    with technical_col:
        st.subheader("Technical Details")
        render_metric_card("Raw RSF score (technical)", f"{prediction.raw_risk_score:.2f}", accent="#7c2d12")
        st.caption(
            "Global importance is precomputed from the external validation setting and is not "
            "individualized to the current patient."
        )

    render_technical_details(prediction, importance_table)
