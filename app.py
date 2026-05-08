"""Streamlit inference app for the external RSF survival model."""

from __future__ import annotations

import numpy as np
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

THEMES = {
    "dark": {
        "bg": "#0e1117",
        "panel": "#151924",
        "text": "#f8fafc",
        "muted": "#98a2b3",
        "accent": "#0f766e",
        "blue": "#1565c0",
        "danger": "#b93815",
        "plot_bg": "#ffffff",
        "plot_text": "#111827",
        "input_bg": "#262730",
        "input_text": "#f8fafc",
        "border": "#2d3340",
        "expander": "#11161f",
    },
    "light": {
        "bg": "#f5f7fb",
        "panel": "#ffffff",
        "text": "#101828",
        "muted": "#475467",
        "accent": "#0f766e",
        "blue": "#124b7a",
        "danger": "#b42318",
        "plot_bg": "#ffffff",
        "plot_text": "#111827",
        "input_bg": "#ffffff",
        "input_text": "#101828",
        "border": "#d0d5dd",
        "expander": "#ffffff",
    },
}


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


def get_theme() -> dict[str, str]:
    return THEMES["light" if st.session_state.get("light_mode", False) else "dark"]


def apply_theme_css(theme: dict[str, str]) -> None:
    """Apply app-level light/dark styling."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {theme["bg"]};
            color: {theme["text"]};
        }}
        .stApp [data-testid="stMarkdownContainer"],
        .stApp label,
        .stApp p,
        .stApp span,
        .stApp div {{
            color: inherit;
        }}
        h1, h2, h3 {{
            color: {theme["text"]} !important;
        }}
        [data-testid="stHeader"] {{
            background: {theme["bg"]};
        }}
        [data-testid="stNumberInputContainer"] input,
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        .stSlider [data-baseweb="slider"] {{
            color: {theme["input_text"]} !important;
        }}
        [data-testid="stNumberInputContainer"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"] > div,
        [data-testid="stExpander"] {{
            background: {theme["input_bg"]} !important;
            border-color: {theme["border"]} !important;
        }}
        [data-testid="stRadio"] label,
        [data-testid="stWidgetLabel"] {{
            color: {theme["text"]} !important;
        }}
        [data-testid="stExpander"] details {{
            background: {theme["expander"]} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_band(five_year_risk: float) -> tuple[str, str, str]:
    """Map 5-year event risk to the configured categorical band."""
    for label, lower, upper, text_color, background in RISK_BANDS:
        if lower <= five_year_risk < upper:
            return label, text_color, background
    return "Unclassified", "#475467", "#f2f4f7"

def render_metric_card(
    label: str,
    value: str,
    theme: dict[str, str],
    *,
    accent: str = "#124b7a",
    primary: bool = False,
) -> None:
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
            <div style="font-size:0.92rem;font-weight:600;color:{theme["muted"]};margin-bottom:0.35rem;">
                {label}
            </div>
            <div style="font-size:{value_size};font-weight:700;color:{accent};line-height:1.1;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_gradient(five_year_risk: float, theme: dict[str, str]) -> None:
    """Render a training-band gradient with a marker for the patient's 5-year risk."""
    label, _, _ = risk_band(five_year_risk)
    marker_position = max(0.0, min(100.0, five_year_risk / 0.30 * 100.0))
    st.markdown(
        f"""
        <div style="margin-top:0.5rem;">
            <div style="font-size:0.95rem;font-weight:700;color:{theme["text"]};margin-bottom:0.65rem;">
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
            <div style="font-size:0.82rem;color:{theme["muted"]};margin-top:0.45rem;">
                Marker position is based on the predicted 5-year event risk.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_survival_curve(prediction, theme: dict[str, str]) -> None:
    """Render the survival curve with Plotly, or matplotlib as fallback."""
    if HAS_PLOTLY:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=prediction.curve_years,
                y=prediction.curve_survival,
                mode="lines",
                line={"color": theme["blue"], "width": 3},
                name="Survival probability",
            )
        )
        figure.update_layout(
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            height=430,
            xaxis_title="Years",
            yaxis_title="Survival probability",
            plot_bgcolor=theme["plot_bg"],
            paper_bgcolor=theme["plot_bg"],
            font={"color": theme["plot_text"]},
        )
        figure.update_yaxes(
            range=[0, 1],
            gridcolor="#e4e7ec",
            tickfont={"color": theme["plot_text"]},
            title_font={"color": theme["plot_text"]},
        )
        figure.update_xaxes(
            gridcolor="#e4e7ec",
            tickfont={"color": theme["plot_text"]},
            title_font={"color": theme["plot_text"]},
        )
        st.plotly_chart(figure, use_container_width=True)
        return

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    figure.patch.set_facecolor(theme["plot_bg"])
    axis.set_facecolor(theme["plot_bg"])
    axis.plot(prediction.curve_years, prediction.curve_survival, color=theme["blue"], linewidth=2.5)
    axis.set_xlabel("Years")
    axis.set_ylabel("Survival probability")
    axis.set_ylim(0, 1)
    axis.tick_params(colors=theme["plot_text"])
    axis.xaxis.label.set_color(theme["plot_text"])
    axis.yaxis.label.set_color(theme["plot_text"])
    axis.grid(alpha=0.25)
    st.pyplot(figure, clear_figure=True)


def render_raw_score_distribution(raw_score: float, theme: dict[str, str]) -> None:
    """Render an approximate normal reference curve for the raw RSF score."""
    p25 = TRAINING_REFERENCE_RAW_SCORE["P25"]
    p50 = TRAINING_REFERENCE_RAW_SCORE["P50"]
    p75 = TRAINING_REFERENCE_RAW_SCORE["P75"]
    p99 = TRAINING_REFERENCE_RAW_SCORE["P99"]

    sigma = max((p75 - p25) / 1.349, 1e-6)
    mu = p50

    x_min = min(mu - 3.5 * sigma, raw_score - 0.6 * sigma)
    x_max = max(mu + 3.5 * sigma, raw_score + 0.6 * sigma, p99 + 0.25 * sigma)
    x = np.linspace(x_min, x_max, 500)
    y = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    peak = float(y.max())

    marker_y = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((raw_score - mu) / sigma) ** 2)
    marker_y = float(min(marker_y, peak))

    if HAS_PLOTLY:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="tozeroy",
                line={"color": "#2563eb", "width": 2.5},
                fillcolor="rgba(37, 99, 235, 0.18)",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_vline(x=raw_score, line_width=2, line_color=theme["danger"])
        figure.add_annotation(
            x=raw_score,
            y=marker_y,
            text="▼",
            showarrow=False,
            font={"size": 18, "color": theme["danger"]},
            yshift=14,
        )
        figure.add_annotation(
            x=raw_score,
            y=peak * 1.02,
            text=f"RSF score: {raw_score:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.6,
            arrowcolor=theme["danger"],
            ax=0,
            ay=-32,
            font={"size": 12, "color": theme["plot_text"]},
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(180,35,24,0.25)",
        )
        figure.update_layout(
            margin={"l": 10, "r": 10, "t": 18, "b": 10},
            height=210,
            plot_bgcolor=theme["plot_bg"],
            paper_bgcolor=theme["plot_bg"],
            font={"color": theme["plot_text"]},
            xaxis_title="Raw RSF score",
            yaxis_title="Relative density",
        )
        figure.update_xaxes(
            gridcolor="#e4e7ec",
            tickfont={"color": theme["plot_text"]},
            title_font={"color": theme["plot_text"]},
        )
        figure.update_yaxes(
            showticklabels=False,
            gridcolor="#f2f4f7",
            title_font={"color": theme["plot_text"]},
            range=[0, peak * 1.18],
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Approximate normal reference curve based on external training percentiles.")
        return

    figure, axis = plt.subplots(figsize=(6.0, 2.6))
    figure.patch.set_facecolor(theme["plot_bg"])
    axis.set_facecolor(theme["plot_bg"])
    axis.plot(x, y, color="#2563eb", linewidth=2.2)
    axis.fill_between(x, y, 0, color="#2563eb", alpha=0.18)
    axis.axvline(raw_score, color=theme["danger"], linewidth=2)
    axis.annotate(
        f"RSF score: {raw_score:.2f}",
        xy=(raw_score, marker_y),
        xytext=(raw_score, peak * 1.08),
        ha="center",
        color=theme["plot_text"],
        arrowprops={"arrowstyle": "->", "color": theme["danger"], "lw": 1.5},
    )
    axis.set_xlabel("Raw RSF score")
    axis.set_ylabel("Relative density")
    axis.set_yticks([])
    axis.set_ylim(0, peak * 1.18)
    axis.tick_params(colors=theme["plot_text"])
    axis.xaxis.label.set_color(theme["plot_text"])
    axis.yaxis.label.set_color(theme["plot_text"])
    axis.grid(axis="x", alpha=0.2)
    st.pyplot(figure, clear_figure=True)
    st.caption("Approximate normal reference curve based on external training percentiles.")


def render_technical_details(prediction, importance_table) -> None:
    """Expose technical details and training-distribution references."""
    with st.expander("Model Reference", expanded=False):
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

toggle_col, title_col = st.columns([0.18, 0.82])
with toggle_col:
    st.toggle("Light mode", key="light_mode")

theme = get_theme()
apply_theme_css(theme)

with title_col:
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
            theme,
            accent="#0f766e",
            primary=True,
        )
    with category_col:
        render_risk_gradient(five_year_risk, theme)

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("1-year event risk", format_percent(prediction.event_risks["1-year"]), theme, accent=theme["blue"])
    with metric_cols[1]:
        render_metric_card("3-year event risk", format_percent(prediction.event_risks["3-year"]), theme, accent=theme["blue"])
    with metric_cols[2]:
        render_metric_card("10-year event risk", format_percent(prediction.event_risks["10-year"]), theme, accent=theme["blue"])

    curve_col, technical_col = st.columns([1.1, 0.9], gap="large")
    with curve_col:
        st.subheader("Survival Curve")
        render_survival_curve(prediction, theme)
    with technical_col:
        render_metric_card("Raw RSF score", f"{prediction.raw_risk_score:.2f}", theme, accent=theme["danger"])
        render_raw_score_distribution(prediction.raw_risk_score, theme)
        st.caption(
            "Global importance is precomputed from the external validation setting and is not "
            "individualized to the current patient."
        )

    render_technical_details(prediction, importance_table)
