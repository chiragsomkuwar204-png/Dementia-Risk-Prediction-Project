from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from app_support import (
    BATCH_RECORD_ID_COLUMN,
    BATCH_REQUIRED_COLUMNS,
    MAX_BATCH_ROWS,
    RESULT_TABLES,
    InputValidationError,
    positive_class_probability,
    positive_class_shap_values,
    read_result_table,
    validate_batch_frame,
    validate_model_bundle,
    validate_prediction_input,
)


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Dementia Risk Predictor",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(13, 148, 136, 0.09), transparent 34rem),
            linear-gradient(180deg, #f8fbfc 0%, #f2f7f8 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(248, 251, 252, 0.82);
    }
    .main .block-container {
        max-width: 1320px;
        padding: 2rem 2.5rem 4rem;
    }
    .brand-hero {
        position: relative;
        overflow: hidden;
        margin: 0.2rem 0 1.5rem;
        padding: clamp(1.7rem, 4vw, 3.2rem);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 26px;
        color: #f7fbff;
        background:
            radial-gradient(circle at 84% 18%, rgba(94, 234, 212, 0.22), transparent 18rem),
            linear-gradient(118deg, #102b40 0%, #12485a 56%, #087e83 100%);
        box-shadow: 0 22px 55px rgba(16, 43, 64, 0.16);
    }
    .brand-hero-inner {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(15rem, 0.62fr);
        align-items: center;
        gap: 2rem;
    }
    .brand-hero::after {
        position: absolute;
        right: -5rem;
        bottom: -14rem;
        width: 31rem;
        height: 31rem;
        border: 1px solid rgba(204, 251, 241, 0.15);
        border-radius: 50%;
        box-shadow: 0 0 0 2.2rem rgba(204, 251, 241, 0.035),
                    0 0 0 4.5rem rgba(204, 251, 241, 0.03),
                    0 0 0 7rem rgba(204, 251, 241, 0.025);
        content: "";
        pointer-events: none;
    }
    .brand-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 1.1rem;
        color: #b7eee5;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .brand-kicker::before {
        width: 0.48rem;
        height: 0.48rem;
        border-radius: 50%;
        background: #6ee7c8;
        box-shadow: 0 0 0 0.25rem rgba(110, 231, 200, 0.14);
        content: "";
    }
    .brand-hero h1 {
        margin: 0;
        color: #fff;
        font-size: clamp(2.2rem, 4.7vw, 3.55rem);
        font-weight: 650;
        letter-spacing: -0.055em;
        line-height: 1.04;
    }
    .brand-hero h1 span {
        color: #a9e8dc;
    }
    .brand-hero p {
        max-width: 43rem;
        margin: 1.15rem 0 0;
        color: #d3e6eb;
        font-size: 1.02rem;
        line-height: 1.7;
    }
    .hero-art {
        justify-self: end;
        width: min(100%, 22rem);
    }
    .study-card {
        padding: 1.25rem 1.35rem;
        border: 1px solid rgba(226, 255, 249, 0.23);
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(235, 255, 250, 0.14), rgba(235, 255, 250, 0.055));
        box-shadow: 0 18px 40px rgba(4, 20, 32, 0.16), inset 0 1px rgba(255,255,255,.12);
        backdrop-filter: blur(8px);
    }
    .study-card-top {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #c0eee7;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.14em;
    }
    .study-card-top::before {
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 50%;
        background: #74e6cb;
        box-shadow: 0 0 0 0.22rem rgba(116, 230, 203, 0.13);
        content: "";
    }
    .study-card-number {
        margin-top: 1.1rem;
        color: #fff;
        font-size: 3.35rem;
        font-weight: 650;
        letter-spacing: -0.07em;
        line-height: 1;
    }
    .study-card-label {
        margin-top: 0.35rem;
        color: #d3e6eb;
        font-size: 0.83rem;
    }
    .study-card-rule {
        height: 1px;
        margin: 1.1rem 0 0.9rem;
        background: rgba(226, 255, 249, 0.18);
    }
    .study-card-bottom {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 1rem;
    }
    .study-card-stat strong {
        display: block;
        color: #e8fffa;
        font-size: 1.35rem;
        letter-spacing: -0.04em;
    }
    .study-card-stat small {
        color: #c3dedf;
        font-size: 0.7rem;
    }
    .study-bars {
        display: flex;
        align-items: end;
        gap: 0.24rem;
        height: 2.8rem;
        padding-bottom: 0.15rem;
    }
    .study-bars i {
        width: 0.32rem;
        border-radius: 999px;
        background: linear-gradient(180deg, #9ef0dc, #56baa9);
    }
    .brand-footer {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1.7rem;
    }
    .brand-pill {
        display: inline-block;
        padding: 0.42rem 0.72rem;
        border: 1px solid rgba(219, 251, 245, 0.22);
        border-radius: 999px;
        color: #e2f6f2;
        background: rgba(255, 255, 255, 0.08);
        font-size: 0.73rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    [data-testid="stSegmentedControl"] {
        margin: 0.3rem 0 1.8rem;
        padding: 0.35rem;
        border: 1px solid #dce8eb;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 5px 18px rgba(24, 58, 72, 0.055);
    }
    [data-testid="stMetric"] {
        min-height: 7rem;
        padding: 1rem 1.1rem;
        border: 1px solid #dce8eb;
        border-radius: 16px;
        background: linear-gradient(145deg, #fff 0%, #f8fbfc 100%);
        box-shadow: 0 6px 18px rgba(24, 58, 72, 0.05);
    }
    [data-testid="stMetricLabel"] {
        color: #52707a;
        font-size: 0.83rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #123849;
        letter-spacing: -0.035em;
    }
    [data-testid="stForm"] {
        border-color: #dce8eb;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.76);
        box-shadow: 0 8px 26px rgba(24, 58, 72, 0.045);
    }
    [data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2ecef;
        border-radius: 15px;
        background: linear-gradient(145deg, #fff 0%, #f8fbfc 100%);
    }
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        min-height: 2.8rem;
        border-radius: 11px;
        font-weight: 600;
        transition: transform 140ms ease, box-shadow 140ms ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(12, 113, 120, 0.14);
    }
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    [data-testid="stFormSubmitButton"] > button:focus-visible,
    [data-testid="stSelectbox"] [role="combobox"]:focus-visible {
        outline: 3px solid #68c9ba;
        outline-offset: 3px;
    }
    [data-testid="stAlert"] {
        border-radius: 14px;
    }
    h2, h3 {
        letter-spacing: -0.025em;
    }
    @media (max-width: 900px) {
        .main .block-container {
            padding: 1.25rem 1.25rem 3rem;
        }
        .brand-hero {
            border-radius: 20px;
        }
        .brand-hero-inner {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        .hero-art {
            justify-self: start;
            width: min(100%, 24rem);
        }
    }
    @media (max-width: 560px) {
        .main .block-container {
            padding: 0.8rem 0.7rem 2rem;
        }
        .brand-hero {
            padding: 1.35rem;
            border-radius: 17px;
        }
        .brand-hero h1 {
            font-size: 2.25rem;
        }
        .brand-hero p {
            font-size: 0.92rem;
        }
        .brand-footer {
            gap: 0.4rem;
            margin-top: 1.1rem;
        }
        .brand-pill {
            padding: 0.35rem 0.55rem;
            font-size: 0.66rem;
        }
        .hero-art {
            display: none;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    <section class="brand-hero" aria-label="Dementia Risk Predictor research dashboard">
      <div class="brand-hero-inner">
        <div>
          <div class="brand-kicker">Neuroscience · Research explorer</div>
          <h1>Dementia Risk<br><span>Predictor</span></h1>
          <p>
            Explore how brain-scan measurements and clinical inputs shape a
            machine-learning score trained on the OASIS-2 research cohort.
          </p>
          <div class="brand-footer">
            <span class="brand-pill">Research prototype</span>
            <span class="brand-pill">Not clinically validated</span>
            <span class="brand-pill">No in-app data persistence</span>
          </div>
        </div>
        <div class="hero-art" aria-hidden="true">
          <div class="study-card">
            <div class="study-card-top">OASIS-2 RESEARCH COHORT</div>
            <div class="study-card-number">150</div>
            <div class="study-card-label">participants · 373 longitudinal visits</div>
            <div class="study-card-rule"></div>
            <div class="study-card-bottom">
              <div class="study-card-stat"><strong>7</strong><small>model features</small></div>
              <div class="study-card-stat"><strong>200</strong><small>trees in forest</small></div>
              <div class="study-bars">
                <i style="height:34%"></i><i style="height:52%"></i>
                <i style="height:43%"></i><i style="height:72%"></i>
                <i style="height:58%"></i><i style="height:88%"></i>
                <i style="height:68%"></i><i style="height:100%"></i>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

IS_DARK_THEME = st.context.theme.type == "dark"
if IS_DARK_THEME:
    theme_overrides = {
        "__PAGE_BACKGROUND__": (
            "radial-gradient(ellipse at 8% 0%, rgba(32, 132, 125, 0.14), transparent 34rem), "
            "linear-gradient(180deg, #10191f 0%, #0c151a 100%)"
        ),
        "__FIGURE_BACKGROUND__": "#10191f",
        "__HEADER_BACKGROUND__": "rgba(16, 25, 31, 0.88)",
        "__PANEL_BACKGROUND__": "linear-gradient(145deg, #17262d 0%, #132128 100%)",
        "__BORDER_COLOR__": "#2c414a",
        "__MUTED_TEXT__": "#adc0c6",
        "__STRONG_TEXT__": "#e5f0f2",
        "__GRID_COLOR__": "#334850",
        "__AXIS_BACKGROUND__": "#132128",
        "__AXIS_TEXT__": "#c4d2d7",
        "__POSITIVE_COLOR__": "#56c7ba",
        "__NEGATIVE_COLOR__": "#86a8c7",
    }
else:
    theme_overrides = {
        "__PAGE_BACKGROUND__": (
            "radial-gradient(ellipse at 8% 0%, rgba(13, 148, 136, 0.09), transparent 34rem), "
            "linear-gradient(180deg, #f8fbfc 0%, #f2f7f8 100%)"
        ),
        "__FIGURE_BACKGROUND__": "#f8fbfc",
        "__HEADER_BACKGROUND__": "rgba(248, 251, 252, 0.88)",
        "__PANEL_BACKGROUND__": "linear-gradient(145deg, #ffffff 0%, #f8fbfc 100%)",
        "__BORDER_COLOR__": "#dce8eb",
        "__MUTED_TEXT__": "#52707a",
        "__STRONG_TEXT__": "#123849",
        "__GRID_COLOR__": "#dce8eb",
        "__AXIS_BACKGROUND__": "#f8fbfc",
        "__AXIS_TEXT__": "#385663",
        "__POSITIVE_COLOR__": "#087e83",
        "__NEGATIVE_COLOR__": "#587a94",
    }

st.markdown(
    """
    <style>
    .stApp { background: __PAGE_BACKGROUND__ !important; }
    [data-testid="stHeader"] { background: __HEADER_BACKGROUND__ !important; }
    [data-testid="stSegmentedControl"] {
        border-color: __BORDER_COLOR__;
        background: __PANEL_BACKGROUND__;
    }
    [data-testid="stMetric"] {
        border-color: __BORDER_COLOR__;
        background: __PANEL_BACKGROUND__;
    }
    [data-testid="stMetricLabel"] { color: __MUTED_TEXT__; }
    [data-testid="stMetricValue"] { color: __STRONG_TEXT__; }
    [data-testid="stForm"] {
        border-color: __BORDER_COLOR__;
        background: __PANEL_BACKGROUND__;
    }
    [data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: __BORDER_COLOR__;
        background: __PANEL_BACKGROUND__;
    }
    .app-footer {
        border-top: 1px solid __BORDER_COLOR__;
        color: __MUTED_TEXT__;
    }
    .app-footer strong { color: __STRONG_TEXT__; }
    </style>
    """.replace(
        "__PAGE_BACKGROUND__", theme_overrides["__PAGE_BACKGROUND__"]
    )
    .replace("__HEADER_BACKGROUND__", theme_overrides["__HEADER_BACKGROUND__"])
    .replace("__PANEL_BACKGROUND__", theme_overrides["__PANEL_BACKGROUND__"])
    .replace("__BORDER_COLOR__", theme_overrides["__BORDER_COLOR__"])
    .replace("__MUTED_TEXT__", theme_overrides["__MUTED_TEXT__"])
    .replace("__STRONG_TEXT__", theme_overrides["__STRONG_TEXT__"]),
    unsafe_allow_html=True,
)

st.info(
    "Do not enter identifiable or protected health information. This model is "
    "not validated for screening, diagnosis, triage, treatment, or other care. "
    "Hosting providers may have separate logging and retention policies.",
    icon=":material/health_and_safety:",
)

loading_message = st.empty()
loading_message.info("Loading the prediction model...")


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError("Required model artifact model.pkl was not found.")
    bundle = joblib.load(model_path)
    validate_model_bundle(bundle)
    explainer = shap.TreeExplainer(bundle["model"])
    return bundle, explainer


try:
    bundle, explainer = load_model()
except Exception as error:
    loading_message.error(
        "The prediction model could not be loaded or does not match the app's "
        f"validated input contract ({type(error).__name__}). Deploy a trusted, "
        "compatible model.pkl artifact."
    )
    st.stop()
else:
    loading_message.empty()

model = bundle["model"]
features = tuple(bundle["features"])
imaging = tuple(bundle["imaging"])
clinical = tuple(bundle["clinical"])
ranges = bundle["ranges"]
asf_constant = float(bundle["asf_const"])

NAMES = {
    "eTIV": "eTIV",
    "nWBV": "nWBV",
    "ASF": "ASF",
    "Age": "Age",
    "SES": "SES",
    "MMSE": "MMSE",
    "M/F": "Sex",
}

EXAMPLES = {
    "Lower-score synthetic profile": {
        "etiv": 1474,
        "nwbv": 0.739,
        "age": 77,
        "sex": "Female",
        "ses": 2,
        "mmse": 29,
    },
    "Higher-score synthetic profile": {
        "etiv": 1463,
        "nwbv": 0.713,
        "age": 76,
        "sex": "Male",
        "ses": 2,
        "mmse": 27,
    },
    "Lower-MMSE synthetic profile": {
        "etiv": 1480,
        "nwbv": 0.68,
        "age": 80,
        "sex": "Male",
        "ses": 3,
        "mmse": 18,
    },
}
START = {
    "etiv": 1469,
    "nwbv": 0.726,
    "age": 76,
    "sex": "Female",
    "ses": 2,
    "mmse": 28,
}

ETIV_LO, ETIV_HI = int(np.ceil(ranges["eTIV"][0])), int(np.floor(ranges["eTIV"][1]))
NWBV_LO, NWBV_HI = float(ranges["nWBV"][0]), float(ranges["nWBV"][1])
AGE_LO, AGE_HI = int(np.ceil(ranges["Age"][0])), int(np.floor(ranges["Age"][1]))
MMSE_LO, MMSE_HI = int(np.ceil(ranges["MMSE"][0])), int(np.floor(ranges["MMSE"][1]))

WIDGET_BOUNDS = {
    "etiv": (ETIV_LO, ETIV_HI),
    "nwbv": (NWBV_LO, NWBV_HI),
    "age": (AGE_LO, AGE_HI),
    "mmse": (MMSE_LO, MMSE_HI),
    "ses": (1, 5),
}
for key, value in START.items():
    if key == "sex":
        continue
    low, high = WIDGET_BOUNDS[key]
    current = st.session_state.get(key, value)
    st.session_state[key] = min(max(current, low), high)
if st.session_state.get("sex") not in ("Female", "Male"):
    st.session_state["sex"] = START["sex"]


def use_example(name):
    example_values = START if name == "__initial_values__" else EXAMPLES[name]
    for key, value in example_values.items():
        if key in WIDGET_BOUNDS:
            low, high = WIDGET_BOUNDS[key]
            value = min(max(value, low), high)
        st.session_state[key] = value


def predict_and_explain(values, feature_names, _model, _explainer):
    row = pd.DataFrame([values], columns=feature_names)
    probability = float(positive_class_probability(_model, row)[0])
    contributions = positive_class_shap_values(
        _explainer.shap_values(row), len(feature_names)
    )
    return probability, tuple(float(value) for value in contributions)


def calculate_what_if(values, feature_names, varied_feature, nwbv_range, _model):
    row = pd.DataFrame([values], columns=feature_names)
    if varied_feature == "MMSE":
        grid = np.arange(MMSE_LO, MMSE_HI + 1)
    else:
        grid = np.linspace(nwbv_range[0], nwbv_range[1], 60)
    scenarios = pd.concat([row] * len(grid), ignore_index=True)
    scenarios[varied_feature] = grid
    scores = positive_class_probability(
        _model, scenarios[list(feature_names)]
    ) * 100
    return tuple(float(value) for value in grid), tuple(float(value) for value in scores)


@st.cache_data(ttl=3600, max_entries=8)
def load_results_csv(path, expected_columns, score_columns):
    return read_result_table(path, expected_columns, score_columns)


view = st.segmented_control(
    "Dashboard section",
    ["Predict", "Batch CSV", "How it works", "Results and limits"],
    default="Predict",
    key="dashboard_view",
    width="stretch",
)

if view == "Predict":
    st.subheader("Single-case explorer")
    st.caption(
        "Start with an illustrative example or enter values within the model's "
        "training ranges. Results update when you submit the form."
    )
    example_column, apply_column, reset_column = st.columns(
        [0.62, 0.21, 0.17], vertical_alignment="bottom"
    )
    with example_column:
        selected_example = st.selectbox(
            "Start from an illustrative profile",
            list(EXAMPLES),
            key="example_choice",
            help="Examples are synthetic values, not real patient records.",
        )
    with apply_column:
        st.button(
            "Load example",
            key="load_example",
            on_click=use_example,
            args=(selected_example,),
            width="stretch",
            icon=":material/auto_awesome:",
        )
    with reset_column:
        st.button(
            "Reset",
            key="reset_inputs",
            on_click=use_example,
            args=("__initial_values__",),
            width="stretch",
            help="Restore the initial synthetic values.",
        )
    st.caption(
        "Examples are synthetic. Editing any field does not change the model or "
        "its training data."
    )

    with st.form("prediction_inputs", border=True):
        scan_column, clinical_column = st.columns(2, gap="large")
        with scan_column:
            with st.container(border=True):
                st.markdown("**Brain scan measurements**")
                st.caption("Structural MRI features represented in the research model.")
                st.slider(
                    "eTIV · estimated intracranial volume",
                    ETIV_LO,
                    ETIV_HI,
                    key="etiv",
                    help="Estimated total intracranial volume from the scan.",
                )
                st.slider(
                    "nWBV · normalized whole brain volume",
                    NWBV_LO,
                    NWBV_HI,
                    step=0.001,
                    format="%.3f",
                    key="nwbv",
                    help="Normalized whole brain volume from the scan.",
                )
                asf = asf_constant / st.session_state["etiv"]
                st.caption(f"Derived scale factor (ASF): **{asf:.3f}**")

        with clinical_column:
            with st.container(border=True):
                st.markdown("**Participant details**")
                st.caption("Clinical and demographic features from the training cohort.")
                st.slider("Age · years", AGE_LO, AGE_HI, key="age")
                st.selectbox("Sex", ["Female", "Male"], key="sex")
                st.slider(
                    "Socioeconomic status · 1 = highest",
                    1,
                    5,
                    key="ses",
                )
                st.slider(
                    "MMSE · score out of 30",
                    MMSE_LO,
                    MMSE_HI,
                    key="mmse",
                    help="The slider is limited to the score range represented in the training data.",
                )

        st.form_submit_button(
            "Update estimate",
            key="update_estimate",
            type="primary",
            width="stretch",
            icon=":material/analytics:",
        )

    state = st.session_state
    asf = asf_constant / state["etiv"]
    input_values = {
        "eTIV": state["etiv"],
        "nWBV": state["nwbv"],
        "ASF": asf,
        "Age": state["age"],
        "SES": state["ses"],
        "MMSE": state["mmse"],
        "M/F": 1 if state["sex"] == "Male" else 0,
    }
    try:
        input_values = validate_prediction_input(input_values, ranges)
    except InputValidationError as error:
        st.error(f"Cannot produce an estimate: {error}")
        st.stop()
    row_values = tuple(input_values[feature] for feature in features)
    try:
        probability, contribution_values = predict_and_explain(
            row_values, features, model, explainer
        )
    except Exception as error:
        st.error(
            "The model could not produce a valid estimate or explanation for "
            f"these inputs ({type(error).__name__}). Check the deployed model artifact."
        )
        st.stop()
    contributions = pd.Series(contribution_values, index=features)

    with st.container(border=True):
        st.subheader("Model estimate")
        with st.container(horizontal=True, gap="small"):
            st.metric(
                "Dementia-group model score",
                f"{probability * 100:.0f}%",
                border=True,
                help="The score is not calibrated as an individual clinical probability.",
            )
            st.metric(
                "Research group at 0.5 threshold",
                "Dementia-labelled" if probability >= 0.5 else "Nondementia-labelled",
                border=True,
            )
        st.progress(probability)
        st.caption(
            "The score is not a diagnosis, screening result, or calibrated "
            "individual clinical probability. The 0.5 threshold is a model "
            "convention, not a validated clinical cutoff."
        )

    imaging_push = contributions[list(imaging)].sum()
    clinical_push = contributions[list(clinical)].sum()

    def direction(value):
        return "toward demented" if value > 0 else "toward nondemented"

    factors_column, exploration_column = st.columns([1.15, 0.85], gap="large")
    with factors_column:
        with st.container(border=True):
            st.subheader("What shaped this score")
            st.caption(
                "Positive values push the model output toward the dementia-labelled class."
            )
            ordered_features = contributions.abs().sort_values().index
            labels = []
            for feature in ordered_features:
                value = input_values[feature]
                if feature == "M/F":
                    shown = state["sex"]
                elif feature in ("nWBV", "ASF"):
                    shown = f"{value:.3f}"
                else:
                    shown = f"{value:.0f}"
                labels.append(f"{NAMES[feature]} = {shown}")

            figure, axis = plt.subplots(figsize=(8, 4.8))
            figure.patch.set_facecolor(theme_overrides["__FIGURE_BACKGROUND__"])
            axis.set_facecolor(theme_overrides["__AXIS_BACKGROUND__"])
            bar_colors = [
                theme_overrides[
                    "__POSITIVE_COLOR__" if contributions[feature] > 0 else "__NEGATIVE_COLOR__"
                ]
                for feature in ordered_features
            ]
            axis.barh(
                labels,
                [contributions[feature] for feature in ordered_features],
                color=bar_colors,
                height=0.62,
                edgecolor="none",
            )
            axis.axvline(
                0,
                color=theme_overrides["__MUTED_TEXT__"],
                linewidth=0.9,
                alpha=0.65,
            )
            axis.set_xlabel(
                "← Nondementia-labelled     Contribution     Dementia-labelled →",
                color=theme_overrides["__MUTED_TEXT__"],
                labelpad=12,
            )
            axis.tick_params(
                axis="both",
                colors=theme_overrides["__AXIS_TEXT__"],
                labelsize=9,
                length=0,
            )
            axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
            axis.xaxis.grid(
                True,
                color=theme_overrides["__GRID_COLOR__"],
                linewidth=0.7,
            )
            axis.set_axisbelow(True)
            figure.tight_layout(pad=1.5)
            st.pyplot(
                figure,
                width="stretch",
                alt="Horizontal SHAP chart showing how each input contributed to this estimate.",
            )
            plt.close(figure)

            with st.container(horizontal=True, gap="small"):
                st.metric(
                    "Imaging contribution",
                    f"{imaging_push:+.2f}",
                    help=f"Combined influence from eTIV, nWBV, and ASF: {direction(imaging_push)}.",
                    border=True,
                )
                st.metric(
                    "Clinical contribution",
                    f"{clinical_push:+.2f}",
                    help=f"Combined influence from age, sex, SES, and MMSE: {direction(clinical_push)}.",
                    border=True,
                )

    with exploration_column:
        with st.container(border=True):
            st.subheader("Explore one input")
            st.caption("See how the score shifts as one value changes.")
            varied_feature = st.segmented_control(
                "Input to vary",
                ["MMSE", "nWBV"],
                default="MMSE",
                key="what_if_feature",
                help="All other entered values remain unchanged.",
            )
            try:
                grid, scores = calculate_what_if(
                    row_values,
                    features,
                    varied_feature,
                    (NWBV_LO, NWBV_HI),
                    model,
                )
            except Exception as error:
                st.error(
                    "The model could not produce the input-exploration chart "
                    f"({type(error).__name__}). Check the deployed model artifact."
                )
                st.stop()
            what_if_data = pd.DataFrame(
                {varied_feature: grid, "Dementia-group model score (%)": scores}
            )
            st.line_chart(
                what_if_data,
                x=varied_feature,
                y="Dementia-group model score (%)",
                y_label="Model score (%)",
                alt=f"Model score as {varied_feature} changes while other inputs remain fixed.",
            )
            st.caption("This chart shows model behavior, not a causal effect.")

elif view == "Batch CSV":
    st.subheader("Score a CSV batch")
    st.caption(
        "Upload a structured dataset to generate one research score per row. "
        "Use synthetic or de-identified values only."
    )
    with st.container(border=True):
        st.warning(
            "Do not upload identifiable or protected health information. Uploaded "
            "rows are processed for this session and are not saved by this app; "
            "your hosting provider may have separate logging, retention, or backup "
            "policies. Obtain privacy and security approval before using real patient data."
        )
        st.caption(
            "CSV columns: eTIV, nWBV, Age, SES, and MMSE, plus either Sex "
            "(Female/F or Male/M) or M/F (0 = female, 1 = male). Optional "
            "record_id values must be unique and "
            "pseudonymous. Up to 500 rows are accepted; all values must be within "
            "the model's training ranges."
        )
        template = pd.DataFrame(
            [
                {
                    BATCH_RECORD_ID_COLUMN: "synthetic-001",
                    "eTIV": 1474,
                    "nWBV": 0.739,
                    "Age": 77,
                    "Sex": "Female",
                    "SES": 2,
                    "MMSE": 29,
                }
            ],
            columns=(BATCH_RECORD_ID_COLUMN, *BATCH_REQUIRED_COLUMNS),
        )
        st.download_button(
            "Download synthetic CSV example",
            template.to_csv(index=False).encode("utf-8"),
            file_name="dementia-risk-synthetic-example.csv",
            mime="text/csv",
        )
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            key="batch_csv_upload",
            help="The uploaded file is not stored by the application.",
        )
        if uploaded_file is not None:
            if uploaded_file.size > 5 * 1024 * 1024:
                st.error("CSV uploads must be 5 MB or smaller.")
            else:
                try:
                    batch_source = pd.read_csv(
                        uploaded_file,
                        dtype={BATCH_RECORD_ID_COLUMN: "string", "Sex": "string"},
                    )
                except (
                    UnicodeDecodeError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    ValueError,
                ) as error:
                    st.error(f"Could not read the uploaded CSV ({type(error).__name__}).")
                else:
                    try:
                        batch_rows = validate_batch_frame(
                            batch_source, ranges, asf_constant
                        )
                    except InputValidationError as error:
                        st.error(f"CSV validation failed: {error}")
                    else:
                        try:
                            batch_scores = positive_class_probability(
                                model, batch_rows[list(features)]
                            )
                        except (TypeError, ValueError, IndexError) as error:
                            st.error(
                                "The model could not score this CSV "
                                f"({type(error).__name__}). Verify the deployed model artifact."
                            )
                        else:
                            batch_results = pd.DataFrame(
                                {
                                    "row_number": np.arange(1, len(batch_rows) + 1),
                                    "model_score_percent": np.round(batch_scores * 100, 1),
                                }
                            )
                            if BATCH_RECORD_ID_COLUMN in batch_rows:
                                batch_results.insert(
                                    1,
                                    BATCH_RECORD_ID_COLUMN,
                                    batch_rows[BATCH_RECORD_ID_COLUMN],
                                )
                            st.dataframe(
                                batch_results,
                                hide_index=True,
                                width="stretch",
                                alt="Research model scores for uploaded CSV rows.",
                            )
                            st.download_button(
                                "Download batch scores",
                                batch_results.to_csv(index=False).encode("utf-8"),
                                file_name="dementia-risk-research-scores.csv",
                                mime="text/csv",
                            )
                            st.caption(
                                "Scores are uncalibrated research outputs. They are "
                                "not diagnoses, individual clinical probabilities, "
                                "or recommendations for care."
                            )

elif view == "How it works":
    st.subheader("How the estimate is produced")
    st.write(
        "A Random Forest trained on OASIS-2 combines three scan measurements "
        "(eTIV, nWBV, and ASF) with age, sex, socioeconomic status, and MMSE. "
        "SHAP summarizes how the fitted model used those inputs for an estimate."
    )
    process_columns = st.columns(3, gap="medium")
    process_cards = (
        (
            "01 · Inputs",
            "Seven study features",
            "Brain-volume measurements and selected participant details represented in the training cohort.",
        ),
        (
            "02 · Model",
            "Random Forest",
            "A fitted research model returns an uncalibrated score for the dementia-labelled class.",
        ),
        (
            "03 · Explanation",
            "SHAP values",
            "Feature contributions describe the model's behavior for this input, not cause and effect.",
        ),
    )
    for column, (eyebrow, title, description) in zip(process_columns, process_cards):
        with column:
            with st.container(border=True):
                st.caption(eyebrow)
                st.markdown(f"#### {title}")
                st.write(description)
    st.subheader("Relative influence by input group")
    st.write(
        "Across five analysis splits, imaging accounts for about 34% of mean "
        "absolute SHAP influence and clinical details about 66%."
    )
    modality_figure = BASE_DIR / "modality_share_final.png"
    if modality_figure.is_file():
        st.image(
            modality_figure,
            caption="Imaging versus clinical share of SHAP influence",
            alt="Bar chart showing approximately 34 percent imaging and 66 percent clinical influence.",
            width="stretch",
        )
    feature_figure = BASE_DIR / "shap_summary_final.png"
    if feature_figure.is_file():
        st.subheader("Feature-level influence")
        st.image(
            feature_figure,
            caption="SHAP summary; each dot represents a visit",
            alt="SHAP summary plot showing feature values and their influence on model scores.",
            width="stretch",
        )
    st.caption(
        "An explanation describes model behavior. It does not show what causes dementia."
    )

elif view == "Results and limits":
    st.subheader("Evaluation results")
    st.caption(
        "Published research summaries for the project model; these are not "
        "prospective validation or evidence of clinical utility."
    )
    st.write(
        "OASIS-2 contains 150 people and 373 visits. All visits from each person "
        "were kept together during cross-validation. The OASIS-1 results below "
        "are a separate external test."
    )
    result_tables = (
        (
            "Cross-validation on OASIS-2: modality analysis",
            "final_model_summary.csv",
        ),
        (
            "Cross-validation on OASIS-2: classifier comparison",
            "model_comparison_results.csv",
        ),
        ("External test: trained on OASIS-2, tested on OASIS-1", "oasis1_test_results.csv"),
    )
    for title, filename in result_tables:
        csv_path = BASE_DIR / filename
        table_schema = RESULT_TABLES[filename]
        try:
            table = load_results_csv(
                str(csv_path),
                table_schema["columns"],
                table_schema["scores"],
            )
        except Exception as error:
            st.markdown(f"**{title}**")
            st.warning(
                f"This results table is unavailable or invalid "
                f"({type(error).__name__}). Verify {filename} before deployment."
            )
        else:
            st.markdown(f"**{title}**")
            st.dataframe(table, hide_index=True, width="stretch", alt=title)
    st.subheader("Limitations")
    st.write(
        "The dataset is small and limited to adults aged 60 and older. MMSE is "
        "closely related to how the dementia group is defined, and the model "
        "score is not a calibrated probability. The final dashboard model is "
        "trained on all OASIS-2 records and has no prospective or clinical "
        "validation. It is an experimental research/educational aid only and "
        "must not inform care, screening, diagnosis, triage, or treatment. "
        "SHAP explains the model, not the causes of dementia."
    )

st.markdown(
    """
    <div class="app-footer" style="
      margin-top: 3rem;
      padding-top: 1.15rem;
      font-size: 0.78rem;
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;">
      <span><strong>Dementia Risk Predictor</strong> · Research prototype</span>
      <span>For exploration only · Not for clinical decision-making</span>
    </div>
    """,
    unsafe_allow_html=True,
)
