"""
Disease Risk Predictor — Streamlit App
======================================
Run locally:  streamlit run app.py
Deploy:       push to GitHub, connect to Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")          # must be before pyplot import
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Risk Predictor",
    page_icon="🏥",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────
# Disease configuration
# Each entry defines paths to saved artefacts and the input form.
#
# IMPORTANT: the "name" in each field dict must match EXACTLY the
# column names in your cleaned CSV (i.e. the features the model was
# trained on).  Check against your <disease>_features.pkl if unsure.
# ─────────────────────────────────────────────────────────────────────
DISEASE_CONFIG = {
    "🩸 Diabetes": {
        "model":       "models/diabetes_best.pkl",
        "scaler":      "models/diabetes_scaler.pkl",
        "background":  "models/diabetes_background.pkl",
        "features":    "models/diabetes_features.pkl",
        "description": (
            "Estimates the likelihood of Type 2 Diabetes using metabolic "
            "markers from the Pima Indians Diabetes dataset."
        ),
        "positive_label": "Diabetic",
        "fields": [
            dict(name="Pregnancies",               label="Number of Pregnancies",
                 widget="int_slider",   min=0,    max=20,    default=1,
                 help="Total number of times pregnant"),
            dict(name="Glucose",                   label="Plasma Glucose (mg/dL)",
                 widget="int_slider",   min=0,    max=300,   default=120,
                 help="2-hour plasma glucose concentration (oral glucose tolerance test)"),
            dict(name="BloodPressure",             label="Diastolic Blood Pressure (mm Hg)",
                 widget="int_slider",   min=0,    max=150,   default=72,
                 help="Diastolic blood pressure on admission"),
            dict(name="SkinThickness",             label="Skin Fold Thickness (mm)",
                 widget="int_slider",   min=0,    max=100,   default=23,
                 help="Triceps skin fold thickness"),
            dict(name="Insulin",                   label="Serum Insulin (μU/mL)",
                 widget="int_slider",   min=0,    max=900,   default=79,
                 help="2-hour serum insulin level"),
            dict(name="BMI",                       label="BMI (kg/m²)",
                 widget="float_input",  min=0.0,  max=70.0,  default=32.0, step=0.1,
                 help="Body mass index: weight(kg) / height(m)²"),
            dict(name="DiabetesPedigreeFunction",  label="Diabetes Pedigree Function",
                 widget="float_input",  min=0.0,  max=3.0,   default=0.47, step=0.01,
                 help="Genetic likelihood based on family history (higher = stronger history)"),
            dict(name="Age",                       label="Age (years)",
                 widget="int_slider",   min=18,   max=100,   default=33,
                 help="Age in years"),
        ],
    },

    "❤️ Heart Disease": {
        "model":       "models/heart_disease_best.pkl",
        "scaler":      "models/heart_disease_scaler.pkl",
        "background":  "models/heart_disease_background.pkl",
        "features":    "models/heart_disease_features.pkl",
        "description": (
            "Estimates the likelihood of coronary heart disease using clinical "
            "measurements from the UCI Heart Disease dataset."
        ),
        "positive_label": "Heart Disease",
        "fields": [
            dict(name="age",      label="Age (years)",
                 widget="int_slider",   min=20,   max=80,    default=54,
                 help="Age in years"),
            dict(name="sex",      label="Sex",
                 widget="radio",        options={"Female": 0, "Male": 1}, default="Female",
                 help="Biological sex"),
            dict(name="cp",       label="Chest Pain Type",
                 widget="select",
                 options={"Typical Angina (0)": 0, "Atypical Angina (1)": 1,
                          "Non-Anginal Pain (2)": 2, "Asymptomatic (3)": 3},
                 default="Asymptomatic (3)",
                 help="Type of chest pain"),
            dict(name="trestbps", label="Resting Blood Pressure (mm Hg)",
                 widget="int_slider",   min=90,   max=200,   default=131,
                 help="Resting systolic blood pressure on admission"),
            dict(name="chol",     label="Serum Cholesterol (mg/dL)",
                 widget="int_slider",   min=100,  max=600,   default=246,
                 help="Serum cholesterol in mg/dL"),
            dict(name="fbs",      label="Fasting Blood Sugar > 120 mg/dL",
                 widget="radio",        options={"No": 0, "Yes": 1}, default="No",
                 help="Whether fasting blood sugar exceeds 120 mg/dL"),
            dict(name="restecg",  label="Resting ECG Result",
                 widget="select",
                 options={"Normal (0)": 0, "ST-T Abnormality (1)": 1,
                          "Left Ventricular Hypertrophy (2)": 2},
                 default="Normal (0)",
                 help="Resting electrocardiographic results"),
            dict(name="thalach",  label="Max Heart Rate Achieved",
                 widget="int_slider",   min=60,   max=220,   default=149,
                 help="Maximum heart rate achieved during stress test"),
            dict(name="exang",    label="Exercise-Induced Angina",
                 widget="radio",        options={"No": 0, "Yes": 1}, default="No",
                 help="Whether exercise induced chest pain"),
            dict(name="oldpeak",  label="ST Depression (Oldpeak)",
                 widget="float_input",  min=0.0,  max=7.0,   default=1.0, step=0.1,
                 help="ST depression induced by exercise relative to rest"),
            dict(name="slope",    label="Slope of Peak ST Segment",
                 widget="select",
                 options={"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2},
                 default="Flat (1)",
                 help="Slope of the peak exercise ST segment"),
            dict(name="ca",       label="Major Vessels Coloured (0–4)",
                 widget="int_slider",   min=0,    max=4,     default=0,
                 help="Number of major vessels coloured by fluoroscopy"),
            dict(name="thal",     label="Thalassemia",
                 widget="select",
                 options={"Normal (1)": 1, "Fixed Defect (2)": 2, "Reversible Defect (3)": 3},
                 default="Normal (1)",
                 help="Thalassemia blood disorder type"),
        ],
    },

    "🧠 Stroke": {
        "model":       "models/stroke_best.pkl",
        "scaler":      "models/stroke_scaler.pkl",
        "background":  "models/stroke_background.pkl",
        "features":    "models/stroke_features.pkl",
        "description": (
            "Estimates the likelihood of stroke using demographic and clinical "
            "risk factors."
        ),
        "positive_label": "Stroke",
        # ⚠️  These field *names* must match your stroke_cleaned.csv columns exactly.
        # If your cleaning used one-hot encoding (e.g. work_type_Private,
        # work_type_Self-employed …) replace the single 'work_type' field below
        # with one binary toggle per encoded column.
        "fields": [
            dict(name="gender",            label="Gender",
                 widget="radio",       options={"Female": 0, "Male": 1}, default="Female",
                 help="Biological sex"),
            dict(name="age",               label="Age (years)",
                 widget="int_slider",  min=0,    max=100,   default=43,
                 help="Age in years"),
            dict(name="hypertension",      label="Hypertension",
                 widget="radio",       options={"No": 0, "Yes": 1}, default="No",
                 help="Diagnosed with hypertension"),
            dict(name="heart_disease",     label="Heart Disease",
                 widget="radio",       options={"No": 0, "Yes": 1}, default="No",
                 help="History of heart disease"),
            dict(name="ever_married",      label="Ever Married",
                 widget="radio",       options={"No": 0, "Yes": 1}, default="No",
                 help="Has the patient ever been married"),
            dict(name="avg_glucose_level", label="Avg Glucose Level (mg/dL)",
                 widget="float_input", min=50.0, max=300.0, default=106.0, step=0.1,
                 help="Average blood glucose level"),
            dict(name="bmi",               label="BMI (kg/m²)",
                 widget="float_input", min=10.0, max=70.0,  default=28.9, step=0.1,
                 help="Body mass index"),
            dict(name="work_type",         label="Work Type",
                 widget="select",
                 options={"Children (0)": 0, "Govt Job (1)": 1,
                          "Never Worked (2)": 2, "Private (3)": 3, "Self-employed (4)": 4},
                 default="Private (3)",
                 help="Type of employment"),
            dict(name="Residence_type",    label="Residence Type",
                 widget="radio",       options={"Rural": 0, "Urban": 1}, default="Urban",
                 help="Urban or rural residence"),
            dict(name="smoking_status",    label="Smoking Status",
                 widget="select",
                 options={"Unknown (0)": 0, "Formerly Smoked (1)": 1,
                          "Never Smoked (2)": 2, "Smokes (3)": 3},
                 default="Never Smoked (2)",
                 help="Smoking history"),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────
# Helper: load artefacts (cached so files are read only once per session)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artefacts(disease_key: str):
    cfg = DISEASE_CONFIG[disease_key]
    model      = joblib.load(cfg["model"])
    scaler     = joblib.load(cfg["scaler"])
    background = joblib.load(cfg["background"])   # numpy array (scaled)
    features   = joblib.load(cfg["features"])     # list of column names
    return model, scaler, background, features


# ─────────────────────────────────────────────────────────────────────
# Helper: render one field and return its value
# ─────────────────────────────────────────────────────────────────────
def render_field(field: dict):
    w = field["widget"]
    label = field["label"]
    help_ = field.get("help", "")

    if w == "int_slider":
        return st.slider(label, min_value=field["min"], max_value=field["max"],
                         value=field["default"], step=1, help=help_)

    if w == "float_input":
        return st.number_input(label, min_value=float(field["min"]),
                               max_value=float(field["max"]),
                               value=float(field["default"]),
                               step=float(field["step"]), help=help_)

    if w == "radio":
        opts = field["options"]          # {"Label": value, …}
        chosen = st.radio(label, list(opts.keys()),
                          index=list(opts.keys()).index(field["default"]),
                          horizontal=True, help=help_)
        return opts[chosen]

    if w == "select":
        opts = field["options"]
        chosen = st.selectbox(label, list(opts.keys()),
                              index=list(opts.keys()).index(field["default"]),
                              help=help_)
        return opts[chosen]

    raise ValueError(f"Unknown widget type: {w}")


# ─────────────────────────────────────────────────────────────────────
# Helper: pick the right SHAP explainer and return an Explanation object
# (positive class, index 1)
# ─────────────────────────────────────────────────────────────────────
def get_shap_explanation(model, background_data, input_scaled: np.ndarray,
                          feature_names: list) -> shap.Explanation:
    X = input_scaled  # shape (1, n_features)

    # ── Tree-based ────────────────────────────────────────────────────
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # Binary classification → list [neg_class, pos_class]
        values   = (sv[1][0] if isinstance(sv, list) else sv[0]).astype(float)
        base_val = float(explainer.expected_value[1]
                         if hasattr(explainer.expected_value, "__len__")
                         else explainer.expected_value)

    # ── Linear (Logistic Regression) ──────────────────────────────────
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, background_data)
        sv = explainer.shap_values(X)
        values   = (sv[1][0] if isinstance(sv, list) else sv[0]).astype(float)
        base_val = float(explainer.expected_value[1]
                         if hasattr(explainer.expected_value, "__len__")
                         else explainer.expected_value)

    # ── Kernel (SVM / anything else) ─────────────────────────────────
    else:
        # kmeans summarises the background to ~50 representative points
        # so KernelExplainer stays fast enough for a web app
        bg_summary = shap.kmeans(background_data, min(50, len(background_data)))
        explainer  = shap.KernelExplainer(model.predict_proba, bg_summary)
        sv = explainer.shap_values(X, nsamples=200)
        values   = sv[1][0].astype(float)
        base_val = float(explainer.expected_value[1])

    return shap.Explanation(
        values=values,
        base_values=base_val,
        data=X[0],
        feature_names=feature_names,
    )


# ─────────────────────────────────────────────────────────────────────
# Helper: render the SHAP waterfall chart in Streamlit
# ─────────────────────────────────────────────────────────────────────
def render_shap_chart(explanation: shap.Explanation, disease_label: str):
    st.subheader("🔍 Why did the model give this score?")
    st.caption(
        "The chart below shows **which factors pushed your risk score up (red) "
        "or down (blue)**, and by how much. The starting point is the average "
        "risk across everyone in the training dataset."
    )

    fig, _ = plt.subplots(figsize=(8, max(4, len(explanation.values) * 0.45)))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"Feature contributions — {disease_label}", pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Human-readable summary table
    impact_df = pd.DataFrame({
        "Feature":       explanation.feature_names,
        "Your Value":    explanation.data,
        "SHAP Impact":   explanation.values,
    }).sort_values("SHAP Impact", key=abs, ascending=False).reset_index(drop=True)

    impact_df["Direction"] = impact_df["SHAP Impact"].apply(
        lambda v: "⬆ Increases risk" if v > 0 else "⬇ Decreases risk"
    )
    impact_df["SHAP Impact"] = impact_df["SHAP Impact"].map("{:+.4f}".format)
    impact_df["Your Value"]  = impact_df["Your Value"].map(
        lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    )

    st.dataframe(impact_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────
st.title("🏥 Disease Risk Predictor")
st.markdown(
    "> **Disclaimer:** This tool is for **educational purposes only**. "
    "It is not a medical device and must not replace professional medical advice. "
    "Always consult a qualified clinician for health decisions."
)

st.divider()

# ── Disease selector ─────────────────────────────────────────────────
disease = st.radio(
    "Select a condition to assess:",
    list(DISEASE_CONFIG.keys()),
    horizontal=True,
)

cfg = DISEASE_CONFIG[disease]
st.markdown(f"_{cfg['description']}_")
st.divider()

# ── Check artefacts exist ────────────────────────────────────────────
missing = [p for p in [cfg["model"], cfg["scaler"],
                        cfg["background"], cfg["features"]]
           if not Path(p).exists()]
if missing:
    st.error(
        f"Missing model files:\n\n"
        + "\n".join(f"- `{p}`" for p in missing)
        + "\n\nPlease run the training notebook for this disease and copy "
          "the `.pkl` files into `models/`."
    )
    st.stop()

model, scaler, background, feature_names = load_artefacts(disease)

# ── Input form ───────────────────────────────────────────────────────
st.subheader("📋 Enter your health data")

with st.form("input_form"):
    # Two-column layout for compact rendering
    col_left, col_right = st.columns(2)
    fields = cfg["fields"]
    values = {}

    for i, field in enumerate(fields):
        col = col_left if i % 2 == 0 else col_right
        with col:
            values[field["name"]] = render_field(field)

    submitted = st.form_submit_button("🔎 Assess my risk", use_container_width=True)

# ── Prediction + SHAP ────────────────────────────────────────────────
if submitted:
    # Build input DataFrame (column order must match training)
    input_df = pd.DataFrame([{fn: values[fn] for fn in feature_names}])

    # Scale
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    # Probability for positive class
    prob = model.predict_proba(input_scaled)[0, 1]
    pct  = prob * 100

    st.divider()
    st.subheader(f"📊 Risk Assessment: **{cfg['positive_label']}**")

    # Colour-coded risk band
    if pct < 30:
        colour, band = "🟢", "Low risk"
    elif pct < 60:
        colour, band = "🟡", "Moderate risk"
    else:
        colour, band = "🔴", "High risk"

    st.metric(label=f"{colour} {band}", value=f"{pct:.1f}%",
              help="Probability estimated by the machine-learning model.")
    st.progress(int(pct))

    st.caption(
        f"The model estimates a **{pct:.1f}% probability** that this profile "
        f"is consistent with {cfg['positive_label'].lower()}."
    )

    st.divider()

    # SHAP explanation
    with st.spinner("Generating explanation (this may take a few seconds for SVM models)…"):
        try:
            explanation = get_shap_explanation(
                model, background, input_scaled, feature_names
            )
            render_shap_chart(explanation, cfg["positive_label"])
        except Exception as e:
            st.warning(
                f"SHAP explanation could not be generated: {e}\n\n"
                "The risk score above is still valid."
            )

    st.divider()
    st.info(
        "💡 **How to read this:** Each bar shows how much a feature "
        "moved your risk score away from the population average. "
        "Longer red bars mean that feature strongly **increases** your "
        "predicted risk; longer blue bars mean it **decreases** it."
    )
