# Disease Risk Predictor
# Harish Ragavendra Kalyanamurugan
# Run locally: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Disease Risk Predictor",
    layout="centered",
)

# NHS help links shown next to complex fields
NHS_LINKS = {
    "Glucose":                  "https://www.nhs.uk/conditions/blood-sugar-test/",
    "BloodPressure":            "https://www.nhs.uk/conditions/blood-pressure-test/",
    "Insulin":                  "https://www.nhs.uk/medicines/insulin/",
    "BMI":                      "https://www.nhs.uk/live-well/healthy-weight/bmi-calculator/",
    "bmi":                      "https://www.nhs.uk/live-well/healthy-weight/bmi-calculator/",
    "DiabetesPedigreeFunction": "https://www.nhs.uk/conditions/type-2-diabetes/",
    "chol":                     "https://www.nhs.uk/conditions/high-cholesterol/",
    "trestbps":                 "https://www.nhs.uk/conditions/blood-pressure-test/",
    "thalach":                  "https://www.nhs.uk/live-well/exercise/physical-activity-guidelines-for-adults-aged-19-to-64/",
    "oldpeak":                  "https://www.nhs.uk/conditions/electrocardiogram/",
    "restecg":                  "https://www.nhs.uk/conditions/electrocardiogram/",
    "thal":                     "https://www.nhs.uk/conditions/thalassaemia/",
    "ca":                       "https://www.nhs.uk/conditions/coronary-angiography/",
    "avg_glucose_level":        "https://www.nhs.uk/conditions/blood-sugar-test/",
    "hypertension":             "https://www.nhs.uk/conditions/high-blood-pressure-hypertension/",
    "smoking_status":           "https://www.nhs.uk/better-health/quit-smoking/",
}

# Each disease maps to its saved model files and the input fields shown to the user.
# Field names must exactly match the column names used during model training.
# Fields are ordered for a natural clinical intake flow rather than by dataset column order.
# The model receives inputs ordered by feature_names loaded from the pkl file,
# so visual reordering here has no effect on predictions.
DISEASE_CONFIG = {
    "Diabetes": {
        "model":          "models/diabetes_best.pkl",
        "scaler":         "models/diabetes_scaler.pkl",
        "background":     "models/diabetes_background.pkl",
        "features":       "models/diabetes_features.pkl",
        "description":    "Estimates the likelihood of Type 2 Diabetes using metabolic markers from the Pima Indians Diabetes dataset.",
        "positive_label": "Diabetic",
        "data_info": (
            "What you will need: most values here come from a standard fasting blood test and a routine GP check-up. "
            "Plasma glucose, serum insulin, and blood pressure are measured during a fasting blood test. "
            "BMI is calculated from your height and weight. "
            "Skin fold thickness is measured by a clinician using callipers. "
            "Your Diabetes Pedigree Function score, which reflects genetic risk from family history, "
            "may be provided by a GP or diabetes specialist if you have been assessed for diabetes risk."
        ),
        # Pregnancies is listed first so it appears immediately below the sex selector
        # for female users. For male users it is hidden and set to 0 automatically.
        "fields": [
            dict(name="Pregnancies",
                 label="Number of Pregnancies",
                 plain="How many times have you been pregnant?",
                 widget="int_slider", min=0, max=20, default=1),
            dict(name="Age",
                 label="Age (years)",
                 plain="Your age in years.",
                 widget="int_slider", min=21, max=90, default=33),
            dict(name="BMI",
                 label="BMI (kg/m2)",
                 plain="Body Mass Index. Your weight in kg divided by your height in metres squared. Healthy range is 18.5 to 24.9.",
                 widget="float_input", min=15.0, max=70.0, default=32.0, step=0.1),
            dict(name="Glucose",
                 label="Plasma Glucose (mg/dL)",
                 plain="Your blood sugar level from a glucose test. Normal fasting range is 70-99 mg/dL.",
                 widget="int_slider", min=44, max=300, default=120),
            dict(name="BloodPressure",
                 label="Diastolic Blood Pressure (mm Hg)",
                 plain="The lower number in a blood pressure reading (e.g. 80 in 120/80). Normal is under 80.",
                 widget="int_slider", min=40, max=130, default=72),
            dict(name="Insulin",
                 label="Serum Insulin (uU/mL)",
                 plain="Insulin level in your blood measured 2 hours after a glucose test. Normal is under 140 uU/mL.",
                 widget="int_slider", min=14, max=846, default=79),
            dict(name="SkinThickness",
                 label="Skin Fold Thickness (mm)",
                 plain="A measure of body fat taken from the back of your upper arm. Typical range is 10-50 mm.",
                 widget="int_slider", min=7, max=99, default=23),
            dict(name="DiabetesPedigreeFunction",
                 label="Diabetes Pedigree Function",
                 plain="A score between 0.05 and 2.5 estimating your genetic risk based on family history of diabetes. Higher means stronger family history.",
                 widget="float_input", min=0.05, max=2.5, default=0.47, step=0.01),
        ],
    },

    "Heart Disease": {
        "model":          "models/heart_disease_best.pkl",
        "scaler":         "models/heart_disease_scaler.pkl",
        "background":     "models/heart_disease_background.pkl",
        "features":       "models/heart_disease_features.pkl",
        "description":    "Estimates the likelihood of coronary heart disease using clinical measurements from the UCI Heart Disease dataset.",
        "positive_label": "Heart Disease",
        "data_info": (
            "What you will need: some fields here require specialist cardiac assessment. "
            "Resting blood pressure and serum cholesterol are available from a standard GP blood test. "
            "Max heart rate achieved, ST depression (oldpeak), and ST segment slope are recorded during a cardiac stress test. "
            "The resting ECG result comes from an electrocardiogram, which your GP can refer you for. "
            "The number of narrowed major vessels requires a coronary angiography. "
            "Thalassemia type is determined through a blood disorder screening test. "
            "If you have not had these specialist tests, you may not have all the values needed."
        ),
        "fields": [
            dict(name="age",
                 label="Age (years)",
                 plain="Your age in years.",
                 widget="int_slider", min=29, max=77, default=54),
            dict(name="sex",
                 label="Sex",
                 plain="Your biological sex.",
                 widget="radio", options={"Female": 0, "Male": 1}, default="Female"),
            dict(name="trestbps",
                 label="Resting Blood Pressure (mm Hg)",
                 plain="The upper number in your blood pressure reading when at rest. Normal is under 120.",
                 widget="int_slider", min=80, max=200, default=131),
            dict(name="chol",
                 label="Serum Cholesterol (mg/dL)",
                 plain="Total cholesterol level in your blood. Healthy is under 200 mg/dL.",
                 widget="int_slider", min=100, max=600, default=246),
            dict(name="fbs",
                 label="Fasting Blood Sugar",
                 plain="Was your blood sugar above 120 mg/dL after fasting?",
                 widget="radio",
                 options={"No, it was 120 or below": 0, "Yes, it was above 120": 1},
                 default="No, it was 120 or below"),
            dict(name="cp",
                 label="Chest Pain Type",
                 plain="What type of chest pain, if any, do you experience?",
                 widget="select",
                 options={"Typical angina - squeezing chest pain triggered by activity": 0,
                          "Atypical angina - chest pain not matching the usual pattern": 1,
                          "Non-anginal pain - chest pain unrelated to the heart": 2,
                          "Asymptomatic - no chest pain": 3},
                 default="Asymptomatic - no chest pain"),
            dict(name="exang",
                 label="Exercise-Induced Chest Pain",
                 plain="Did you experience chest pain or discomfort during exercise?",
                 widget="radio", options={"No": 0, "Yes": 1}, default="No"),
            dict(name="thalach",
                 label="Max Heart Rate Achieved (bpm)",
                 plain="The highest heart rate you reached during an exercise stress test.",
                 widget="int_slider", min=60, max=220, default=149),
            dict(name="oldpeak",
                 label="ST Depression (Oldpeak)",
                 plain="A measurement from your ECG showing how much your heart electrical activity dips during exercise compared to rest. 0 means no change.",
                 widget="float_input", min=0.0, max=6.2, default=1.0, step=0.1),
            dict(name="slope",
                 label="ST Segment Slope",
                 plain="The direction of the heart electrical activity change during peak exercise.",
                 widget="select",
                 options={"Going upward (upsloping)": 0,
                          "Flat (no change)": 1,
                          "Going downward (downsloping)": 2},
                 default="Flat (no change)"),
            dict(name="restecg",
                 label="Resting ECG Result",
                 plain="Result of your heart electrical activity test (ECG) while at rest.",
                 widget="select",
                 options={"Normal": 0,
                          "Minor abnormality in heart electrical pattern (ST-T change)": 1,
                          "Enlarged left side of heart (left ventricular hypertrophy)": 2},
                 default="Normal"),
            dict(name="ca",
                 label="Number of Major Vessels Narrowed (0-4)",
                 plain="How many of the main heart arteries showed narrowing on an X-ray scan. 0 means none were narrowed.",
                 widget="int_slider", min=0, max=4, default=0),
            dict(name="thal",
                 label="Thalassemia",
                 plain="A blood disorder affecting how your body carries oxygen. Most people are normal.",
                 widget="select",
                 options={"No information available": 0,
                          "Normal": 1,
                          "Fixed defect - permanent reduced blood flow to part of heart": 2,
                          "Reversible defect - reduced flow during stress but normal at rest": 3},
                 default="Normal"),
        ],
    },

    "Stroke": {
        "model":          "models/stroke_best.pkl",
        "scaler":         "models/stroke_scaler.pkl",
        "background":     "models/stroke_background.pkl",
        "features":       "models/stroke_features.pkl",
        "description": "Estimates the likelihood of stroke using demographic and clinical risk factors from the Kaggle Stroke Prediction dataset.",
        "positive_label": "Stroke",
        "data_info": (
            "What you will need: most information here is demographic and lifestyle-based, which you will already know. "
            "Your hypertension and heart disease status should be based on a confirmed GP diagnosis. "
            "Average glucose level is obtained from a fasting blood test, available through your GP or a private health check. "
            "BMI is calculated from your height and weight; many pharmacies and GP surgeries can measure this for you."
        ),
        "fields": [
            dict(name="age",
                 label="Age (years)",
                 plain="Your age in years.",
                 widget="int_slider", min=0, max=100, default=43),
            dict(name="gender",
                 label="Gender",
                 plain="Your biological sex.",
                 widget="radio", options={"Female": 0, "Male": 1}, default="Female"),
            dict(name="hypertension",
                 label="High Blood Pressure (Hypertension)",
                 plain="Have you been diagnosed with high blood pressure?",
                 widget="radio", options={"No": 0, "Yes": 1}, default="No"),
            dict(name="heart_disease",
                 label="Heart Disease",
                 plain="Have you been diagnosed with any heart condition?",
                 widget="radio", options={"No": 0, "Yes": 1}, default="No"),
            dict(name="bmi",
                 label="BMI (kg/m2)",
                 plain="Body Mass Index. Your weight in kg divided by your height in metres squared. Healthy range is 18.5 to 24.9.",
                 widget="float_input", min=10.0, max=70.0, default=28.9, step=0.1),
            dict(name="avg_glucose_level",
                 label="Average Glucose Level (mg/dL)",
                 plain="Your average blood sugar level. Normal is under 100 mg/dL when fasting.",
                 widget="float_input", min=55.0, max=300.0, default=106.0, step=0.1),
            dict(name="smoking_status",
                 label="Smoking Status",
                 plain="What is your current smoking status?",
                 widget="select",
                 options={"Unknown or prefer not to say": 0,
                          "Former smoker": 1,
                          "Never smoked": 2,
                          "Current smoker": 3},
                 default="Never smoked"),
            dict(name="ever_married",
                 label="Ever Married",
                 plain="Have you ever been married or in a civil partnership?",
                 widget="radio", options={"No": 0, "Yes": 1}, default="No"),
            dict(name="work_type",
                 label="Work Type",
                 plain="What best describes your current or most recent work situation?",
                 widget="select",
                 options={"Government job": 0,
                          "Never worked": 1,
                          "Private sector": 2,
                          "Self-employed": 3,
                          "Child (under working age)": 4},
                 default="Private sector"),
            dict(name="Residence_type",
                 label="Residence Type",
                 plain="Do you live in an urban or rural area?",
                 widget="radio", options={"Rural": 0, "Urban": 1}, default="Urban"),
        ],
    },
}


# Load model files once per session and cache them.
# Increment _version by 1 each time new pkl files are deployed
# to force Streamlit to reload from disk rather than use the cached version.
@st.cache_resource(show_spinner="Loading model...")
def load_artefacts(disease_key: str, _version: int = 2):
    cfg = DISEASE_CONFIG[disease_key]
    model      = joblib.load(cfg["model"])
    scaler     = joblib.load(cfg["scaler"])
    background = joblib.load(cfg["background"])
    features   = joblib.load(cfg["features"])
    return model, scaler, background, features


# Render a single input field with a plain description and optional NHS link
def render_field(field: dict):
    w     = field["widget"]
    label = field["label"]
    plain = field.get("plain", "")
    nhs   = NHS_LINKS.get(field["name"], "")

    if plain:
        if nhs:
            st.caption(f"{plain} [Learn more on NHS]({nhs})")
        else:
            st.caption(plain)

    if w == "int_slider":
        return st.slider(label, min_value=field["min"], max_value=field["max"],
                         value=field["default"], step=1)

    if w == "float_input":
        return st.number_input(label, min_value=float(field["min"]),
                               max_value=float(field["max"]),
                               value=float(field["default"]),
                               step=float(field["step"]))

    if w == "radio":
        opts   = field["options"]
        chosen = st.radio(label, list(opts.keys()),
                          index=list(opts.keys()).index(field["default"]),
                          horizontal=True)
        return opts[chosen]

    if w == "select":
        opts   = field["options"]
        chosen = st.selectbox(label, list(opts.keys()),
                              index=list(opts.keys()).index(field["default"]))
        return opts[chosen]

    raise ValueError(f"Unknown widget type: {w}")


# Build a SHAP explanation for the positive class.
# input_raw is passed separately so the waterfall chart and impact table
# display the original user-entered values rather than the scaled equivalents.
# Tested against SHAP 0.49.1 confirmed output shapes:
#   LinearExplainer  -> sv: (1, n_features),    ev: scalar
#   TreeExplainer    -> sv: (1, n_features, 2), ev: array shape (2,)
#   KernelExplainer  -> sv: (1, n_features, 2), ev: array shape (2,)
def get_shap_explanation(model, background_data, input_scaled: np.ndarray,
                          feature_names: list, input_raw: np.ndarray,
                          invert_shap: bool = False) -> shap.Explanation:
    X = input_scaled  # shape (1, n_features)

    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(model)
        sv        = np.array(explainer.shap_values(X))
        values    = sv[0, :, 1].astype(float)
        base_val  = float(np.array(explainer.expected_value)[1])

    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, background_data)
        sv        = np.array(explainer.shap_values(X))
        values    = sv[0].astype(float)
        base_val  = float(explainer.expected_value)

    else:
        # SVM: KernelExplainer returns (1, n_features, 2) in SHAP 0.49.1
        bg_summary = shap.kmeans(background_data, min(50, len(background_data)))
        explainer  = shap.KernelExplainer(model.predict_proba, bg_summary)
        sv         = np.array(explainer.shap_values(X, nsamples=200))
        values     = sv[0, :, 1].astype(float)
        base_val   = float(np.array(explainer.expected_value)[1])

    # The heart disease dataset encodes class 1 = no disease, so the predicted
    # probability is inverted in the main block to display correct risk.
    # SHAP values are negated here so feature directions remain consistent
    # with the displayed risk score rather than the original model class.
    if invert_shap:
        values   = -values
        base_val = -base_val

    return shap.Explanation(
        values=values,
        base_values=base_val,
        data=input_raw[0],
        feature_names=feature_names,
    )


# Render the SHAP waterfall chart and an impact table
def render_shap_chart(explanation: shap.Explanation, disease_label: str):
    st.subheader("Why did the model give this score?")
    st.caption(
        "The chart below shows which factors pushed your risk score up (red) "
        "or down (blue), and by how much. The starting point is the average "
        "risk across everyone in the training dataset."
    )

    fig, _ = plt.subplots(figsize=(8, max(4, len(explanation.values) * 0.45)))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"Feature contributions - {disease_label}", pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    impact_df = pd.DataFrame({
        "Feature":     explanation.feature_names,
        "Your Value":  explanation.data,
        "SHAP Impact": explanation.values,
    }).sort_values("SHAP Impact", key=abs, ascending=False).reset_index(drop=True)

    impact_df["Direction"]   = impact_df["SHAP Impact"].apply(
        lambda v: "Increases risk" if v > 0 else "Decreases risk"
    )
    impact_df["SHAP Impact"] = impact_df["SHAP Impact"].map("{:+.4f}".format)
    impact_df["Your Value"]  = impact_df["Your Value"].map(
        lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    )

    st.dataframe(impact_df, width='stretch', hide_index=True)


# ─── Main UI ─────────────────────────────────────────────────────────────────

st.title("Disease Risk Predictor")
st.markdown(
    "> **Disclaimer:** This tool is for **educational purposes only**. "
    "It is not a medical device and must not replace professional medical advice. "
    "Always consult a qualified clinician for health decisions."
)

st.divider()

disease = st.radio(
    "Select a condition to assess:",
    list(DISEASE_CONFIG.keys()),
    horizontal=True,
)

cfg = DISEASE_CONFIG[disease]
st.markdown(f"_{cfg['description']}_")
st.info(cfg["data_info"])
st.divider()

# Check all required model files exist
missing_files = [p for p in [cfg["model"], cfg["scaler"],
                              cfg["background"], cfg["features"]]
                 if not Path(p).exists()]
if missing_files:
    st.error(
        "The following model files are missing:\n\n"
        + "\n".join(f"- `{p}`" for p in missing_files)
        + "\n\nRun the training notebook and copy the `.pkl` files into `models/`."
    )
    st.stop()

model, scaler, background, feature_names = load_artefacts(disease, _version=2)

st.subheader("Enter your health data")
st.caption("Descriptions are shown beneath each field. Click any NHS link to learn more about a measurement you are unsure about or if you need more information.")

# Initialise session state keys on first load
if "form_submitted"    not in st.session_state:
    st.session_state.form_submitted    = False
if "last_disease"      not in st.session_state:
    st.session_state.last_disease      = None
if "submitted_values"  not in st.session_state:
    st.session_state.submitted_values  = {}
if "submitted_disease" not in st.session_state:
    st.session_state.submitted_disease = None

# Clear results when the user switches between disease tabs
if st.session_state.last_disease != disease:
    st.session_state.last_disease   = disease
    st.session_state.form_submitted = False

# For diabetes, the sex selector is rendered at full page width before the column
# grid. This is the only reliable approach in Streamlit for dynamic show/hide,
# because st.form defers all widget updates until submission and st.columns does
# not reliably accept content added to a column object after it has been created.
# The sex value is purely for UI control and is never passed to the model.
show_pregnancies = True
if disease == "Diabetes":
    st.caption("Your biological sex determines whether the Number of Pregnancies field appears below.")
    diabetes_sex = st.radio(
        "Sex",
        ["Female", "Male"],
        horizontal=True,
        key="diabetes_sex_selector",
    )
    show_pregnancies = (diabetes_sex == "Female")

fields   = cfg["fields"]
values   = {}
col_left, col_right = st.columns(2)
col_idx  = 0

for field in fields:
    if field["name"] == "Pregnancies" and not show_pregnancies:
        values["Pregnancies"] = 0
        continue
    col = col_left if col_idx % 2 == 0 else col_right
    with col:
        values[field["name"]] = render_field(field)
    col_idx += 1

st.write("")
if st.button("Assess my risk", use_container_width=True):
    st.session_state.form_submitted    = True
    st.session_state.submitted_values  = values.copy()
    st.session_state.submitted_disease = disease

if (st.session_state.form_submitted
        and st.session_state.submitted_disease == disease):

    submitted_values = st.session_state.submitted_values
    input_df         = pd.DataFrame([{fn: submitted_values[fn] for fn in feature_names}])
    input_scaled     = scaler.transform(input_df)

    raw_prob = model.predict_proba(input_scaled)[0, 1]
    # The heart disease dataset encodes target 1 = no disease, so the
    # probability of class 1 is inverted to represent disease risk correctly.
    prob = (1 - raw_prob) if disease == "Heart Disease" else raw_prob
    pct  = prob * 100

    st.divider()
    st.subheader(f"Risk Assessment: {cfg['positive_label']}")

    if pct < 30:
        band = "Low risk"
    elif pct < 60:
        band = "Moderate risk"
    else:
        band = "High risk"

    st.metric(label=band, value=f"{pct:.1f}%",
              help="Probability estimated by the machine learning model.")
    st.progress(int(pct))
    st.caption(
        f"The model estimates a **{pct:.1f}% probability** that this profile "
        f"is consistent with {cfg['positive_label'].lower()}."
    )

    st.divider()

    with st.spinner("Generating explanation..."):
        try:
            explanation = get_shap_explanation(
                model, background, input_scaled, feature_names,
                input_raw=input_df.values,
                invert_shap=(disease == "Heart Disease")
            )
            render_shap_chart(explanation, cfg["positive_label"])
        except Exception as e:
            st.warning(
                f"SHAP explanation could not be generated: {e}\n\n"
                "The risk score above is still valid."
            )

    st.divider()
    st.info(
        "**How to read this:** Each bar shows how much a feature "
        "moved your risk score away from the population average. "
        "Longer red bars mean that feature strongly increases your "
        "predicted risk. Longer blue bars mean it decreases it."
    )