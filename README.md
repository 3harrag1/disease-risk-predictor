# Disease Risk Predictor

A machine learning web application that predicts a user's risk of **Diabetes**, **Heart Disease**, or **Stroke** based on clinical and demographic inputs. Built with Streamlit and powered by scikit-learn models trained on publicly available health datasets.

Live demo: [disease-risk-predictor.streamlit.app](https://disease-risk-predictor.streamlit.app)

---

## What it does

Users select a disease, enter their health data through a simple form, and receive:

- A **risk score percentage** estimated by a trained ML model
- A **SHAP explanation chart** showing which factors increased or decreased their risk and by how much, so the prediction is transparent rather than a black box

---

## Diseases covered

| Disease | Dataset | Model selection |
|---|---|---|
| Diabetes | Pima Indians Diabetes Dataset | Best F1-score from LR, SVM, DT, RF |
| Heart Disease | UCI Heart Disease Dataset | Best F1-score from LR, SVM, DT, RF |
| Stroke | Kaggle Brain Stroke Dataset | Best F1-score from LR, SVM, DT, RF |

---

## Project structure

```
disease-risk-predictor/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
└── models/
    ├── diabetes_best.pkl
    ├── diabetes_scaler.pkl
    ├── diabetes_background.pkl
    ├── diabetes_features.pkl
    ├── heart_disease_best.pkl
    ├── heart_disease_scaler.pkl
    ├── heart_disease_background.pkl
    ├── heart_disease_features.pkl
    ├── stroke_best.pkl
    ├── stroke_scaler.pkl
    ├── stroke_background.pkl
    └── stroke_features.pkl
```

---

## Run locally

```bash
git clone https://github.com/3harrag1/disease-risk-predictor.git
cd disease-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech stack

- **Streamlit** — web interface
- **scikit-learn** — model training and inference
- **SHAP** — model explainability
- **pandas / numpy** — data handling
- **matplotlib** — SHAP visualisation
- **joblib** — model serialisation

---

## Disclaimer

This tool is for **educational purposes only**. It is not a medical device and must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional with any questions you may have regarding a medical condition.
