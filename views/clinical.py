import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from train import FEATURES, MODEL_PATH, load_data
from ui import BAD, GOOD, card_header, gauge, hero, impact_bar, rate_bar, result_box, style, tips

# ---------- Codes (as used in heart_disease_data.csv) ----------
CP = {0: "Asymptomatic (no chest pain)", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Typical angina"}
RESTECG = {1: "Normal", 2: "ST-T wave abnormality", 0: "Left ventricular hypertrophy"}
SLOPE = {2: "Upsloping (normal)", 1: "Flat", 0: "Downsloping"}
THAL = {2: "Normal", 1: "Fixed defect", 3: "Reversible defect"}

LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting blood pressure",
    "chol": "Cholesterol", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "hr_pct": "Max heart rate (vs. age)", "exang": "Exercise angina", "oldpeak": "ST depression",
    "slope": "ST slope", "ca": "Blocked major vessels", "thal": "Thalassemia / perfusion",
}

PRESETS = {
    "Custom": None,
    "🏃 Fit young adult": dict(age=30, sex=0, cp=0, trestbps=115, chol=180, restecg=1,
                               thalach=185, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2),
    "🧑 Average middle-aged": dict(age=50, sex=1, cp=2, trestbps=130, chol=230, restecg=1,
                                  thalach=155, exang=0, oldpeak=0.8, slope=2, ca=0, thal=2),
    "⚠️ High-risk profile": dict(age=63, sex=1, cp=0, trestbps=150, chol=290, restecg=0,
                                 thalach=115, exang=1, oldpeak=2.8, slope=1, ca=2, thal=3),
}
DEFAULTS = PRESETS["🧑 Average middle-aged"]


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    # Average encoded patient, used as the baseline for "what pushed risk up/down"
    baseline = pipe[:-1].transform(load_data()[FEATURES]).mean(axis=0)
    return pipe, bundle["meta"], baseline


@st.cache_data
def dataset():
    return load_data()


def contributions(pipe, baseline, row):
    """Per-feature contribution to the log-odds vs. an average patient (logistic regression only)."""
    clf = pipe["clf"]
    if not hasattr(clf, "coef_"):
        return None
    x = pipe[:-1].transform(row)[0]
    contrib = clf.coef_[0] * (x - baseline)
    # "num__hr_pct" -> "hr_pct", "cat__thal_2.0" -> "thal" (one-hot columns grouped back)
    names = [n.split("__", 1)[1] if n.startswith(("num", "bin")) else n.split("__", 1)[1].rsplit("_", 1)[0]
             for n in pipe["pre"].get_feature_names_out()]
    return pd.Series(contrib, index=names).groupby(level=0).sum().sort_values()


def apply_preset():
    preset = PRESETS[st.session_state.preset]
    if preset:
        for k, v in preset.items():
            st.session_state[k] = v


try:
    pipe, meta, baseline = load_model()
except (FileNotFoundError, KeyError):
    st.error("Model not found or outdated. Run `python train.py` first.")
    st.stop()

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

hero("🫀 Clinical Check",
     "Estimate heart disease risk from 12 clinical measurements using machine learning.",
     [f"🤖 {meta['model_name']}", f"🎯 {meta['test_accuracy']*100:.0f}% test accuracy",
      f"📈 ROC-AUC {meta['test_roc_auc']:.2f}", f"🧪 {meta['n_samples']} patients (UCI Cleveland)"])

tab_predict, tab_data, tab_about = st.tabs(["🔍 Risk Assessment", "📊 Data Insights", "ℹ️ About"])

# ---------- Prediction tab ----------
with tab_predict:
    st.selectbox("Quick-fill with an example patient", list(PRESETS), key="preset", on_change=apply_preset)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            card_header("👤 Personal", "Basic demographics")
            st.slider("Age", 20, 90, key="age")
            st.radio("Sex", [0, 1], key="sex", horizontal=True,
                     format_func=lambda x: "Male" if x else "Female")
            st.slider("Resting blood pressure (mm Hg)", 90, 200, key="trestbps",
                      help="Normal is below 120 mm Hg")
            st.slider("Cholesterol (mg/dl)", 120, 570, key="chol", help="Desirable is below 200 mg/dl")

    with c2:
        with st.container(border=True):
            card_header("🩺 Symptoms & ECG", "From clinical exam")
            st.selectbox("Chest pain type", list(CP), key="cp", format_func=CP.get,
                         help="In this dataset, patients with no chest pain who were still referred "
                              "for testing ('silent' ischemia) had the highest disease rate.")
            st.selectbox("Resting ECG", list(RESTECG), key="restecg", format_func=RESTECG.get)
            st.radio("Chest pain during exercise", [0, 1], key="exang", horizontal=True,
                     format_func=lambda x: "Yes" if x else "No")
            st.selectbox("Thalassemia / blood flow scan", list(THAL), key="thal", format_func=THAL.get)

    with c3:
        with st.container(border=True):
            card_header("🏃 Stress Test", "Exercise ECG & angiography")
            st.slider("Max heart rate achieved (bpm)", 70, 205, key="thalach",
                      help=f"Age-predicted max ≈ {220 - st.session_state.age} bpm")
            st.slider("ST depression (oldpeak)", 0.0, 6.2, step=0.1, key="oldpeak",
                      help="ST segment drop during exercise vs rest. 0 is normal.")
            st.selectbox("ST segment slope", list(SLOPE), key="slope", format_func=SLOPE.get)
            st.select_slider("Major vessels coloured by fluoroscopy", [0, 1, 2, 3], key="ca",
                             help="Number of major vessels with visible narrowing (0 is best)")

    st.write("")
    go_btn = st.button("Analyze my heart risk", width="stretch")

    if go_btn:
        inputs = {f: st.session_state[f] for f in FEATURES}
        row = pd.DataFrame([inputs])[FEATURES]
        risk = float(pipe.predict_proba(row)[0, 1])

        level, css, msg = (
            ("Low risk", "low", "Your profile looks similar to healthy patients.") if risk < 0.35 else
            ("Moderate risk", "medium", "Some indicators are concerning. Consider a check-up.") if risk < 0.65 else
            ("High risk", "high", "Profile resembles patients with heart disease. Please see a cardiologist.")
        )

        st.divider()
        r1, r2 = st.columns([1, 1.3], gap="large")
        with r1:
            st.plotly_chart(gauge(risk * 100, 100, [(0, 35, "low"), (35, 65, "medium"), (65, 100, "high")]),
                            width="stretch")
            result_box(level, css, msg)

        with r2:
            contrib = contributions(pipe, baseline, row)
            if contrib is not None:
                st.markdown("**What drove this result** (compared with an average patient)")
                fig = impact_bar(contrib, LABELS, suffix=" log-odds")
                st.plotly_chart(fig, width="stretch")

        if contrib is not None:
            t1, t2 = st.columns(2, gap="large")
            with t1:
                st.markdown("#### 🔺 Biggest risk factors")
                tips([f"<b>{LABELS[f]}</b> is increasing your risk"
                      for f in contrib[contrib > 0.1].sort_values(ascending=False).head(4).index], "bad")
                if not (contrib > 0.1).any():
                    st.caption("No major risk factors found.")
            with t2:
                st.markdown("#### 💚 Protective factors")
                tips([f"<b>{LABELS[f]}</b> is lowering your risk"
                      for f in contrib[contrib < -0.1].head(4).index], "good")
                if not (contrib < -0.1).any():
                    st.caption("No strong protective factors found.")

        with st.expander("Show input values"):
            st.json(inputs)

        st.caption("⚠️ Educational tool only — not a medical diagnosis. Always consult a doctor.")

# ---------- Data tab ----------
with tab_data:
    df = dataset()
    df["Status"] = df["disease"].map({0: "Healthy", 1: "Disease"})
    colors = {"Healthy": GOOD, "Disease": BAD}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patients", len(df))
    m2.metric("With disease", int(df["disease"].sum()))
    m3.metric("Avg age", f"{df['age'].mean():.0f}")
    m4.metric("Model ROC-AUC", f"{meta['test_roc_auc']:.2f}")

    feat = st.selectbox("Compare a measurement between healthy and diseased patients",
                        ["thalach", "age", "oldpeak", "chol", "trestbps"], format_func=LABELS.get)
    fig = go.Figure()
    for status in ["Healthy", "Disease"]:
        fig.add_trace(go.Histogram(x=df.loc[df.Status == status, feat], name=status,
                                   marker_color=colors[status], opacity=0.75, nbinsx=25))
    style(fig, barmode="overlay", xaxis_title=LABELS[feat], yaxis_title="Patients")
    st.plotly_chart(fig, width="stretch")

    cat = st.selectbox("Disease rate by category", ["cp", "exang", "ca", "thal", "slope", "sex"],
                       format_func=LABELS.get)
    names = {"cp": CP, "thal": THAL, "slope": SLOPE, "exang": {0: "No", 1: "Yes"},
             "sex": {0: "Female", 1: "Male"}, "ca": {i: str(i) for i in range(4)}}[cat]
    rate = df.dropna(subset=[cat]).groupby(cat)["disease"].mean().rename(index=lambda i: names.get(int(i), i))
    fig2 = rate_bar(rate.index, rate.values * 100, ytitle="% with disease")
    st.plotly_chart(fig2, width="stretch")

# ---------- About tab ----------
with tab_about:
    st.markdown(f"""
### How it works
A **{meta['model_name']}** model trained on the UCI Cleveland heart disease dataset
({meta['n_samples']} patients). Numeric features are standardised, categorical ones are
one-hot encoded, and missing codes (`ca = 4`, `thal = 0`) are imputed.

**Feature engineering:** max heart rate is converted to a percentage of the age-predicted
maximum (`thalach / (220 − age)`), and fasting blood sugar is dropped because it carries no
signal in this dataset. Without these, the model wrongly treated older age and high blood sugar
as *protective*.

| Metric | Value |
|---|---|
| Test accuracy | {meta['test_accuracy']*100:.1f}% |
| Test ROC-AUC | {meta['test_roc_auc']:.3f} |
| 5-fold CV ROC-AUC | {meta['cv_roc_auc']:.3f} |

### Data note
In this version of the dataset the `target` column is **inverted** (`1` = healthy, `0` = disease).
The model is trained on `disease = 1 - target`, so fit people correctly get a low risk score.

### Limitations
- Small dataset (302 patients after removing 1 duplicate), all referred for heart testing — not the general population.
- Several inputs (fluoroscopy, thalium scan, ST depression) need clinical tests.
- **Not medical advice.**
""")
