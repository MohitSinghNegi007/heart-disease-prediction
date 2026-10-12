import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from train import FEATURES, MODEL_PATH, load_data

st.set_page_config(page_title="CardioCheck · Heart Disease Risk", page_icon="🫀", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #e11d48 0%, #9f1239 60%, #4c0519 100%);
    border-radius: 20px; padding: 2.2rem 2.5rem; color: white; margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(225, 29, 72, .25);
}
.hero h1 { color: white; font-size: 2.4rem; font-weight: 800; margin: 0; }
.hero p { color: #ffe4e6; font-size: 1.05rem; margin: .4rem 0 0 0; }
.hero .badges span {
    display: inline-block; background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px; padding: .25rem .8rem; margin: .9rem .4rem 0 0; font-size: .85rem;
}

.card {
    background: var(--secondary-background-color); border-radius: 16px; padding: 1.2rem 1.4rem;
    border: 1px solid rgba(128,128,128,.15); height: 100%;
}
.section-title { font-weight: 700; font-size: 1.05rem; margin-bottom: .2rem; }
.section-sub { opacity: .65; font-size: .85rem; margin-bottom: .8rem; }

.result {
    border-radius: 16px; padding: 1.4rem; text-align: center; color: white;
}
.result h2 { color: white; margin: 0; font-size: 1.8rem; font-weight: 800; }
.result p { margin: .3rem 0 0 0; opacity: .9; }
.low    { background: linear-gradient(135deg, #10b981, #047857); }
.medium { background: linear-gradient(135deg, #f59e0b, #b45309); }
.high   { background: linear-gradient(135deg, #ef4444, #991b1b); }

.tip { border-left: 4px solid; padding: .6rem .9rem; border-radius: 8px; margin-bottom: .5rem;
       background: var(--secondary-background-color); }
.tip.bad  { border-color: #ef4444; }
.tip.good { border-color: #10b981; }

div.stButton > button {
    background: linear-gradient(135deg, #e11d48, #9f1239); color: white; border: 0;
    font-weight: 700; padding: .75rem; border-radius: 12px; font-size: 1.05rem;
}
div.stButton > button:hover { filter: brightness(1.1); color: white; }
.footer { text-align: center; opacity: .6; font-size: .85rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

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

# ---------- Hero ----------
st.markdown(f"""
<div class="hero">
  <h1>🫀 CardioCheck</h1>
  <p>Estimate heart disease risk from 12 clinical measurements using machine learning.</p>
  <div class="badges">
    <span>🤖 {meta['model_name']}</span>
    <span>🎯 {meta['test_accuracy']*100:.0f}% test accuracy</span>
    <span>📈 ROC-AUC {meta['test_roc_auc']:.2f}</span>
    <span>🧪 {meta['n_samples']} patients (UCI Cleveland)</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_data, tab_about = st.tabs(["🔍 Risk Assessment", "📊 Data Insights", "ℹ️ About"])

# ---------- Prediction tab ----------
with tab_predict:
    st.selectbox("Quick-fill with an example patient", list(PRESETS), key="preset", on_change=apply_preset)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown('<div class="section-title">👤 Personal</div>'
                        '<div class="section-sub">Basic demographics</div>', unsafe_allow_html=True)
            st.slider("Age", 20, 90, key="age")
            st.radio("Sex", [0, 1], key="sex", horizontal=True,
                     format_func=lambda x: "Male" if x else "Female")
            st.slider("Resting blood pressure (mm Hg)", 90, 200, key="trestbps",
                      help="Normal is below 120 mm Hg")
            st.slider("Cholesterol (mg/dl)", 120, 570, key="chol", help="Desirable is below 200 mg/dl")

    with c2:
        with st.container(border=True):
            st.markdown('<div class="section-title">🩺 Symptoms & ECG</div>'
                        '<div class="section-sub">From clinical exam</div>', unsafe_allow_html=True)
            st.selectbox("Chest pain type", list(CP), key="cp", format_func=CP.get,
                         help="In this dataset, patients with no chest pain who were still referred "
                              "for testing ('silent' ischemia) had the highest disease rate.")
            st.selectbox("Resting ECG", list(RESTECG), key="restecg", format_func=RESTECG.get)
            st.radio("Chest pain during exercise", [0, 1], key="exang", horizontal=True,
                     format_func=lambda x: "Yes" if x else "No")
            st.selectbox("Thalassemia / blood flow scan", list(THAL), key="thal", format_func=THAL.get)

    with c3:
        with st.container(border=True):
            st.markdown('<div class="section-title">🏃 Stress Test</div>'
                        '<div class="section-sub">Exercise ECG & angiography</div>', unsafe_allow_html=True)
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
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk * 100,
                number={"suffix": "%", "font": {"size": 48}},
                title={"text": "Estimated disease probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f2937", "thickness": 0.25},
                    "steps": [{"range": [0, 35], "color": "#10b981"},
                              {"range": [35, 65], "color": "#f59e0b"},
                              {"range": [65, 100], "color": "#ef4444"}],
                }))
            gauge.update_layout(height=290, margin=dict(t=60, b=10, l=30, r=30),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge, width="stretch")
            st.markdown(f'<div class="result {css}"><h2>{level}</h2><p>{msg}</p></div>',
                        unsafe_allow_html=True)

        with r2:
            contrib = contributions(pipe, baseline, row)
            if contrib is not None:
                st.markdown("**What drove this result** (compared with an average patient)")
                fig = go.Figure(go.Bar(
                    x=contrib.values, y=[LABELS[i] for i in contrib.index], orientation="h",
                    marker_color=["#ef4444" if v > 0 else "#10b981" for v in contrib.values]))
                fig.update_layout(height=420, margin=dict(t=10, b=30, l=10, r=10),
                                  xaxis_title="← lowers risk     raises risk →",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")

        if contrib is not None:
            t1, t2 = st.columns(2, gap="large")
            with t1:
                st.markdown("#### 🔺 Biggest risk factors")
                for f, v in contrib[contrib > 0.1].sort_values(ascending=False).head(4).items():
                    st.markdown(f'<div class="tip bad"><b>{LABELS[f]}</b> is increasing your risk</div>',
                                unsafe_allow_html=True)
                if not (contrib > 0.1).any():
                    st.caption("No major risk factors found.")
            with t2:
                st.markdown("#### 💚 Protective factors")
                for f, v in contrib[contrib < -0.1].head(4).items():
                    st.markdown(f'<div class="tip good"><b>{LABELS[f]}</b> is lowering your risk</div>',
                                unsafe_allow_html=True)
                if not (contrib < -0.1).any():
                    st.caption("No strong protective factors found.")

        with st.expander("Show input values"):
            st.json(inputs)

        st.caption("⚠️ Educational tool only — not a medical diagnosis. Always consult a doctor.")

# ---------- Data tab ----------
with tab_data:
    df = dataset()
    df["Status"] = df["disease"].map({0: "Healthy", 1: "Disease"})
    colors = {"Healthy": "#10b981", "Disease": "#ef4444"}

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
    fig.update_layout(barmode="overlay", height=380, xaxis_title=LABELS[feat], yaxis_title="Patients",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    cat = st.selectbox("Disease rate by category", ["cp", "exang", "ca", "thal", "slope", "sex"],
                       format_func=LABELS.get)
    names = {"cp": CP, "thal": THAL, "slope": SLOPE, "exang": {0: "No", 1: "Yes"},
             "sex": {0: "Female", 1: "Male"}, "ca": {i: str(i) for i in range(4)}}[cat]
    rate = df.dropna(subset=[cat]).groupby(cat)["disease"].mean().rename(index=lambda i: names.get(int(i), i))
    fig2 = go.Figure(go.Bar(x=rate.index, y=rate.values * 100, marker_color="#e11d48",
                            text=[f"{v:.0f}%" for v in rate.values * 100], textposition="outside"))
    fig2.update_layout(height=380, yaxis_title="% with disease", yaxis_range=[0, 110],
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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
- Small dataset (303 patients), all referred for heart testing — not the general population.
- Several inputs (fluoroscopy, thalium scan, ST depression) need clinical tests.
- **Not medical advice.**
""")

st.markdown('<div class="footer">Developed by Mohit Singh Negi © 2026 · ML Project</div>',
            unsafe_allow_html=True)
