import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from train_lifestyle import FEATURES, INFO_PATH, MODEL_PATH
from ui import BAD, GOOD, card_header, gauge, hero, impact_bar, rate_bar, result_box, style, tips

AGE = {1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49", 7: "50–54",
       8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"}
GENHLTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
YESNO = {0: "No", 1: "Yes"}

LABELS = {
    "Age": "Age", "Sex": "Sex", "BMI": "BMI", "HighBP": "High blood pressure",
    "HighChol": "High cholesterol", "Diabetes_binary": "Diabetes / prediabetes", "Smoker": "Smoking history",
    "Stroke": "Previous stroke", "PhysActivity": "Physical activity", "Fruits": "Daily fruit",
    "Veggies": "Daily vegetables", "GenHlth": "General health", "MentHlth": "Poor mental health days",
    "PhysHlth": "Poor physical health days", "DiffWalk": "Difficulty walking",
}

# Things a person can change: (feature, healthier value, description)
CHANGES = [
    ("Smoker", 0, "🚭 Compared with a never-smoker"),
    ("HighBP", 0, "🩺 Get blood pressure under control"),
    ("HighChol", 0, "🧪 Bring cholesterol into the normal range"),
    ("Diabetes_binary", 0, "🍬 Keep blood sugar in the normal range"),
    ("BMI", 24.9, "⚖️ Reach a healthy weight (BMI under 25)"),
    ("PhysActivity", 1, "🏃 Exercise regularly"),
    ("GenHlth", 2, "💪 Improve overall health to 'very good'"),
]

PRESETS = {
    "Custom": None,
    "🏃 Fit 30-year-old": dict(ls_age=3, ls_sex=1, ls_height=178, ls_weight=72, ls_bp=0, ls_chol=0,
                              ls_diab=0, ls_stroke=0, ls_smoke=0, ls_active=1, ls_fruit=1, ls_veg=1,
                              ls_gen=1, ls_phys=0, ls_ment=0, ls_walk=0),
    "🧑 Typical 55-year-old": dict(ls_age=8, ls_sex=0, ls_height=163, ls_weight=75, ls_bp=1, ls_chol=1,
                                  ls_diab=0, ls_stroke=0, ls_smoke=0, ls_active=1, ls_fruit=1, ls_veg=1,
                                  ls_gen=3, ls_phys=2, ls_ment=2, ls_walk=0),
    "⚠️ High-risk 70-year-old": dict(ls_age=11, ls_sex=1, ls_height=175, ls_weight=101, ls_bp=1, ls_chol=1,
                                     ls_diab=1, ls_stroke=0, ls_smoke=1, ls_active=0, ls_fruit=0, ls_veg=0,
                                     ls_gen=4, ls_phys=10, ls_ment=5, ls_walk=1),
}
DEFAULTS = PRESETS["🧑 Typical 55-year-old"]


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    with open(INFO_PATH, encoding="utf-8") as f:
        info = json.load(f)
    return bundle["model"], info


def apply_preset():
    preset = PRESETS[st.session_state.ls_preset]
    if preset:
        st.session_state.update(preset)


def predict(model, person):
    return float(model.predict_proba(pd.DataFrame([person])[FEATURES])[0, 1])


try:
    model, info = load_model()
except FileNotFoundError:
    st.error("Lifestyle model not found. Run `python train_lifestyle.py` first.")
    st.stop()
meta = info["meta"]

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

hero("🏃 Lifestyle Check",
     "Answer everyday health questions — no hospital tests needed.",
     [f"🤖 {meta['model_name']}", f"📈 ROC-AUC {meta['test_roc_auc']:.2f}",
      f"👥 {meta['n_samples']:,} US adults (CDC BRFSS)"],
     variant="lifestyle")

tab_check, tab_data, tab_model = st.tabs(["🔍 Risk Check", "📊 Population Insights", "🧠 Model Details"])

# ---------- Risk check ----------
with tab_check:
    st.selectbox("Quick-fill with an example person", list(PRESETS), key="ls_preset", on_change=apply_preset)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            card_header("👤 About you", "Age, sex and body size")
            st.select_slider("Age group", list(AGE), key="ls_age", format_func=AGE.get)
            st.radio("Sex", [0, 1], key="ls_sex", horizontal=True, format_func=lambda x: "Male" if x else "Female")
            st.number_input("Height (cm)", 120, 220, key="ls_height")
            st.number_input("Weight (kg)", 30, 250, key="ls_weight")
            bmi = st.session_state.ls_weight / (st.session_state.ls_height / 100) ** 2
            st.caption(f"Your BMI: **{bmi:.1f}**")

    with c2:
        with st.container(border=True):
            card_header("🩺 Health conditions", "Has a doctor ever told you that you have…")
            st.radio("High blood pressure", [0, 1], key="ls_bp", horizontal=True, format_func=YESNO.get)
            st.radio("High cholesterol", [0, 1], key="ls_chol", horizontal=True, format_func=YESNO.get)
            st.radio("Diabetes or prediabetes", [0, 1], key="ls_diab", horizontal=True, format_func=YESNO.get)
            st.radio("A stroke", [0, 1], key="ls_stroke", horizontal=True, format_func=YESNO.get)

    with c3:
        with st.container(border=True):
            card_header("🥗 Lifestyle & wellbeing", "Your habits over the past month")
            st.radio("Smoked at least 100 cigarettes in your life", [0, 1], key="ls_smoke",
                     horizontal=True, format_func=YESNO.get)
            st.radio("Exercised in the past 30 days (outside work)", [0, 1], key="ls_active",
                     horizontal=True, format_func=YESNO.get)
            st.radio("Eat fruit daily", [0, 1], key="ls_fruit", horizontal=True, format_func=YESNO.get)
            st.radio("Eat vegetables daily", [0, 1], key="ls_veg", horizontal=True, format_func=YESNO.get)

    with st.container(border=True):
        card_header("💭 How you feel", "Self-rated health over the past 30 days")
        g1, g2, g3, g4 = st.columns(4, gap="medium")
        g1.select_slider("General health", list(GENHLTH), key="ls_gen", format_func=GENHLTH.get)
        g2.slider("Days physical health was not good", 0, 30, key="ls_phys")
        g3.slider("Days mental health was not good", 0, 30, key="ls_ment")
        g4.radio("Serious difficulty walking or climbing stairs", [0, 1], key="ls_walk",
                 horizontal=True, format_func=YESNO.get)

    st.write("")
    if st.button("Check my heart risk", width="stretch", key="ls_go"):
        s = st.session_state
        person = dict(Age=s.ls_age, Sex=s.ls_sex, BMI=round(bmi), HighBP=s.ls_bp, HighChol=s.ls_chol,
                      Diabetes_binary=s.ls_diab, Smoker=s.ls_smoke, Stroke=s.ls_stroke,
                      PhysActivity=s.ls_active, Fruits=s.ls_fruit, Veggies=s.ls_veg, GenHlth=s.ls_gen,
                      MentHlth=s.ls_ment, PhysHlth=s.ls_phys, DiffWalk=s.ls_walk)
        risk = predict(model, person)
        age_avg = info["rates"]["Age"][str(s.ls_age)]

        level, css, msg = (
            ("Low", "low", "Your answers look like people who rarely have heart disease.") if risk < 0.07 else
            ("Moderate", "medium", "Some risk factors are present. A routine check-up is a good idea.") if risk < 0.20 else
            ("High", "high", "Your answers resemble people with heart disease. Please talk to a doctor.")
        )

        st.divider()
        r1, r2 = st.columns([1, 1.3], gap="large")
        with r1:
            st.plotly_chart(gauge(risk * 100, 60, [(0, 7, "low"), (7, 20, "medium"), (20, 60, "high")],
                                  title="Chance of having heart disease"), width="stretch")
            result_box(f"{level} risk", css, msg)
        with r2:
            st.markdown("**How you compare**")
            comp = pd.Series({"You": risk, f"Average age {AGE[s.ls_age]}": age_avg,
                              "Average US adult": meta["prevalence"]}) * 100
            fig = go.Figure(go.Bar(
                x=comp.values, y=comp.index, orientation="h",
                marker=dict(color=[BAD if risk > age_avg else GOOD, "#9ca3af", "#9ca3af"], cornerradius=4),
                text=[f"{v:.1f}%" for v in comp.values], textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            style(fig, height=200, xaxis_range=[0, comp.max() * 1.25], xaxis_title="% with heart disease",
                  yaxis_autorange="reversed", margin=dict(t=10, b=40, l=10, r=10))
            st.plotly_chart(fig, width="stretch")

            # Impact: change in risk if this one answer were that of a typical adult
            ref = info["reference"]
            impact = pd.Series({f: risk - predict(model, {**person, f: ref[f]}) for f in FEATURES}) * 100
            impact = impact[impact.abs() >= 0.1].sort_values()
            st.markdown("**What drove this result** (vs. a typical adult, in percentage points)")
            if impact.empty:
                st.caption("Your answers are close to a typical adult's.")
            else:
                st.plotly_chart(impact_bar(impact, LABELS, height=max(220, 34 * len(impact) + 80),
                                           suffix=" pts"), width="stretch")

        st.markdown("#### 🎯 What could lower your risk")
        options = [(desc, predict(model, {**person, f: v})) for f, v, desc in CHANGES
                   if (person[f] > v if f != "PhysActivity" else person[f] < v)]
        options = [(d, p) for d, p in options if risk - p >= 0.001]
        if options:
            all_better = {**person, **{f: v for f, v, _ in CHANGES
                                        if (person[f] > v if f != "PhysActivity" else person[f] < v)}}
            p_all = predict(model, all_better)
            items = [f"<b>{d}</b>: {risk*100:.1f}% → {p*100:.1f}%" for d, p in sorted(options, key=lambda x: x[1])]
            if len(options) > 1:
                items.append(f"<b>✨ All of the above together</b>: {risk*100:.1f}% → {p_all*100:.1f}%")
            tips(items, "good")
            st.caption("Exercise, diet and weight mainly protect the heart by improving blood pressure, "
                       "cholesterol and blood sugar, so the model shows most of their benefit through those. "
                       "Smoking is recorded as lifetime history; after quitting, risk falls gradually over years.")
        else:
            st.success("Nothing obvious to change: your modifiable risk factors are already in good shape. 💚")

        st.caption("⚠️ Educational tool only — not a medical diagnosis. Always consult a doctor.")

# ---------- Population insights ----------
with tab_data:
    rates = info["rates"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Adults surveyed", f"{meta['n_samples']:,}")
    m2.metric("Have heart disease", f"{meta['prevalence']*100:.1f}%")
    m3.metric("Smokers vs non-smokers", f"{rates['Smoker']['1']*100:.1f}% vs {rates['Smoker']['0']*100:.1f}%")
    m4.metric("Ages 80+", f"{rates['Age']['13']*100:.1f}%")

    st.markdown("**Heart disease rate by age**")
    age_rates = pd.Series(rates["Age"]).rename(index=lambda k: AGE[int(k)]) * 100
    st.plotly_chart(rate_bar(age_rates.index, age_rates.values), width="stretch")

    choice = st.selectbox("Compare by", ["General health", "BMI", "Risk factors (with vs without)"])
    if choice == "General health":
        r = pd.Series(rates["GenHlth"]).rename(index=lambda k: GENHLTH[int(k)]) * 100
        st.plotly_chart(rate_bar(r.index, r.values), width="stretch")
    elif choice == "BMI":
        r = pd.Series(rates["BMI"]) * 100
        st.plotly_chart(rate_bar(r.index, r.values), width="stretch")
        st.caption("Underweight is also linked to higher rates, so BMI risk is U-shaped.")
    else:
        cols = ["HighBP", "HighChol", "Diabetes_binary", "Smoker", "Stroke", "DiffWalk", "PhysActivity"]
        fig = go.Figure()
        for val, name, color in [("0", "Without", GOOD), ("1", "With", BAD)]:
            fig.add_trace(go.Bar(
                name=name, y=[LABELS[c] for c in cols], x=[rates[c][val] * 100 for c in cols],
                orientation="h", marker=dict(color=color, cornerradius=4),
                hovertemplate="%{y} (" + name.lower() + "): %{x:.1f}%<extra></extra>"))
        style(fig, height=420, barmode="group", xaxis_title="% with heart disease", bargap=0.3,
              bargroupgap=0.08, yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
        st.caption("Physical activity is the only factor here where having it lowers the rate.")

# ---------- Model details ----------
with tab_model:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f"""
### How it works
A **gradient boosting** model trained on {meta['n_samples']:,} answers to the CDC's Behavioral Risk
Factor Surveillance System (BRFSS 2015) survey.

| Metric | Value |
|---|---|
| ROC-AUC (test set, 50k people) | **{meta['test_roc_auc']:.3f}** |
| Baseline logistic regression | {meta['baseline_lr_roc_auc']:.3f} |
| PR-AUC (vs. {meta['prevalence']:.3f} by chance) | {meta['test_pr_auc']:.3f} |
| Brier score (lower is better) | {meta['test_brier']:.4f} |

### Design decisions
- **Monotonic constraints:** risk can only go up with smoking, blood pressure, cholesterol and age,
  and only down with exercise, fruit and vegetables. Unconstrained, the model gave exercise a
  slightly *harmful* effect.
- **Heavy alcohol use removed:** heavy drinkers show *lower* heart disease rates because
  people are told to stop drinking after a heart attack ("sick quitter" effect).
- **Calibrated probabilities:** the model is not re-weighted for class imbalance, so a 20%
  estimate means about 20 in 100 similar people have heart disease.
""")
    with b:
        st.markdown("**Calibration:** predicted vs. actual rate (10 groups)")
        cal = info["calibration"]
        fig = go.Figure()
        top = max(cal["pred"] + cal["obs"]) * 100 * 1.05
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect",
                                 line=dict(color="#9ca3af", width=2, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[v * 100 for v in cal["pred"]], y=[v * 100 for v in cal["obs"]],
                                 mode="lines+markers", name="Model", line=dict(color=GOOD, width=2),
                                 marker=dict(size=9),
                                 hovertemplate="Predicted %{x:.1f}% → actual %{y:.1f}%<extra></extra>"))
        style(fig, height=320, xaxis_title="Predicted %", yaxis_title="Actual %")
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Which answers matter most** (permutation importance)")
        imp = pd.Series(info["importance"]).clip(lower=0).sort_values()
        imp = imp[imp > 0]
        fig = go.Figure(go.Bar(x=imp.values, y=[LABELS[i] for i in imp.index], orientation="h",
                               marker=dict(color=GOOD, cornerradius=4),
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        style(fig, height=340, xaxis_title="Drop in ROC-AUC when shuffled", bargap=0.25)
        st.plotly_chart(fig, width="stretch")

    st.markdown("""
### Limitations
- The survey asks whether someone **has ever had** heart disease, so this estimates how likely it is
  that a person with your answers has it, not your future risk.
- Answers are self-reported, and some (stroke, difficulty walking, general health) can be
  *consequences* of heart disease as well as causes.
- US data from 2015; risk levels in other countries differ.
- **Not medical advice.**
""")
