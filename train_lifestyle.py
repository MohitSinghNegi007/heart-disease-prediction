"""
Train the lifestyle heart disease model (v2) on CDC BRFSS 2015 survey data.

Data: "CDC Diabetes Health Indicators" (UCI ML Repository, id 891), 253,680 US adults.
Target: HeartDiseaseorAttack = has ever reported coronary heart disease or a heart attack.
The model therefore estimates how likely it is that someone with this profile HAS heart
disease (a screening estimate), not their 10-year risk of getting it.

Modelling choices:
  - HvyAlcoholConsump is dropped: heavy drinkers show LOWER heart disease rates (5.9% vs 9.6%)
    because people are told to stop drinking after a heart attack ("sick quitter" effect).
    Keeping it would teach the model that alcohol protects the heart.
  - Education, income, healthcare access and CholCheck are dropped to keep the app about
    health and lifestyle questions anyone can answer.
  - Gradient boosting with monotonic constraints: risk can only rise with smoking, blood
    pressure, age, etc. and only fall with exercise, fruit and vegetables. Unconstrained
    models gave exercise a slightly positive weight. BMI is left free (U-shaped risk).
"""
import json
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train import MODEL_DIR, ROOT

DATA_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
DATA_PATH = ROOT / "data" / "cdc_brfss_2015.csv"
MODEL_PATH = MODEL_DIR / "lifestyle_model.pkl"
INFO_PATH = MODEL_DIR / "lifestyle_info.json"
TARGET = "HeartDiseaseorAttack"

# feature -> monotonic constraint (1 = risk can only rise, -1 = only fall, 0 = free)
CONSTRAINTS = {
    "Age": 1, "Sex": 0, "BMI": 0, "HighBP": 1, "HighChol": 1, "Diabetes_binary": 1,
    "Smoker": 1, "Stroke": 1, "PhysActivity": -1, "Fruits": -1, "Veggies": -1,
    "GenHlth": 1, "MentHlth": 0, "PhysHlth": 1, "DiffWalk": 1,
}
FEATURES = list(CONSTRAINTS)


def load_data():
    if not DATA_PATH.exists():
        print(f"Downloading CDC BRFSS data to {DATA_PATH} ...")
        DATA_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected data format, missing columns: {missing}")
    return df


def rate_by(df, col, bins=None, labels=None):
    key = pd.cut(df[col], bins, labels=labels) if bins else df[col]
    r = df.groupby(key, observed=True)[TARGET].mean()
    return {str(k): round(float(v), 4) for k, v in r.items()}


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    p_base = baseline.predict_proba(X_test)[:, 1]

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
        monotonic_cst=[CONSTRAINTS[f] for f in FEATURES], random_state=42,
    ).fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]

    print(f"Logistic Regression  ROC-AUC {roc_auc_score(y_test, p_base):.4f}")
    print(f"Monotonic GBM        ROC-AUC {roc_auc_score(y_test, p):.4f}")

    # Calibration by decile: predicted vs observed rate
    cal = pd.DataFrame({"pred": p, "obs": y_test.values})
    cal["bin"] = pd.qcut(cal["pred"], 10, labels=False, duplicates="drop")
    cal = cal.groupby("bin")[["pred", "obs"]].mean().round(4)

    sample = X_test.sample(20000, random_state=42)
    imp = permutation_importance(model, sample, y_test.loc[sample.index], scoring="roc_auc",
                                 n_repeats=3, random_state=42)
    importance = dict(sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()),
                             key=lambda kv: -kv[1]))

    meta = {
        "model_name": "Gradient Boosting (monotonic)",
        "features": FEATURES,
        "n_samples": len(df),
        "prevalence": round(float(y.mean()), 4),
        "test_roc_auc": round(roc_auc_score(y_test, p), 4),
        "test_pr_auc": round(average_precision_score(y_test, p), 4),
        "test_brier": round(brier_score_loss(y_test, p), 4),
        "baseline_lr_roc_auc": round(roc_auc_score(y_test, p_base), 4),
    }
    insights = {
        "calibration": {"pred": cal["pred"].tolist(), "obs": cal["obs"].tolist()},
        "importance": importance,
        "reference": {f: float(X[f].median()) for f in FEATURES},
        "rates": {
            "Age": rate_by(df, "Age"),
            "GenHlth": rate_by(df, "GenHlth"),
            "BMI": rate_by(df, "BMI", [0, 18.5, 25, 30, 35, 100],
                           ["Under 18.5", "18.5–25", "25–30", "30–35", "35+"]),
            **{c: rate_by(df, c) for c in ["Smoker", "HighBP", "HighChol", "Diabetes_binary",
                                           "PhysActivity", "Stroke", "DiffWalk", "Sex",
                                           "Fruits", "Veggies"]},
        },
    }

    # Refit on all data for the deployed model
    model.fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, MODEL_PATH)
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **insights}, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2))
    print("Feature importance:", importance)

    fit = dict(Age=3, Sex=1, BMI=23, HighBP=0, HighChol=0, Diabetes_binary=0, Smoker=0,
               Stroke=0, PhysActivity=1, Fruits=1, Veggies=1, GenHlth=1, MentHlth=0,
               PhysHlth=0, DiffWalk=0)
    unhealthy = dict(Age=11, Sex=1, BMI=33, HighBP=1, HighChol=1, Diabetes_binary=1, Smoker=1,
                     Stroke=0, PhysActivity=0, Fruits=0, Veggies=0, GenHlth=4, MentHlth=5,
                     PhysHlth=10, DiffWalk=1)
    for label, person in [("fit", fit), ("unhealthy", unhealthy)]:
        r = model.predict_proba(pd.DataFrame([person])[FEATURES])[0, 1]
        print(f"Sanity {label}: {r:.1%}")


if __name__ == "__main__":
    main()
