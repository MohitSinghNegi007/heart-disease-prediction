"""
Train the heart disease model.

IMPORTANT: in this (Kaggle) version of the UCI dataset the `target` column is
inverted: target=1 means NO disease, target=0 means disease. Patients with
target=1 are younger, reach higher max heart rates, rarely have exercise angina
and have fewer blocked vessels. We therefore train on `disease = 1 - target`.

Category codes in this file (verified against the original UCI data):
  cp      0 asymptomatic, 1 atypical angina, 2 non-anginal pain, 3 typical angina
  restecg 0 LV hypertrophy, 1 normal, 2 ST-T wave abnormality
  slope   0 downsloping, 1 flat, 2 upsloping
  ca      0-3 vessels, 4 = missing
  thal    1 fixed defect, 2 normal, 3 reversible defect, 0 = missing

Feature choices:
  - `fbs` is dropped: it has no signal here (49% vs 45% disease rate) and only adds noise.
  - Max heart rate is used as % of age-predicted max (thalach / (220 - age)).
    Raw thalach overlaps with age, which made age look protective in the model.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Raw inputs the model expects (what the app collects)
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]
# Columns used after feature engineering
NUMERIC = ["age", "trestbps", "chol", "hr_pct", "oldpeak", "ca"]
BINARY = ["sex", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart_disease_data.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "heart_disease_model.pkl"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, encoding="utf-8-sig").drop_duplicates()  # 1 duplicated patient
    df["ca"] = df["ca"].replace(4, np.nan)      # 4 = missing
    df["thal"] = df["thal"].replace(0, np.nan)  # 0 = missing
    df["disease"] = 1 - df["target"]            # fix inverted label
    return df


def add_features(X):
    X = X.copy()
    X["hr_pct"] = X["thalach"] / (220 - X["age"])  # % of age-predicted max heart rate
    return X


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("features", FunctionTransformer(add_features)), ("pre", pre), ("clf", model)])


def main():
    df = load_data()
    X, y = df[FEATURES], df["disease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5,
                                                random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, m in candidates.items():
        scores = cross_val_score(build_pipeline(m), X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = scores.mean()
        print(f"{name:20s} CV ROC-AUC {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Logistic regression is preferred when close: it is explainable per feature.
    best = max(results, key=results.get)
    if results["Logistic Regression"] >= results[best] - 0.01:
        best = "Logistic Regression"
    print("Selected:", best)

    pipe = build_pipeline(candidates[best]).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc, auc = accuracy_score(y_test, pred), roc_auc_score(y_test, proba)
    print(classification_report(y_test, pred, target_names=["Healthy", "Disease"]))

    # Refit on all data for the deployed model
    pipe = build_pipeline(candidates[best]).fit(X, y)
    meta = {
        "model_name": best,
        "features": FEATURES,
        "test_accuracy": round(acc, 3),
        "test_roc_auc": round(auc, 3),
        "cv_roc_auc": round(results[best], 3),
        "n_samples": len(df),
        "label_note": "Trained on disease = 1 - target (target is inverted in source CSV)",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "meta": meta}, MODEL_PATH)
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    # Sanity checks: a fit person must be low risk, a sick profile high risk
    fit = dict(age=30, sex=0, cp=0, trestbps=115, chol=180, fbs=0, restecg=1,
               thalach=185, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2)
    sick = dict(age=65, sex=1, cp=0, trestbps=160, chol=300, fbs=1, restecg=0,
                thalach=110, exang=1, oldpeak=3.0, slope=1, ca=3, thal=3)
    for label, p in [("fit", fit), ("sick", sick)]:
        r = pipe.predict_proba(pd.DataFrame([p])[FEATURES])[0, 1]
        print(f"Sanity {label}: disease risk = {r:.1%}")


if __name__ == "__main__":
    # Run via the `train` module so the pickled pipeline references train.add_features
    # (not __main__.add_features), which lets app.py load it.
    import train
    train.main()
