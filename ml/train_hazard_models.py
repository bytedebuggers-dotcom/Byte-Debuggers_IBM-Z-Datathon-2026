"""QuakeFix - train hazard models (pipe burst + gas leak) on SYNTHETIC data.

Run from the repo:  python ml/train_hazard_models.py
Saves to models/:   <target>_v1.json       (portable XGBoost model)
                    <target>_v1.meta.json  (features, threshold, version - read by ml/predict.py)
                    <target>_v1.pkl        (backup bundle)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42
TARGET_RECALL = 0.85          # catch at least 85% of real events
VERSION = "v1"

_here = Path(__file__).resolve().parent
MODELS_DIR = (_here.parent if _here.name == "ml" else _here) / "models"

# Which columns each model is allowed to use
FEATURES = {
    "pipe_burst": ["pga", "magnitude", "pipe_age_yrs", "pressure_psi", "dist_to_fault_km"],
    "gas_leak":   ["pga", "magnitude", "gas_line_age_yrs", "gas_pressure_kpa", "dist_to_fault_km"],
}
BASE_RATE = {"pipe_burst": 0.04, "gas_leak": 0.03}   # share of rows that are events


def _z(s):
    return (s - s.mean()) / s.std()


def _labels(rng, score, rate):
    """Turn a risk score into 0/1 labels with an average event rate close to `rate`."""
    lo, hi = -20.0, 20.0
    for _ in range(60):                       # bisection: find the shift that hits the rate
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(score + mid)))
        lo, hi = (mid, hi) if p.mean() < rate else (lo, mid)
    p = 1 / (1 + np.exp(-(score + mid)))
    return (rng.random(len(score)) < p).astype(int)


def make_synthetic_data(n=20_000, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "pga": rng.gamma(2.0, 0.05, n),                  # peak ground acceleration (g)
        "magnitude": rng.uniform(3.0, 7.5, n),
        "dist_to_fault_km": rng.exponential(15, n),
        "pipe_age_yrs": rng.integers(1, 80, n),
        "pressure_psi": rng.normal(60, 12, n),
        "gas_line_age_yrs": rng.integers(1, 70, n),
        "gas_pressure_kpa": rng.normal(400, 60, n),
    })
    pipe_score = (1.4 * _z(df.pga) + 0.9 * _z(df.pipe_age_yrs) + 0.5 * _z(df.magnitude)
                  + 0.4 * _z(df.pressure_psi) - 0.6 * _z(df.dist_to_fault_km))
    gas_score = (1.3 * _z(df.pga) + 0.8 * _z(df.gas_line_age_yrs) + 0.5 * _z(df.magnitude)
                 + 0.4 * _z(df.gas_pressure_kpa) - 0.6 * _z(df.dist_to_fault_km))
    df["pipe_burst"] = _labels(rng, pipe_score, BASE_RATE["pipe_burst"])
    df["gas_leak"] = _labels(rng, gas_score, BASE_RATE["gas_leak"])
    return df


def train_target(df, target):
    feats = FEATURES[target]
    X, y = df[feats], df[target]

    # 60% train / 20% validation (pick threshold) / 20% test (final honest score)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="aucpr", early_stopping_rounds=30,
        random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Threshold chosen on VALIDATION data only
    val_proba = model.predict_proba(X_val)[:, 1]
    _, recall, thr = precision_recall_curve(y_val, val_proba)
    ok = np.where(recall[:-1] >= TARGET_RECALL)[0]
    threshold = float(thr[ok[-1]]) if len(ok) else float(thr[0])

    # Final score on TEST data the model never saw
    test_proba = model.predict_proba(X_test)[:, 1]
    pred = (test_proba >= threshold).astype(int)
    pr_auc = average_precision_score(y_test, test_proba)
    print(f"\n===== {target} =====")
    print(f"event rate {y.mean():.3f} | PR-AUC {pr_auc:.3f} (random guessing = {y_test.mean():.3f})")
    print(f"threshold {threshold:.3f} (chosen on validation data)")
    print(classification_report(y_test, pred, digits=3))
    print(confusion_matrix(y_test, pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"target": target, "model": model, "features": feats,
                 "threshold": threshold, "version": VERSION,
                 "note": "trained on synthetic data"},
                MODELS_DIR / f"{target}_{VERSION}.pkl")
    model.save_model(MODELS_DIR / f"{target}_{VERSION}.json")   # portable copy

    # Small JSON file the backend reads: no pickle needed on the server
    meta = {"target": target, "version": VERSION, "features": feats,
            "threshold": threshold, "trained_on": "synthetic data",
            "test_recall": round(float(recall_score(y_test, pred)), 3),
            "test_precision": round(float(precision_score(y_test, pred)), 3),
            "test_pr_auc": round(float(pr_auc), 3)}
    (MODELS_DIR / f"{target}_{VERSION}.meta.json").write_text(json.dumps(meta, indent=2))
    return model


def score_one(target, row: dict):
    """How the backend will use a saved model: returns (probability, flagged?)."""
    bundle = joblib.load(MODELS_DIR / f"{target}_{VERSION}.pkl")
    X = pd.DataFrame([row])[bundle["features"]]          # enforce column order
    p = float(bundle["model"].predict_proba(X)[:, 1][0])
    return p, p >= bundle["threshold"]


if __name__ == "__main__":
    data = make_synthetic_data()
    for t in FEATURES:
        train_target(data, t)

    print("\nDemo - one pipe near the fault, old, high shaking:")
    print(score_one("pipe_burst", {"pga": 0.4, "magnitude": 6.8, "pipe_age_yrs": 70,
                                   "pressure_psi": 75, "dist_to_fault_km": 3}))
