"""
QuakeFix - Member 3: explainability charts for every hazard model.

Reads models/model.joblib (from train_model.py) and writes, into docs/:
    shap_summary.png        SHAP beeswarm for the pipeline-burst model (the headline chart)
    shap_<hazard>.png       one SHAP beeswarm per hazard
    feature_importance.png  one bar chart comparing what drives each hazard (share of mean |SHAP|)

Explanations use events held out by the same grouped split as training. (The saved model was
refit on all events, so treat the charts as "what drives the model", not as a test score.)
If SHAP fails for a model type it falls back to permutation importance for that hazard.

Run from the repo root:
    python notebooks/explain_model.py
    python notebooks/explain_model.py --synthetic        # if features.csv is not available
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_model as tm  # same folder: reuse data loading, constants and the split seed

TEAM_COLORS = {"pipeline_burst": "#0B2A6B", "fire_risk": "#E36414",
               "flood_risk": "#1FA37A", "infrastructure": "#D9A21B"}


def positive_class_values(raw) -> np.ndarray:
    """Different SHAP explainers return different shapes; return (n_rows, n_features)."""
    if isinstance(raw, list):
        raw = raw[-1]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[:, :, -1]
    return raw


def shap_values_for(clf, Xt: pd.DataFrame) -> np.ndarray:
    import shap
    if isinstance(clf, LogisticRegression):
        return positive_class_values(shap.LinearExplainer(clf, Xt).shap_values(Xt))
    if isinstance(clf, RandomForestClassifier):
        return positive_class_values(shap.TreeExplainer(clf).shap_values(Xt))
    try:  # other tree models
        return positive_class_values(shap.TreeExplainer(clf).shap_values(Xt))
    except Exception:  # model-agnostic fallback (slower, fine for a few hundred rows)
        background = Xt.sample(min(80, len(Xt)), random_state=tm.SEED)
        ex = shap.Explainer(lambda d: clf.predict_proba(pd.DataFrame(d, columns=Xt.columns))[:, 1], background)
        return np.asarray(ex(Xt).values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--data", default="data/processed/features.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--max-rows", type=int, default=600)
    args = ap.parse_args()

    art = joblib.load(args.model)
    if "pipelines" not in art:  # old single-model file
        art = {"pipelines": {"pipeline_burst": art["pipeline"]}, "hazard_features": {"pipeline_burst": art["features"]}}
    df = tm.load_data(argparse.Namespace(synthetic=args.synthetic, data=args.data))
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=tm.SEED)
                 .split(df, groups=df[tm.GROUP]))
    held = df.iloc[te]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import shap
    shares = {}
    for hz, pipe in art["pipelines"].items():
        feats = art["hazard_features"][hz]
        label = tm.HAZARDS[hz]["label"]
        X = held[feats].sample(min(args.max_rows, len(held)), random_state=tm.SEED)
        pre = Pipeline(pipe.steps[:-1])
        clf = pipe.steps[-1][1]
        Xt = pd.DataFrame(pre.transform(X), columns=feats, index=X.index)
        try:
            sv = shap_values_for(clf, Xt)
            plt.figure(figsize=(7.5, 4.2))
            shap.summary_plot(sv, X, feature_names=feats, show=False)
            plt.title(f"{label}: what pushes the risk up or down", fontsize=11)
            plt.tight_layout()
            for name in ([f"shap_{hz}.png"] + (["shap_summary.png"] if hz == "pipeline_burst" else [])):
                plt.savefig(out / name, dpi=160, bbox_inches="tight")
            plt.close()
            imp = np.abs(sv).mean(axis=0)
            method = "SHAP"
        except Exception as e:
            print(f"[{hz}] SHAP failed ({type(e).__name__}: {e}); using permutation importance")
            y = held[tm.HAZARDS[hz]["target"]].loc[X.index]
            imp = permutation_importance(pipe, X, y, n_repeats=5, random_state=tm.SEED, scoring="roc_auc").importances_mean
            imp = np.clip(imp, 0, None)
            method = "permutation"
        shares[hz] = imp / imp.sum() if imp.sum() > 0 else imp
        top = sorted(zip(feats, shares[hz]), key=lambda t: -t[1])
        print(f"[{hz}] ({method}) " + ", ".join(f"{f} {s:.0%}" for f, s in top))

    # combined chart: share of total mean |SHAP| per feature, one bar per hazard
    feats = tm.FEATURES
    fig, ax = plt.subplots(figsize=(8, 4.6))
    h = 0.8 / max(1, len(shares))
    for i, (hz, share) in enumerate(shares.items()):
        ax.barh(np.arange(len(feats)) + i * h, share * 100, height=h * 0.92,
                color=TEAM_COLORS.get(hz, "#888"), label=tm.HAZARDS[hz]["label"])
    ax.set_yticks(np.arange(len(feats)) + h * (len(shares) - 1) / 2)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("Share of model's total feature influence (%)")
    ax.set_title("What drives each hazard prediction (simulated data)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=160)
    print(f"[done] charts written to {out}/")


if __name__ == "__main__":
    main()
