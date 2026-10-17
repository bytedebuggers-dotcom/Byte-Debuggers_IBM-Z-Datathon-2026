"""
QuakeFix - Member 3: explainability charts for the four models (burst, fire, flood, damage).

Reads models/model.joblib (from train_model.py) and writes, into docs/:
    explain_burst.png, explain_fire.png, explain_flood.png, explain_damage.png
        SHAP beeswarm per model: which features push the prediction up or down
    shap_summary.png        copy of the burst chart (the headline picture for the pitch deck)
    feature_importance.png  one bar chart comparing what drives each model

Explanations use events held out by the same grouped split as training. (The saved models
were refit on all events, so read the charts as "what drives the model", not as a test score.)
If SHAP fails for a model, that model falls back to permutation importance.

Run from the repo root:
    python notebooks/explain_model.py
    python notebooks/explain_model.py --synthetic        # if features.csv is not available
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_model as tm  # same folder: reuse data loading, constants and the split seed

LABELS = {"burst": "Pipe burst / gas leak", "fire": "Fire risk", "flood": "Flood risk",
          "damage": "Structural damage score"}
COLORS = {"burst": "#0B2A6B", "fire": "#E36414", "flood": "#1FA37A", "damage": "#D9A21B"}


def inner_pipeline(pipe) -> Pipeline:
    """The damage model is wrapped to clip to 0..1; unwrap it to get the sklearn Pipeline."""
    return pipe.regressor_ if hasattr(pipe, "regressor_") else pipe


def positive_class_values(raw) -> np.ndarray:
    """Different SHAP explainers return different shapes; return (n_rows, n_features)."""
    if isinstance(raw, list):
        raw = raw[-1]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[:, :, -1]
    return raw


def shap_values_for(model, Xt: pd.DataFrame, predict) -> np.ndarray:
    import shap
    if isinstance(model, (LogisticRegression, Ridge)):
        return positive_class_values(shap.LinearExplainer(model, Xt).shap_values(Xt))
    try:  # random forests and other tree models
        return positive_class_values(shap.TreeExplainer(model).shap_values(Xt))
    except Exception:  # model-agnostic fallback (slower, fine for a few hundred rows)
        background = Xt.sample(min(80, len(Xt)), random_state=tm.SEED)
        ex = shap.Explainer(lambda d: predict(pd.DataFrame(d, columns=Xt.columns)), background)
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
    feats = art["features"]
    df = tm.load_data(argparse.Namespace(synthetic=args.synthetic, data=args.data))
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=tm.SEED)
                 .split(df, groups=df[tm.GROUP]))
    held = df.iloc[te]
    X = held[feats].sample(min(args.max_rows, len(held)), random_state=tm.SEED)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import shap
    shares = {}
    targets = {"burst": "burst_48h", "fire": "fire_48h", "flood": "flood_48h", "damage": "damage_score"}
    for key in ["burst", "fire", "flood", "damage"]:
        if key not in art:
            print(f"[{key}] not in model file, skipped")
            continue
        pipe = art[key]["pipeline"]
        inner = inner_pipeline(pipe)
        model = inner.steps[-1][1]
        Xt = pd.DataFrame(Pipeline(inner.steps[:-1]).transform(X), columns=feats, index=X.index)
        predict = (lambda d, m=model: m.predict_proba(d)[:, 1]) if hasattr(model, "predict_proba") \
            else (lambda d, m=model: m.predict(d))
        try:
            sv = shap_values_for(model, Xt, predict)
            plt.figure(figsize=(7.5, 4.6))
            shap.summary_plot(sv, X, feature_names=feats, show=False)
            plt.title(f"{LABELS[key]}: what pushes the prediction up or down", fontsize=11)
            plt.tight_layout()
            plt.savefig(out / f"explain_{key}.png", dpi=160, bbox_inches="tight")
            plt.close()
            imp, method = np.abs(sv).mean(axis=0), "SHAP"
        except Exception as e:
            print(f"[{key}] SHAP failed ({type(e).__name__}: {e}); using permutation importance")
            y = held[targets[key]].loc[X.index]
            scoring = "r2" if key == "damage" else "roc_auc"
            imp = np.clip(permutation_importance(pipe, X, y, n_repeats=5, random_state=tm.SEED,
                                                 scoring=scoring).importances_mean, 0, None)
            method = "permutation"
            plt.figure(figsize=(7.5, 4))
            plt.barh(feats, imp, color=COLORS[key])
            plt.title(f"{LABELS[key]}: feature importance (permutation)")
            plt.tight_layout()
            plt.savefig(out / f"explain_{key}.png", dpi=160, bbox_inches="tight")
            plt.close()
        shares[key] = imp / imp.sum() if imp.sum() > 0 else imp
        top = sorted(zip(feats, shares[key]), key=lambda t: -t[1])[:4]
        print(f"[{key}] ({method}) " + ", ".join(f"{f} {s:.0%}" for f, s in top))

    if (out / "explain_burst.png").exists():
        shutil.copyfile(out / "explain_burst.png", out / "shap_summary.png")

    # combined chart: share of total mean |SHAP| per feature, one bar per model
    fig, ax = plt.subplots(figsize=(8.5, 6))
    h = 0.8 / max(1, len(shares))
    for i, (key, share) in enumerate(shares.items()):
        ax.barh(np.arange(len(feats)) + i * h, share * 100, height=h * 0.92,
                color=COLORS[key], label=LABELS[key])
    ax.set_yticks(np.arange(len(feats)) + h * (len(shares) - 1) / 2)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("Share of the model's total feature influence (%)")
    ax.set_title("What drives each prediction (simulated data)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=160)
    print(f"[done] charts written to {out}/")


if __name__ == "__main__":
    main()
