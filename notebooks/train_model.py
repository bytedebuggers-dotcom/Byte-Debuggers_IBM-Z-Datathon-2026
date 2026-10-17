"""
QuakeFix - Member 3: multi-hazard classifier training.

Hazards (one binary model each, same event-grouped split for all):
    pipeline_burst  <- burst_48h
    fire_risk       <- fire_48h
    flood_risk      <- flood_48h
    infrastructure  <- infra_damage

Design rules
------------
* Train/test split is GROUPED by event_id: no earthquake is in both sets.
* Per hazard and per candidate model, the threshold is chosen on grouped
  out-of-fold predictions of the training events so recall >= --min-recall
  (default 0.90) at the best possible precision. Held-out events are used only
  to REPORT, never to tune or choose the model.
* Best model per hazard = highest out-of-fold precision at the recall target.
* A hazard whose label column is missing from the CSV is SKIPPED with a warning
  (labels are never invented for real data). pipeline_burst is required.

Artifact (models/model.joblib), a dict:
    "pipelines":  {hazard: sklearn Pipeline}        # new multi-hazard keys
    "thresholds": {hazard: float}
    "features":   [...shared feature list...]
    "hazard_features": {hazard: [...]}              # same as "features" unless changed
    "pipeline", "threshold": the burst model/threshold. These LEGACY keys keep
                 Member 1's current API (ART["pipeline"], ART["threshold"]) working.

Usage (repo root):
    python notebooks/train_model.py                # data/processed/features.csv
    python notebooks/train_model.py --synthetic    # built-in data incl. all 4 hazards
    python notebooks/train_model.py --min-recall 0.93
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
GROUP = "event_id"
# Shared by the API contract (Member 1's SegmentFeatures). Changing this list means
# changing docs/api_contract.md and api/main.py too.
FEATURES = ["magnitude", "dist_km", "shaking", "pipe_age", "sensor_var"]
HAZARDS = {
    "pipeline_burst": {"target": "burst_48h", "label": "Pipeline burst", "features": FEATURES},
    "fire_risk": {"target": "fire_48h", "label": "Fire risk", "features": FEATURES},
    "flood_risk": {"target": "flood_48h", "label": "Flood risk", "features": FEATURES},
    "infrastructure": {"target": "infra_damage", "label": "Infrastructure damage", "features": FEATURES},
}
MODELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist Gradient Boosting",
}


# --------------------------------------------------------------------------- data
def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_synthetic() -> pd.DataFrame:
    """Burst part = Member 2's generator (40 events x 60 segments, same schema).
    Fire / flood / infrastructure labels are SIMULATED by the documented rules below,
    driven partly by hidden variables (gas-line distance, elevation, building age) that
    are NOT model features, so the models only see part of the signal.
      fire:   logit = 1.0*(shaking-5.5) + 2.0*exp(-gas_dist_m/150) - 1.5
      flood:  logit = 2.0*burst + 1.5*(elevation_m<10) - 3.0
      infra:  logit = 0.9*(shaking-5.0) + 0.04*(building_age-40) - 0.8
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for event_id in range(40):
        mag = rng.uniform(4.5, 7.5)
        for seg in range(60):
            dist_km = rng.uniform(1, 80)
            shaking = max(0.0, mag * 1.5 - 1.2 * np.log1p(dist_km) + rng.normal(0, 0.4))
            pipe_age = int(rng.integers(5, 60))
            sensor_var = rng.uniform(0, 1)
            p = _sigmoid(shaking - 4.5 + 0.03 * pipe_age + sensor_var - 1.5)
            rows.append(dict(event_id=event_id, segment_id=seg, magnitude=mag, dist_km=dist_km,
                             shaking=shaking, pipe_age=pipe_age, sensor_var=sensor_var,
                             burst_48h=int(rng.random() < p), is_synthetic=True))
    df = pd.DataFrame(rows)
    r2 = np.random.default_rng(SEED + 1)
    n = len(df)
    gas_dist = r2.uniform(10, 600, n)
    elevation = r2.uniform(0, 60, n)
    bldg_age = r2.integers(5, 90, n)
    sh = df["shaking"].to_numpy()
    df["fire_48h"] = (r2.random(n) < _sigmoid(1.0 * (sh - 5.5) + 2.0 * np.exp(-gas_dist / 150) - 1.5)).astype(int)
    df["flood_48h"] = (r2.random(n) < _sigmoid(2.0 * df["burst_48h"] + 1.5 * (elevation < 10) - 3.0)).astype(int)
    df["infra_damage"] = (r2.random(n) < _sigmoid(0.9 * (sh - 5.0) + 0.04 * (bldg_age - 40) - 0.8)).astype(int)
    return df


def load_data(args) -> pd.DataFrame:
    if args.synthetic:
        print("[data] using built-in SYNTHETIC data (all four hazards simulated)")
        return make_synthetic()
    path = Path(args.data)
    if not path.exists():
        sys.exit(f"[data] {path} not found. git pull, or run with --synthetic.")
    print(f"[data] loading {path}")
    return pd.read_csv(path)


# --------------------------------------------------------------------------- model
def build_pipeline(kind: str) -> Pipeline:
    impute = ("impute", SimpleImputer(strategy="median"))
    if kind == "logistic_regression":
        return Pipeline([impute, ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    if kind == "random_forest":
        return Pipeline([impute, ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED))])
    if kind == "hist_gradient_boosting":
        return Pipeline([impute, ("clf", HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.06, class_weight="balanced", random_state=SEED))])
    raise ValueError(kind)


def pick_threshold(y_true, proba, min_recall: float) -> float:
    """Highest-precision threshold whose recall is still >= min_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    ok = np.where(recall >= min_recall)[0]
    if len(ok) == 0:
        print("    [threshold] WARNING: recall target unreachable; using lowest threshold")
        return float(thresholds.min())
    return float(thresholds[ok[np.argmax(precision[ok])]])


def metrics(y, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "positive_rate": float(np.mean(y)),
    }


def fmt(m: dict) -> str:
    return (f"recall={m['recall']:.3f} precision={m['precision']:.3f} F1={m['f1']:.3f} "
            f"AUC={m['roc_auc']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")


def evaluate(kind, X_tr, y_tr, g_tr, X_te, y_te, min_recall) -> dict:
    oof = np.zeros(len(X_tr))
    for a, b in GroupKFold(n_splits=min(5, g_tr.nunique())).split(X_tr, y_tr, g_tr):
        oof[b] = build_pipeline(kind).fit(X_tr.iloc[a], y_tr.iloc[a]).predict_proba(X_tr.iloc[b])[:, 1]
    thr = pick_threshold(y_tr.to_numpy(), oof, min_recall)
    cv = metrics(y_tr.to_numpy(), oof, thr)
    fitted = build_pipeline(kind).fit(X_tr, y_tr)
    test = metrics(y_te.to_numpy(), fitted.predict_proba(X_te)[:, 1], thr)
    print(f"  [{kind}] thr={thr:.3f}\n     cv-oof: {fmt(cv)}\n     test  : {fmt(test)}")
    return {"kind": kind, "threshold": thr, "cv": cv, "test": test}


def write_results_md(path: Path, all_results: dict, args, n_events: int, synthetic: bool) -> None:
    L = ["# Model results", "",
         f"Data: {'SYNTHETIC (built-in generator; fire/flood/infrastructure labels are simulated)' if synthetic else args.data}; "
         f"{n_events} earthquake events; split by event_id (no event in both train and test).",
         f"Recall target >= {args.min_recall:.0%}. Thresholds are chosen on grouped out-of-fold predictions "
         "of the training events and reported once on held-out events.", ""]
    for hz, (results, best) in all_results.items():
        L += [f"## {HAZARDS[hz]['label']} (`{HAZARDS[hz]['target']}`)", "",
              "| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |",
              "|---|---|---|---|---|---|---|---|"]
        for r in results:
            t, c = r["test"], r["cv"]
            star = " **(chosen)**" if r["kind"] == best else ""
            L.append(f"| {MODELS[r['kind']]}{star} | {r['threshold']:.3f} | {t['recall']:.3f} | {t['precision']:.3f} | "
                     f"{t['f1']:.3f} | {t['roc_auc']:.3f} | {c['recall']:.3f} | {c['precision']:.3f} |")
        L.append("")
    L += ["## Limitations",
          "1. Trained on simulated data; not validated on real failures. Fire, flood and infrastructure labels come from simple documented rules, so those scores show the pipeline works, not real-world accuracy.",
          "2. Held-out recall varies from earthquake to earthquake, so 90% is a target, not a guarantee.",
          "3. The threshold trades precision for recall: more false alarms, fewer missed hazards.",
          "4. The models use a small shared feature set (magnitude, distance, shaking, pipe age, sensor variance).",
          "5. Advisory output only; a commander approves every action.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/features.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--min-recall", type=float, default=0.90)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--out", default="models/model.joblib")
    ap.add_argument("--results-md", default="docs/model_results.md")
    args = ap.parse_args()

    df = load_data(args)
    need = FEATURES + [GROUP, HAZARDS["pipeline_burst"]["target"]]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"[data] missing required columns: {miss}. Found: {list(df.columns)}")
    groups = df[GROUP]
    print(f"[data] rows={len(df)} events={groups.nunique()}")

    # one event split shared by every hazard
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                    random_state=SEED).split(df, groups=groups))
    g_tr = groups.iloc[tr]
    assert set(g_tr).isdisjoint(set(groups.iloc[te])), "event leakage between train and test!"
    print(f"[split] train events={g_tr.nunique()} test events={groups.iloc[te].nunique()}")

    pipelines, thresholds, hazard_features, all_results = {}, {}, {}, {}
    for hz, cfg in HAZARDS.items():
        tcol, feats = cfg["target"], cfg["features"]
        print(f"\n=== {cfg['label']} ({tcol}) ===")
        if tcol not in df.columns:
            print(f"  SKIPPED: column '{tcol}' not in data (ask Member 2 to add it; labels are not invented).")
            continue
        y = df[tcol].astype(int)
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2 or y.sum() < 30:
            print(f"  SKIPPED: not enough positive/negative examples ({int(y.sum())} positives).")
            continue
        print(f"  positive_rate={y.mean():.3f}")
        X = df[feats]
        results = [evaluate(k, X.iloc[tr], y.iloc[tr], g_tr, X.iloc[te], y.iloc[te], args.min_recall)
                   for k in MODELS]
        best = max(results, key=lambda r: r["cv"]["precision"])
        print(f"  -> best: {best['kind']} (thr {best['threshold']:.3f}); "
              f"held-out recall {best['test']['recall']:.3f}")
        pipelines[hz] = build_pipeline(best["kind"]).fit(X, y)  # final refit on all events
        thresholds[hz] = float(best["threshold"])
        hazard_features[hz] = list(feats)
        all_results[hz] = (results, best["kind"])

    burst = "pipeline_burst"
    if burst not in pipelines:
        sys.exit("[train] pipeline_burst could not be trained; nothing to save.")
    artifact = {
        "pipelines": pipelines, "thresholds": thresholds,
        "features": list(FEATURES), "hazard_features": hazard_features,
        "targets": {h: HAZARDS[h]["target"] for h in pipelines},
        # legacy contract keys (current api/main.py reads these)
        "pipeline": pipelines[burst], "threshold": thresholds[burst],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)

    meta = {"min_recall_target": args.min_recall, "features": FEATURES, "thresholds": thresholds,
            "sklearn_version": sklearn.__version__, "n_events": int(groups.nunique()),
            "synthetic": bool(args.synthetic),
            "hazards": {h: {"chosen": b, "results": r} for h, (r, b) in all_results.items()}}
    (out.parent / "metrics.json").write_text(json.dumps(meta, indent=2))
    write_results_md(Path(args.results_md), all_results, args, int(groups.nunique()), args.synthetic)

    # round-trip check with the API's example input, for both new and legacy keys
    loaded = joblib.load(out)
    ex = pd.DataFrame([{"magnitude": 6.1, "dist_km": 12, "shaking": 5.2, "pipe_age": 40, "sensor_var": 0.6}])
    legacy = float(loaded["pipeline"].predict_proba(ex[loaded["features"]])[0, 1])
    print(f"\n[save] {out} OK; legacy burst risk for example input = {legacy:.3f} "
          f"(flag={legacy >= loaded['threshold']})")
    for h, p in loaded["pipelines"].items():
        r = float(p.predict_proba(ex[loaded["hazard_features"][h]])[0, 1])
        print(f"       {h}: risk={r:.3f} thr={loaded['thresholds'][h]:.3f} flag={r >= loaded['thresholds'][h]}")
    print(f"[save] sklearn {sklearn.__version__} (pin the same version on the LinuxONE server)")


if __name__ == "__main__":
    main()
