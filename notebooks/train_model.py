"""
QuakeFix - Member 3: train the four hazard models (multi-hazard playbook version).

    burst   classifier  <- burst_48h      (pipe burst / gas leak)
    fire    classifier  <- fire_48h
    flood   classifier  <- flood_48h
    damage  regressor   <- damage_score   (structural damage, 0 to 1)

Design rules
------------
* Train/test split is GROUPED by event_id: no earthquake is in both sets.
* For every classifier and candidate model, the threshold is chosen on grouped out-of-fold
  predictions of the training events so recall >= --min-recall (default 0.90) at the best
  possible precision. Held-out events are only used to REPORT, never to tune or to pick models.
* Best classifier = highest out-of-fold precision at the recall target.
  Best damage model = lowest out-of-fold mean absolute error.
* scikit-learn only, so everything is safe to run on the s390x LinuxONE server. A tree model is
  only chosen over a linear one if it is >= --tree-margin (3%) better, because linear models
  pickle as plain arrays and load on any CPU / scikit-learn version.

Artifact (models/model.joblib), exactly as in docs/api_contract.md:
    {"features": [8 column names, in order],
     "burst":  {"pipeline": Pipeline (predict_proba), "threshold": float, "model_name": str},
     "fire":   {...same...},
     "flood":  {...same...},
     "damage": {"pipeline": regression pipeline, predict() returns 0..1, "model_name": str}}

Usage (repo root):
    python notebooks/train_model.py                 # data/processed/features.csv
    python notebooks/train_model.py --synthetic     # built-in copy of Member 2's generator
    python notebooks/train_model.py --min-recall 0.93
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_recall_curve, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
GROUP = "event_id"
FEATURES = ["magnitude", "dist_km", "shaking", "pipe_age", "sensor_var",
            "gas_main_dist_m", "elevation_m", "building_age"]
# artifact key -> (label column, display name)
CLASSIFIERS = {
    "burst": ("burst_48h", "Pipe burst / gas leak"),
    "fire": ("fire_48h", "Fire"),
    "flood": ("flood_48h", "Flood"),
}
DAMAGE_TARGET = "damage_score"
# Plain coefficient arrays: these unpickle safely on any CPU (including big-endian s390x) and
# across scikit-learn versions. Tree models are only chosen if they are clearly better.
PORTABLE = {"logistic_regression", "ridge"}
CLF_NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
             "hist_gradient_boosting": "Hist Gradient Boosting"}
REG_NAMES = {"ridge": "Ridge regression", "random_forest": "Random Forest",
             "hist_gradient_boosting": "Hist Gradient Boosting"}
EXAMPLE_INPUT = {"magnitude": 6.1, "dist_km": 12, "shaking": 5.2, "pipe_age": 40, "sensor_var": 0.6,
                 "gas_main_dist_m": 60, "elevation_m": 8, "building_age": 70}  # from the API contract


# --------------------------------------------------------------------------- data
def make_synthetic() -> pd.DataFrame:
    """Identical to Member 2's notebooks/make_synthetic.py (40 events x 60 segments, 4 targets).
    RULE 1 burst: shaking + pipe age.  RULE 2 fire: shaking + gas main nearby (+burst).
    RULE 3 flood: burst or shaking + low ground.  RULE 4 damage: shaking + building age."""
    rng = np.random.default_rng(42)
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    rows = []
    for event_id in range(40):
        mag = rng.uniform(4.5, 7.5)
        for seg in range(60):
            dist_km = rng.uniform(1, 80)
            shaking = max(0.0, mag * 1.5 - 1.2 * np.log1p(dist_km) + rng.normal(0, 0.4))
            pipe_age = int(rng.integers(5, 60))
            sensor_var = rng.uniform(0, 1)
            gas_main_dist_m = rng.uniform(5, 500)
            elevation_m = rng.uniform(0, 60)
            building_age = int(rng.integers(5, 100))
            burst = int(rng.random() < sigmoid(shaking - 4.5 + 0.03 * pipe_age + sensor_var - 1.5))
            fire = int(rng.random() < sigmoid(-4.0 + 0.6 * shaking + 1.5 * burst
                                              - 0.008 * gas_main_dist_m + 0.01 * building_age))
            flood = int(rng.random() < sigmoid(-3.0 + 0.4 * shaking + 1.5 * burst - 0.07 * elevation_m))
            damage = float(np.clip(0.10 * shaking + 0.004 * building_age + rng.normal(0, 0.05), 0, 1))
            rows.append(dict(event_id=event_id, segment_id=seg, magnitude=mag, dist_km=dist_km,
                             shaking=shaking, pipe_age=pipe_age, sensor_var=sensor_var,
                             gas_main_dist_m=gas_main_dist_m, elevation_m=elevation_m,
                             building_age=building_age, burst_48h=burst, fire_48h=fire,
                             flood_48h=flood, damage_score=damage, is_synthetic=True))
    return pd.DataFrame(rows)


def load_data(args) -> pd.DataFrame:
    if args.synthetic:
        print("[data] using built-in SYNTHETIC data (same generator as Member 2's file)")
        return make_synthetic()
    path = Path(args.data)
    if not path.exists():
        sys.exit(f"[data] {path} not found. git pull, or run with --synthetic.")
    print(f"[data] loading {path}")
    return pd.read_csv(path)


# --------------------------------------------------------------------------- models
def build_classifier(kind: str) -> Pipeline:
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


def build_regressor(kind: str):
    """Regression pipeline whose predict() is always clipped to the 0..1 range the API promises."""
    impute = ("impute", SimpleImputer(strategy="median"))
    if kind == "ridge":
        inner = Pipeline([impute, ("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    elif kind == "random_forest":
        inner = Pipeline([impute, ("reg", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED))])
    elif kind == "hist_gradient_boosting":
        inner = Pipeline([impute, ("reg", HistGradientBoostingRegressor(
            max_depth=5, learning_rate=0.06, random_state=SEED))])
    else:
        raise ValueError(kind)
    return TransformedTargetRegressor(regressor=inner, func=np.asarray,
                                      inverse_func=partial(np.clip, a_min=0.0, a_max=1.0),
                                      check_inverse=False)


def pick_threshold(y_true, proba, min_recall: float) -> float:
    """Highest-precision threshold whose recall is still >= min_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    ok = np.where(recall >= min_recall)[0]
    if len(ok) == 0:
        print("    [threshold] WARNING: recall target unreachable; using lowest threshold")
        return float(thresholds.min())
    return float(thresholds[ok[np.argmax(precision[ok])]])


def choose(results, score, higher_is_better: bool, margin: float):
    """Pick the best model by score(result). A tree model must beat the best portable (linear)
    model by `margin` (relative) to be chosen, because tree pickles can fail to load on the
    big-endian s390x server or under a different scikit-learn version. margin=0 disables this."""
    sign = 1 if higher_is_better else -1
    best = max(results, key=lambda r: sign * score(r))
    portable = [r for r in results if r["kind"] in PORTABLE]
    if best["kind"] in PORTABLE or not portable:
        return best
    best_p = max(portable, key=lambda r: sign * score(r))
    gain = sign * (score(best) - score(best_p)) / max(abs(score(best_p)), 1e-12)
    if gain < margin:
        print(f"  (keeping {best_p['kind']}: {best['kind']} is only {gain:.1%} better, below the "
              f"{margin:.0%} needed to justify a model that may not load on s390x)")
        return best_p
    return best


def clf_metrics(y, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"recall": float(recall_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "positive_rate": float(np.mean(y))}


def fmt_clf(m: dict) -> str:
    return (f"recall={m['recall']:.3f} precision={m['precision']:.3f} F1={m['f1']:.3f} "
            f"AUC={m['roc_auc']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")


def evaluate_classifier(kind, X_tr, y_tr, g_tr, X_te, y_te, min_recall) -> dict:
    oof = np.zeros(len(X_tr))
    for a, b in GroupKFold(n_splits=min(5, g_tr.nunique())).split(X_tr, y_tr, g_tr):
        oof[b] = build_classifier(kind).fit(X_tr.iloc[a], y_tr.iloc[a]).predict_proba(X_tr.iloc[b])[:, 1]
    thr = pick_threshold(y_tr.to_numpy(), oof, min_recall)
    cv = clf_metrics(y_tr.to_numpy(), oof, thr)
    test = clf_metrics(y_te.to_numpy(), build_classifier(kind).fit(X_tr, y_tr).predict_proba(X_te)[:, 1], thr)
    print(f"  [{kind}] thr={thr:.3f}\n     cv-oof: {fmt_clf(cv)}\n     test  : {fmt_clf(test)}")
    return {"kind": kind, "threshold": thr, "cv": cv, "test": test}


def evaluate_regressor(kind, X_tr, y_tr, g_tr, X_te, y_te) -> dict:
    oof = np.zeros(len(X_tr))
    for a, b in GroupKFold(n_splits=min(5, g_tr.nunique())).split(X_tr, y_tr, g_tr):
        oof[b] = build_regressor(kind).fit(X_tr.iloc[a], y_tr.iloc[a]).predict(X_tr.iloc[b])
    pred = build_regressor(kind).fit(X_tr, y_tr).predict(X_te)
    cv = {"mae": float(mean_absolute_error(y_tr, oof)), "r2": float(r2_score(y_tr, oof))}
    test = {"mae": float(mean_absolute_error(y_te, pred)), "r2": float(r2_score(y_te, pred))}
    print(f"  [{kind}] cv-oof: MAE={cv['mae']:.3f} R2={cv['r2']:.3f} | test: MAE={test['mae']:.3f} R2={test['r2']:.3f}")
    return {"kind": kind, "cv": cv, "test": test}


def write_results_md(path: Path, clf_results: dict, dmg, args, n_events: int, synthetic: bool) -> None:
    L = ["# Model results", "",
         f"Data: {'SYNTHETIC (built-in copy of the shared generator; hazard labels come from simple rules we wrote)' if synthetic else args.data}; "
         f"{n_events} earthquake events; split by event_id (no event in both train and test).",
         f"Classifiers: recall target >= {args.min_recall:.0%}. Thresholds are chosen on grouped out-of-fold "
         "predictions of the training events, then reported once on held-out events.", ""]
    for key, (results, best) in clf_results.items():
        L += [f"## {CLASSIFIERS[key][1]} (`{CLASSIFIERS[key][0]}`)", "",
              "| Model | Threshold | Test recall | Test precision | Test F1 | Test AUC | CV recall | CV precision |",
              "|---|---|---|---|---|---|---|---|"]
        for r in results:
            t, c = r["test"], r["cv"]
            star = " **(chosen)**" if r["kind"] == best else ""
            L.append(f"| {CLF_NAMES[r['kind']]}{star} | {r['threshold']:.3f} | {t['recall']:.3f} | {t['precision']:.3f} | "
                     f"{t['f1']:.3f} | {t['roc_auc']:.3f} | {c['recall']:.3f} | {c['precision']:.3f} |")
        L.append("")
    if dmg:
        results, best = dmg
        L += [f"## Structural damage score (`{DAMAGE_TARGET}`, 0 to 1)", "",
              "| Model | Test MAE | Test R2 | CV MAE | CV R2 |", "|---|---|---|---|---|"]
        for r in results:
            star = " **(chosen)**" if r["kind"] == best else ""
            L.append(f"| {REG_NAMES[r['kind']]}{star} | {r['test']['mae']:.3f} | {r['test']['r2']:.3f} | "
                     f"{r['cv']['mae']:.3f} | {r['cv']['r2']:.3f} |")
        L.append("")
    L += ["## Limitations",
          "1. Trained on simulated data, not validated on real earthquakes, fires or floods.",
          "2. Held-out recall varies from earthquake to earthquake, so 90% is a target, not a guarantee.",
          "3. The threshold trades precision for recall: more false alarms, fewer missed hazards.",
          "4. Evacuation routing cannot see live crowds or traffic.",
          "5. Gas main distance, elevation and building age are partly simulated, and the fire, flood and damage labels come from rules we wrote.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/features.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--min-recall", type=float, default=0.90)
    ap.add_argument("--tree-margin", type=float, default=0.03,
                    help="a tree model must beat the best linear model by this fraction to be chosen "
                         "(keeps the saved model portable to s390x); 0 = pick purely by score")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--out", default="models/model.joblib")
    ap.add_argument("--results-md", default="docs/model_results.md")
    args = ap.parse_args()

    df = load_data(args)
    need = FEATURES + [GROUP, CLASSIFIERS["burst"][0]]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"[data] missing required columns: {miss}. Found: {list(df.columns)}")
    X, groups = df[FEATURES], df[GROUP]
    print(f"[data] rows={len(df)} events={groups.nunique()}")

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                    random_state=SEED).split(X, groups=groups))
    g_tr = groups.iloc[tr]
    assert set(g_tr).isdisjoint(set(groups.iloc[te])), "event leakage between train and test!"
    print(f"[split] train events={g_tr.nunique()} test events={groups.iloc[te].nunique()}")

    artifact = {"features": list(FEATURES)}
    clf_results, summary = {}, {}
    for key, (tcol, label) in CLASSIFIERS.items():
        print(f"\n=== {label} ({tcol}) ===")
        if tcol not in df.columns:
            print(f"  SKIPPED: column '{tcol}' not in data (labels are never invented).")
            continue
        y = df[tcol].astype(int)
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2 or y.sum() < 30:
            print(f"  SKIPPED: not enough positive/negative examples ({int(y.sum())} positives).")
            continue
        print(f"  positive_rate={y.mean():.3f}")
        results = [evaluate_classifier(k, X.iloc[tr], y.iloc[tr], g_tr, X.iloc[te], y.iloc[te], args.min_recall)
                   for k in CLF_NAMES]
        best = choose(results, lambda r: r["cv"]["precision"], True, args.tree_margin)
        print(f"  -> best: {best['kind']} (thr {best['threshold']:.3f}); held-out recall {best['test']['recall']:.3f}")
        artifact[key] = {"pipeline": build_classifier(best["kind"]).fit(X, y),
                         "threshold": float(best["threshold"]), "model_name": best["kind"]}
        clf_results[key] = (results, best["kind"])
        summary[key] = {"chosen": best["kind"], "threshold": best["threshold"], "results": results}
    if "burst" not in artifact:
        sys.exit("[train] the burst model could not be trained; nothing to save.")

    dmg = None
    print(f"\n=== Structural damage ({DAMAGE_TARGET}, regression) ===")
    if DAMAGE_TARGET in df.columns:
        yd = df[DAMAGE_TARGET].astype(float)
        dres = [evaluate_regressor(k, X.iloc[tr], yd.iloc[tr], g_tr, X.iloc[te], yd.iloc[te]) for k in REG_NAMES]
        dbest = choose(dres, lambda r: r["cv"]["mae"], False, args.tree_margin)
        print(f"  -> best: {dbest['kind']} (test MAE {dbest['test']['mae']:.3f}, R2 {dbest['test']['r2']:.3f})")
        artifact["damage"] = {"pipeline": build_regressor(dbest["kind"]).fit(X, yd), "model_name": dbest["kind"]}
        dmg = (dres, dbest["kind"])
        summary["damage"] = {"chosen": dbest["kind"], "results": dres}
    else:
        print(f"  SKIPPED: column '{DAMAGE_TARGET}' not in data.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)
    meta = {"min_recall_target": args.min_recall, "features": FEATURES, "sklearn_version": sklearn.__version__,
            "n_events": int(groups.nunique()), "synthetic": bool(args.synthetic), "models": summary}
    (out.parent / "metrics.json").write_text(json.dumps(meta, indent=2))
    write_results_md(Path(args.results_md), clf_results, dmg, args, int(groups.nunique()), args.synthetic)

    # round-trip check that mimics Member 1's /predict endpoint
    ART = joblib.load(out)
    row = pd.DataFrame([EXAMPLE_INPUT])[ART["features"]]
    print(f"\n[save] {out} OK. /predict-style check on the contract's example input:")
    for h in ["burst", "fire", "flood"]:
        if h in ART:
            p = float(ART[h]["pipeline"].predict_proba(row)[0, 1])
            print(f"       {h}: risk={p:.3f} thr={ART[h]['threshold']:.3f} flag={p >= ART[h]['threshold']}")
    if "damage" in ART:
        d = float(ART["damage"]["pipeline"].predict(row)[0])
        assert 0.0 <= d <= 1.0
        print(f"       damage: score={d:.3f}")
    print(f"[save] sklearn {sklearn.__version__} (use the same version on the LinuxONE server)")


if __name__ == "__main__":
    main()
