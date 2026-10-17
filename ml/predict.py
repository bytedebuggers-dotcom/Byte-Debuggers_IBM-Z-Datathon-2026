"""QuakeFix - hazard prediction module (the ONLY file the backend needs to import).

    from ml.predict import predict, predict_batch

    predict("pipe_burst", {"pga": 0.4, "magnitude": 6.8, "pipe_age_yrs": 70,
                           "pressure_psi": 75, "dist_to_fault_km": 3})
    -> Prediction(target='pipe_burst', probability=0.99, flagged=True, risk_tier='high', ...)

Loads only JSON files (no pickle), so it is safe to move between Windows and the
LinuxONE server. Models are trained on SYNTHETIC data: decision support, not prediction of record.
Quick test:  python ml/predict.py
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- settings
VERSION = "v1"                 # change to "v2" after retraining with real SF data
TARGETS = ("pipe_burst", "gas_leak")

# Risk tiers. "flagged" means probability >= the model's own threshold (tuned for recall).
# "high" is a stricter cut-off for the most urgent directives. Tune this number.
HIGH_AT = 0.60

_here = Path(__file__).resolve().parent
MODELS_DIR = (_here.parent if _here.name == "ml" else _here) / "models"


# ---------------------------------------------------------------- schemas (Pydantic)
class PipeBurstInput(BaseModel):
    pga: float = Field(ge=0, le=3, description="Peak ground acceleration (g)")
    magnitude: float = Field(ge=0, le=10)
    pipe_age_yrs: float = Field(ge=0, le=200)
    pressure_psi: float = Field(ge=0, le=500)
    dist_to_fault_km: float = Field(ge=0, le=1000)


class GasLeakInput(BaseModel):
    pga: float = Field(ge=0, le=3, description="Peak ground acceleration (g)")
    magnitude: float = Field(ge=0, le=10)
    gas_line_age_yrs: float = Field(ge=0, le=200)
    gas_pressure_kpa: float = Field(ge=0, le=5000)
    dist_to_fault_km: float = Field(ge=0, le=1000)


INPUT_SCHEMAS = {"pipe_burst": PipeBurstInput, "gas_leak": GasLeakInput}


class Prediction(BaseModel):
    target: str
    probability: float
    flagged: bool
    risk_tier: str              # "high" | "medium" | "low"
    threshold: float
    model_version: str


# ---------------------------------------------------------------- loading
@lru_cache(maxsize=None)
def _load(target: str):
    """Load model + metadata once, then reuse (cached)."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}'. Choose from {TARGETS}")
    meta_path = MODELS_DIR / f"{target}_{VERSION}.meta.json"
    model_path = MODELS_DIR / f"{target}_{VERSION}.json"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path.name} or {meta_path.name} in {MODELS_DIR}. "
            "Run ml/train_hazard_models.py first.")
    meta = json.loads(meta_path.read_text())
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster, meta


# ---------------------------------------------------------------- prediction
def _tier(p: float, threshold: float) -> str:
    if p >= HIGH_AT:
        return "high"
    return "medium" if p >= threshold else "low"


def predict_batch(target: str, rows: List[Union[dict, BaseModel]]) -> List[Prediction]:
    """Score many rows at once (fast). Raises a pydantic ValidationError on bad input."""
    booster, meta = _load(target)
    schema = INPUT_SCHEMAS[target]
    clean = [schema.model_validate(r.model_dump() if isinstance(r, BaseModel) else r).model_dump()
             for r in rows]
    X = pd.DataFrame(clean)[meta["features"]]            # enforce training column order
    probs = booster.predict(xgb.DMatrix(X, feature_names=meta["features"]))
    return [Prediction(target=target, probability=round(float(p), 4),
                       flagged=bool(p >= meta["threshold"]),
                       risk_tier=_tier(float(p), meta["threshold"]),
                       threshold=round(meta["threshold"], 4),
                       model_version=meta["version"]) for p in probs]


def predict(target: str, row: Union[dict, BaseModel]) -> Prediction:
    """Score one row."""
    return predict_batch(target, [row])[0]


def predict_all(rows_by_target: Dict[str, dict]) -> Dict[str, Prediction]:
    """Score one asset for several hazards: {"pipe_burst": {...}, "gas_leak": {...}}."""
    return {t: predict(t, r) for t, r in rows_by_target.items()}


# ---------------------------------------------------------------- quick test
if __name__ == "__main__":
    risky = {"pga": 0.4, "magnitude": 6.8, "pipe_age_yrs": 70,
             "pressure_psi": 75, "dist_to_fault_km": 3}
    safe = {"pga": 0.02, "magnitude": 4.0, "pipe_age_yrs": 5,
            "pressure_psi": 55, "dist_to_fault_km": 60}
    print(predict("pipe_burst", risky))
    print(predict("pipe_burst", safe))
    print(predict("gas_leak", {"pga": 0.4, "magnitude": 6.8, "gas_line_age_yrs": 60,
                               "gas_pressure_kpa": 450, "dist_to_fault_km": 3}))
    try:
        predict("pipe_burst", {**risky, "magnitude": 99})
    except Exception as e:
        print("Bad input rejected:", type(e).__name__)
