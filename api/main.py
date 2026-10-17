import csv
import io
import json
import os
from datetime import datetime, timezone
from typing import List, Literal, Optional
 
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel
 
app = FastAPI(title="QuakeFix API")
 
class Event(BaseModel):
    id: str
    magnitude: float
    depth_km: float
    lat: float
    lon: float
 
class Directive(BaseModel):
    id: str
    priority: int
    hazard: str
    action: str
    risk: Optional[float] = None
    confidence: Optional[float] = None
    route_nodes: Optional[List[int]] = None
    status: Literal["pending", "approved", "overridden"] = "pending"
 
class Zone(BaseModel):
    id: str
    hazard: str
    lat: float
    lon: float
    radius_m: float
    risk: float
 
class Route(BaseModel):
    id: str
    kind: Literal["ambulance", "evacuation"]
    label: str
    path: List[List[float]]
 
class QuakeState(BaseModel):
    event: Event
    directives: List[Directive]
    zones: List[Zone]
    routes: List[Route]
 
class Decision(BaseModel):
    decision: Literal["approve", "override"]
    note: str = ""
 
class SegmentFeatures(BaseModel):
    magnitude: float
    dist_km: float
    shaking: float
    pipe_age: float
    sensor_var: float
    gas_main_dist_m: float
    elevation_m: float
    building_age: float
 
# --- SAN FRANCISCO DEMO CITY (Fixed from 0.0 placeholder) ---
STATE = QuakeState(
    event=Event(id="sf-demo-1", magnitude=6.7, depth_km=10.0, lat=37.7749, lon=-122.4194),
    directives=[
        Directive(id="D-001", priority=1, hazard="gas_leak", action="Isolate Valve Sector 4 - Mission District",
                  risk=0.87, confidence=0.74),
        Directive(id="D-002", priority=2, hazard="fire", action="Dispatch fire unit to Zone F2 - SOMA",
                  risk=0.71, confidence=0.66),
        Directive(id="D-003", priority=3, hazard="flood", action="Close underpass at Route 7",
                  risk=0.64, confidence=0.61),
        Directive(id="D-004", priority=4, hazard="road_blocked",
                  action="Reroute Ambulance Unit 2 via Route B", route_nodes=[101, 205, 318]),
        Directive(id="D-005", priority=5, hazard="evacuation",
                  action="Evacuate District 3 to Assembly Point A - Embarcadero"),
    ],
    zones=[
        Zone(id="Z-1", hazard="fire", lat=37.7849, lon=-122.4094, radius_m=600, risk=0.71),
        Zone(id="Z-2", hazard="flood", lat=37.8049, lon=-122.4294, radius_m=800, risk=0.64),
        Zone(id="Z-3", hazard="gas_leak", lat=37.7749, lon=-122.4194, radius_m=400, risk=0.87),
    ],
    routes=[
        Route(id="R-1", kind="ambulance", label="Ambulance Unit 2 via Route B",
              path=[[-122.4194, 37.7749], [-122.4094, 37.7849], [-122.4000, 37.7949]]),
        Route(id="R-2", kind="evacuation", label="District 3 to Assembly Point A",
              path=[[-122.4194, 37.7749], [-122.4294, 37.7649], [-122.4394, 37.7549]]),
    ],
)
 
AUDIT_LOG: list = []
os.makedirs("logs", exist_ok=True)
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
try:
    ART = joblib.load(MODEL_PATH)
except Exception:
    ART = None
 
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ART is not None, "city": "San Francisco", "lat": 37.7749, "lon": -122.4194}
 
@app.get("/events/latest", response_model=QuakeState)
def latest():
    return STATE
 
@app.get("/hazards/zones", response_model=List[Zone])
def zones():
    return STATE.zones
 
@app.get("/routes", response_model=List[Route])
def routes():
    return STATE.routes
 
@app.post("/directives/generate", response_model=QuakeState)
def generate():
    return STATE
 
@app.post("/directives/{directive_id}/decision", response_model=Directive)
def decide(directive_id: str, body: Decision):
    for d in STATE.directives:
        if d.id == directive_id:
            d.status = "approved" if body.decision == "approve" else "overridden"
            entry = {"time": datetime.now(timezone.utc).isoformat(), "directive": d.id,
                     "decision": body.decision, "note": body.note}
            AUDIT_LOG.append(entry)
            with open("logs/audit.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            return d
    raise HTTPException(status_code=404, detail="Directive not found")
 
@app.get("/export/sms", response_class=PlainTextResponse)
def export_sms():
    e = STATE.event
    parts = [f"QUAKEFIX M{e.magnitude} SF"]
    parts += [f"{d.id} P{d.priority} {d.hazard.upper()}: {d.action} [{d.status}]"
              for d in STATE.directives]
    return " | ".join(parts)
 
@app.get("/export/csv")
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "priority", "hazard", "action", "risk", "confidence", "status"])
    for d in STATE.directives:
        w.writerow([d.id, d.priority, d.hazard, d.action, d.risk, d.confidence, d.status])
    return Response(content=buf.getvalue(), media_type="text/csv")
 
@app.post("/predict")
def predict(f: SegmentFeatures):
    if ART is None:
        raise HTTPException(status_code=503, detail="Model not loaded - use dummy until Member 3 delivers")
    X = pd.DataFrame([f.model_dump()])[ART["features"]]
    out = {}
    for h in ["burst", "fire", "flood"]:
        p = float(ART[h]["pipeline"].predict_proba(X)[0, 1])
        out[h] = {"risk": p, "flag": p >= ART[h]["threshold"], "threshold": ART[h]["threshold"]}
    out["damage"] = {"score": float(ART["damage"]["pipeline"].predict(X)[0])}
    return out
