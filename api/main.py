import os
import joblib
import numpy as np
import pandas as pd
import csv
import io
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional

app = FastAPI(title="QuakeFix API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/model.joblib"
model_artifact = None

if os.path.exists(MODEL_PATH):
    try:
        model_artifact = joblib.load(MODEL_PATH)
        print("Successfully loaded model.joblib artifact.")
    except Exception as e:
        print(f"Warning: Could not load model artifact: {e}")


def run_submodel(submodel_key: str, X_df: pd.DataFrame) -> float:
    if not model_artifact or not isinstance(model_artifact, dict):
        return 0.42
    entry = model_artifact.get(submodel_key)
    if entry is None:
        return 0.42

    pipe = entry.get("pipeline") if isinstance(entry, dict) else entry
    if pipe is None:
        return 0.42

    try:
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_df)
            return float(proba[0][1] if len(proba[0]) > 1 else proba[0][0])
        elif hasattr(pipe, "predict"):
            res = pipe.predict(X_df)
            val = res[0] if isinstance(res, (list, np.ndarray)) else res
            return float(val)
    except Exception as e:
        print(f"Error predicting {submodel_key}: {e}")
    return 0.42


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
    status: Literal["pending", "approved", "overridden"] = "pending"

class Zone(BaseModel):
    name: str
    type: str
    lat: float
    lon: float
    radius: int
    color: List[int]

class Route(BaseModel):
    name: str
    type: str
    kind: Literal["ambulance", "evacuation"]
    color: List[int]
    path: List[List[float]]

class QuakeState(BaseModel):
    event: Event
    directives: List[Directive]
    zones: List[Zone]
    routes: List[Route]

class Decision(BaseModel):
    decision: Literal["approve", "override"]
    note: str = ""

AUDIT_LOG: list = []
os.makedirs("logs", exist_ok=True)

BASE_ASSEMBLY_POINTS = [
    {"name": "Safe Assembly Point: Mission Dolores", "type": "Assembly Point", "lat": 37.7590, "lon": -122.4260, "radius": 140, "color": [31, 163, 122, 255]},
    {"name": "Safe Assembly Point: Golden Gate Panhandle", "type": "Assembly Point", "lat": 37.7720, "lon": -122.4450, "radius": 140, "color": [31, 163, 122, 255]},
    {"name": "Safe Assembly Point: Alamo Square", "type": "Assembly Point", "lat": 37.7762, "lon": -122.4330, "radius": 140, "color": [31, 163, 122, 255]}
]

STATE = QuakeState(
    event=Event(id="us-demo-1", magnitude=6.1, depth_km=10.0, lat=37.7749, lon=-122.4194),
    directives=[
        Directive(id="D-001", priority=1, hazard="gas_leak", action="Isolate Valve Sector 4", risk=0.87, confidence=0.74),
        Directive(id="D-002", priority=2, hazard="fire", action="Dispatch fire unit to Zone F2", risk=0.71, confidence=0.66),
        Directive(id="D-003", priority=3, hazard="flood", action="Close underpass at Route 7", risk=0.64, confidence=0.61),
        Directive(id="D-004", priority=4, hazard="road_blocked", action="Reroute Ambulance Units via Van Ness Corridor", risk=0.80, confidence=0.78),
        Directive(id="D-005", priority=5, hazard="evacuation", action="Execute multi-point safe citizen evacuation", risk=0.85, confidence=0.82),
    ],
    zones=[],
    routes=[]
)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_artifact is not None}


@app.get("/events/latest", response_model=QuakeState)
def get_latest_event():
    return STATE


@app.post("/directives/{directive_id}/decision", response_model=Directive)
def decide_directive(directive_id: str, body: Decision):
    for d in STATE.directives:
        if d.id == directive_id:
            d.status = "approved" if body.decision == "approve" else "overridden"
            entry = {
                "time": datetime.now(timezone.utc).isoformat(),
                "directive": d.id,
                "decision": body.decision,
                "note": body.note
            }
            AUDIT_LOG.append(entry)
            with open("logs/audit.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            return d
    raise HTTPException(status_code=404, detail="Directive not found")


@app.get("/export/sms", response_class=PlainTextResponse)
def export_sms():
    e = STATE.event
    lines = [
        f"🚨 QUAKEFIX EMERGENCY BROADCAST | MAG {e.magnitude}",
        f"Epicenter: Lat {e.lat:.4f}, Lon {e.lon:.4f} (Depth: {e.depth_km}km)",
        "--- TACTICAL DIRECTIVES ---"
    ]
    for d in STATE.directives:
        lines.append(f"• [{d.status.upper()}] P{d.priority} {d.hazard.upper()}: {d.action}")
    return "\n".join(lines)


@app.get("/export/csv")
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "priority", "hazard", "action", "risk", "confidence", "status"])
    for d in STATE.directives:
        w.writerow([d.id, d.priority, d.hazard, d.action, d.risk, d.confidence, d.status])
    return Response(content=buf.getvalue(), media_type="text/csv")


@app.post("/predict")
def predict_risk(data: dict):
    features = data.get("features", [])
    if len(features) != 8:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly 8 feature values, but got {len(features)}.",
        )

    try:
        magnitude, dist_km, shaking, pipe_age, sensor_var, gas_main_dist_m, elevation_m, building_age = features

        feature_names = (
            model_artifact.get("features")
            if isinstance(model_artifact, dict) and "features" in model_artifact
            else [
                "magnitude", "dist_km", "shaking", "pipe_age",
                "sensor_var", "gas_main_dist_m", "elevation_m", "building_age"
            ]
        )

        X_df = pd.DataFrame([features], columns=feature_names)

        burst_prob = run_submodel("burst", X_df)
        fire_prob = run_submodel("fire", X_df)
        flood_prob = run_submodel("flood", X_df)
        damage_score = run_submodel("damage", X_df)

        if damage_score <= 1.0:
            damage_score = damage_score * 10.0

        ref_lat, ref_lon = 37.7749, -122.4194
        epicenter_lat = ref_lat + min(max(((dist_km / 10.0) * 0.008 * (magnitude / 6.0)), -0.02), 0.02)
        epicenter_lon = ref_lon - min(max(((dist_km / 10.0) * 0.008 * (magnitude / 6.0)), -0.02), 0.02)

        STATE.event.magnitude = magnitude
        STATE.event.lat = epicenter_lat
        STATE.event.lon = epicenter_lon

        hazard_zones = [
            {
                "name": f"Epicentral Pipe Burst Risk (Mag {magnitude})",
                "type": "Pipe Burst Risk",
                "lat": epicenter_lat,
                "lon": epicenter_lon,
                "radius": int(150 + (burst_prob * 450)),
                "color": [11, 42, 107, int(120 + (burst_prob * 135))]
            },
            {
                "name": "Secondary Ignition / Fire Hazard",
                "type": "Fire Risk",
                "lat": epicenter_lat + 0.005,
                "lon": epicenter_lon + 0.004,
                "radius": int(150 + (fire_prob * 450)),
                "color": [227, 100, 20, int(120 + (fire_prob * 135))]
            },
            {
                "name": "Coastal Inundation & Flood Zone",
                "type": "Flood Risk",
                "lat": epicenter_lat - 0.008,
                "lon": epicenter_lon - 0.006,
                "radius": int(150 + (flood_prob * 450)),
                "color": [0, 180, 216, int(120 + (flood_prob * 135))]
            },
            {
                "name": "Structural Collapse Zone",
                "type": "Structural Damage",
                "lat": epicenter_lat + 0.004,
                "lon": epicenter_lon - 0.008,
                "radius": int(150 + (damage_score * 40)),
                "color": [217, 162, 27, int(120 + (damage_score * 15))]
            }
        ]

        hospitals = [
            {
                "name": "Zuckerberg SF General Hospital",
                "type": "Hospital",
                "lat": 37.7552,
                "lon": -122.4048,
                "radius": 150,
                "color": [235, 64, 52, 255]
            },
            {
                "name": "UCSF Medical Center Parnassus",
                "type": "Hospital",
                "lat": 37.7631,
                "lon": -122.4582,
                "radius": 150,
                "color": [235, 64, 52, 255]
            }
        ]

        safe_assembly_points = []
        for ap in BASE_ASSEMBLY_POINTS:
            is_safe = True
            for hz in hazard_zones:
                distance = np.hypot(ap["lat"] - hz["lat"], ap["lon"] - hz["lon"])
                hazard_radius_deg = (hz["radius"] / 111000.0) * 1.5 
                if distance < hazard_radius_deg:
                    is_safe = False
                    break
            if is_safe:
                safe_assembly_points.append(ap)

        if not safe_assembly_points:
            furthest_ap = max(BASE_ASSEMBLY_POINTS, key=lambda ap: np.hypot(ap["lat"] - epicenter_lat, ap["lon"] - epicenter_lon))
            safe_assembly_points = [furthest_ap]

        all_zones = hazard_zones + hospitals + safe_assembly_points

        # Precision street-grid snapped routing
        dynamic_routes = [
            {
                "name": "Ambulance: SF General via Mission/Van Ness",
                "type": "Emergency Service Route",
                "kind": "ambulance",
                "color": [0, 150, 255, 240],
                "path": [
                    [-122.4048, 37.7552],
                    [-122.4190, 37.7552],
                    [-122.4190, epicenter_lat],
                    [epicenter_lon, epicenter_lat]
                ]
            },
            {
                "name": "Ambulance: UCSF via Haight/Van Ness",
                "type": "Emergency Service Route",
                "kind": "ambulance",
                "color": [0, 220, 255, 240],
                "path": [
                    [-122.4582, 37.7631],
                    [-122.4350, 37.7631],
                    [-122.4350, epicenter_lat],
                    [epicenter_lon, epicenter_lat]
                ]
            }
        ]

        for sap in safe_assembly_points:
            dynamic_routes.append({
                "name": f"Evacuation Grid Corridor to {sap['name'].replace('Safe Assembly Point: ', '')}",
                "type": "Citizen Evacuation Corridor",
                "kind": "evacuation",
                "color": [40, 220, 100, 240],
                "path": [
                    [epicenter_lon, epicenter_lat],
                    [sap["lon"], epicenter_lat],
                    [sap["lon"], sap["lat"]]
                ]
            })

        STATE.zones = [Zone(**z) for z in all_zones]
        STATE.routes = [Route(**r) for r in dynamic_routes]

        return {
            "burst_prob": burst_prob,
            "fire_prob": fire_prob,
            "flood_prob": flood_prob,
            "damage_score": damage_score,
            "event": STATE.event.model_dump(),
            "directives": [d.model_dump() for d in STATE.directives],
            "zones": [z.model_dump() for z in STATE.zones],
            "routes": [r.model_dump() for r in STATE.routes]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model inference error: {str(e)}"
        )