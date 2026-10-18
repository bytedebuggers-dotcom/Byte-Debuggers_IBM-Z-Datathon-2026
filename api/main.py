# api/main.py
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import numpy as np

# Import your routing contract functions
from api.routing import get_graph, nodes_in_zones, safe_route_emergency, safe_route_evacuation, nodes_to_lonlat

app = FastAPI(title="QuakeFix API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for graph and model
model_data = None
road_graph = None

@app.on_event("startup")
def startup_load():
    global model_data, road_graph
    
    # 1. Load ML Model pipeline
    model_path = os.getenv("QUAKEFIX_MODEL", "models/model.joblib")
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        
    # 2. Load Road Graph via your routing module
    try:
        road_graph = get_graph()
    except Exception:
        road_graph = None

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "model_loaded": model_data is not None,
        "graph_loaded": road_graph is not None
    }

class RouteRequest(BaseModel):
    origin_lonlat: list[float]
    dest_lonlat: list[float]
    hazard_zones: list[dict] = []

@app.post("/routes/safe-emergency")
def calculate_emergency_route(payload: RouteRequest):
    if road_graph is None:
        raise HTTPException(status_code=503, detail="Road graph not initialized.")
    
    blocked = nodes_in_zones(road_graph, payload.hazard_zones)
    
    result = safe_route_emergency(
        origin_lonlat=payload.origin_lonlat,
        dest_lonlat=payload.dest_lonlat,
        hazard_nodes=blocked,
        G=road_graph
    )
    return result

@app.get("/events/latest")
def get_latest_event():
    if road_graph is None:
        raise HTTPException(status_code=503, detail="Road graph not initialized.")
    
    return {
        "event": {"id": "sf-demo-1", "magnitude": 6.7, "depth_km": 10.0, "lat": 37.7749, "lon": -122.4194},
        "directives": [
            {"id": "D-001", "priority": 1, "hazard": "gas_leak", "action": "Isolate Valve Sector 4", "risk": 0.87, "confidence": 0.74, "status": "pending"},
            {"id": "D-002", "priority": 2, "hazard": "fire", "action": "Dispatch fire unit to Zone F2", "risk": 0.71, "confidence": 0.66, "status": "pending"}
        ],
        "zones": [
            {"id": "Z-1", "hazard": "fire", "lat": 37.7849, "lon": -122.4094, "radius_m": 600, "risk": 0.71}
        ],
        "routes": [
            {"id": "R-1", "kind": "ambulance", "label": "Ambulance Unit 2", "path": [[-122.4194, 37.7749], [-122.4094, 37.8191]]}
        ]
    }

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_hazard_risk(payload: PredictRequest):
    if model_data is None:
        raise HTTPException(status_code=503, detail="ML model pipeline (model.joblib) not loaded.")
    
    if len(payload.features) != 8:
        raise HTTPException(status_code=400, detail="Expected exactly 8 feature values.")
    
    X = np.array([payload.features])
    
    burst_prob = float(model_data["burst"]["pipeline"].predict_proba(X)[0][1])
    fire_prob = float(model_data["fire"]["pipeline"].predict_proba(X)[0][1])
    flood_prob = float(model_data["flood"]["pipeline"].predict_proba(X)[0][1])
    damage_score = float(model_data["damage"]["pipeline"].predict(X)[0])
    
    return {
        "burst_risk": burst_prob,
        "fire_risk": fire_prob,
        "flood_risk": flood_prob,
        "damage_score": damage_score
    }

@app.get("/export/sms")
def export_sms():
    return "EMERGENCY: QuakeFix Alert. Sector 4 Gas Leak Isolated. Evacuate District 3 immediately."

@app.get("/export/csv")
def export_csv():
    return "directive_id,hazard,action,priority,status\nD-001,gas_leak,Isolate Valve Sector 4,1,pending\nD-002,fire,Dispatch fire unit to Zone F2,2,pending"