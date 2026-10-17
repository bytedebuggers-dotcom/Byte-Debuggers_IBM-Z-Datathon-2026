# QuakeFix API contract
 
GET  /health
GET  /events/latest            -> event + directives + zones + routes
GET  /hazards/zones            -> list of hazard zones
GET  /routes                   -> list of ambulance and evacuation routes
POST /directives/generate      -> event + directives + zones + routes
POST /directives/{id}/decision body: {"decision": "approve" | "override", "note": "..."}
GET  /export/sms               -> plain text
GET  /export/csv               -> CSV
POST /predict                  -> burst, fire, flood risk and damage score for one segment
 
## Hazard values used in directives and zones
gas_leak | fire | flood | structural_damage | road_blocked | evacuation
 
## Example response (events/latest) - San Francisco Demo City
{
  "event": {"id": "sf-demo-1", "magnitude": 6.7, "depth_km": 10.0, "lat": 37.7749, "lon": -122.4194},
  "directives": [
    {"id": "D-001", "priority": 1, "hazard": "gas_leak", "action": "Isolate Valve Sector 4",
     "risk": 0.87, "confidence": 0.74, "status": "pending"},
    {"id": "D-002", "priority": 2, "hazard": "fire", "action": "Dispatch fire unit to Zone F2",
     "risk": 0.71, "confidence": 0.66, "status": "pending"},
    {"id": "D-005", "priority": 5, "hazard": "evacuation",
     "action": "Evacuate District 3 to Assembly Point A", "status": "pending"}
  ],
  "zones": [
    {"id": "Z-1", "hazard": "fire", "lat": 37.7849, "lon": -122.4094, "radius_m": 600, "risk": 0.71}
  ],
  "routes": [
    {"id": "R-1", "kind": "ambulance", "label": "Ambulance Unit 2 via Route B",
     "path": [[-122.4194, 37.7749], [-122.4094, 37.7849], [-122.4000, 37.7949]]},
    {"id": "R-2", "kind": "evacuation", "label": "District 3 to Assembly Point A",
     "path": [[-122.4194, 37.7749], [-122.4294, 37.7649], [-122.4394, 37.7549]]}
  ]
}
 
## Model file contract (Member 3 -> Member 1)
models/model.joblib is a dict with these keys:
  "features" : list of 8 column names, in order
  "burst"    : {"pipeline": sklearn Pipeline with predict_proba, "threshold": float}
  "fire"     : {"pipeline": ..., "threshold": float}
  "flood"    : {"pipeline": ..., "threshold": float}
  "damage"   : {"pipeline": sklearn regression Pipeline with predict, returns 0 to 1}
 
Features: magnitude, dist_km, shaking, pipe_age, sensor_var,
          gas_main_dist_m, elevation_m, building_age
 
## Routing contract (Member 3 -> Member 1)
api/routing.py provides: nodes_in_zones, safe_route, evacuation_route
Path points are [lon, lat] pairs so the dashboard can draw them.
