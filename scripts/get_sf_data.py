"""Download real San Francisco data for QuakeFix.
Run from the repository root:   python scripts/get_sf_data.py
Needs internet and:  pip install osmnx geopandas requests pandas
"""
import os

import osmnx as ox
import pandas as pd
import requests

CENTER = (37.7600, -122.4300)     # lat, lon: Mission / central SF area
RADIUS_M = 3000                   # 3 km keeps files small and downloads fast
FULL_CITY = False                 # set True to download all of San Francisco (large, slow)

os.makedirs("data/raw", exist_ok=True)

# --- 1. Road network (for routing) ---
if FULL_CITY:
    G = ox.graph_from_place("San Francisco, California, USA", network_type="drive")
else:
    G = ox.graph_from_point(CENTER, dist=RADIUS_M, network_type="drive")
ox.save_graphml(G, "data/raw/road_graph.graphml")
print("Road graph:", len(G.nodes), "nodes,", len(G.edges), "edges")

# --- 2. Building footprints ---
if FULL_CITY:
    b = ox.features_from_place("San Francisco, California, USA", tags={"building": True})
else:
    b = ox.features_from_point(CENTER, tags={"building": True}, dist=RADIUS_M)
b = b[["geometry"] + [c for c in ["building"] if c in b.columns]].reset_index(drop=True)
b.to_file("data/raw/buildings.geojson", driver="GeoJSON")
print("Buildings:", len(b))

# --- 3. Real earthquakes in the Bay Area (USGS) ---
params = {"format": "geojson", "starttime": "2000-01-01", "minmagnitude": 3.0,
          "minlatitude": 37.0, "maxlatitude": 38.5,
          "minlongitude": -123.2, "maxlongitude": -121.5, "limit": 500}
r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params=params, timeout=60)
r.raise_for_status()
events = pd.DataFrame([{
    "id": f["id"], "mag": f["properties"]["mag"], "place": f["properties"]["place"],
    "time": f["properties"]["time"], "lon": f["geometry"]["coordinates"][0],
    "lat": f["geometry"]["coordinates"][1], "depth_km": f["geometry"]["coordinates"][2],
} for f in r.json()["features"]])
events.to_csv("data/raw/usgs_events.csv", index=False)
print("USGS events:", len(events))
