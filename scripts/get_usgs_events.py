import requests
import pandas as pd

url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
params = {
    "format": "geojson", 
    "starttime": "2025-01-01", 
    "endtime": "2026-10-01",
    "minmagnitude": 5.0, 
    "limit": 200
}

print("Fetching earthquake data from USGS...")
r = requests.get(url, params=params, timeout=30)
r.raise_for_status()
feats = r.json()["features"]

events = pd.DataFrame([{
    "id": f["id"], 
    "mag": f["properties"]["mag"], 
    "place": f["properties"]["place"],
    "time": f["properties"]["time"],
    "lon": f["geometry"]["coordinates"][0], 
    "lat": f["geometry"]["coordinates"][1],
    "depth_km": f["geometry"]["coordinates"][2]
} for f in feats])

events.to_csv("data/raw/usgs_events.csv", index=False)
print(f"Successfully saved {len(events)} events to data/raw/usgs_events.csv")