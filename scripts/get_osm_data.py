import osmnx as ox
import json

# Choose your demo city or neighborhood (smaller area downloads faster)
PLACE = "San Francisco, California, USA"

print(f"Downloading road network for {PLACE}...")
G = ox.graph_from_place(PLACE, network_type="drive")
ox.save_graphml(G, "data/raw/road_graph.graphml")
print("Road graph saved to data/raw/road_graph.graphml")

print("Extracting assembly points (parks/pitches)...")
spaces = ox.features_from_place(PLACE, tags={"leisure": ["park", "pitch"]})

# Convert to projected CRS to find centroids, then back to lat/lon
cent = spaces.to_crs(epsg=32610).geometry.centroid.to_crs(epsg=4326) # UTM Zone 10N for SF
aps = [{"name": f"Assembly {i+1}", "lon": float(pt.x), "lat": float(pt.y)}
       for i, pt in enumerate(cent.head(5))]

with open("data/raw/assembly_points.json", "w") as f:
    json.dump(aps, f, indent=2)

print(f"Saved {len(aps)} assembly points to data/raw/assembly_points.json")