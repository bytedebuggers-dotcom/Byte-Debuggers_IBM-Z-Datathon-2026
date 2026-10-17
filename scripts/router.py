import osmnx as ox
import networkx as nx
import json
import pandas as pd

def find_safe_route(start_lat, start_lon, dest_type="assembly"):
    print("Loading road network graph...")
    G = ox.load_graphml("data/raw/road_graph.graphml")
    
    # Load assembly points or hospitals
    filename = "data/raw/assembly_points.json" if dest_type == "assembly" else "data/raw/hospitals.json"
    with open(filename, "r") as f:
        destinations = json.load(f)
        
    # Find nearest network nodes
    orig_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
    
    # Find nearest destination node
    dest_node = ox.distance.nearest_nodes(G, X=destinations[0]["lon"], Y=destinations[0]["lat"])
    
    print(f"Calculating shortest path to {dest_type}...")
    route = nx.shortest_path(G, orig_node, dest_node, weight="length")
    
    print(f"Route found with {len(route)} nodes!")
    return route

if __name__ == "__main__":
    # Test route from a sample SF coordinate
    find_safe_route(37.7749, -122.4194, dest_type="assembly")