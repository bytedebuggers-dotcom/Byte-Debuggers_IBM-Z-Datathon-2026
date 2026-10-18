import osmnx as ox
import networkx as nx
import json
import pandas as pd

def load_routing_graph():
    """Loads the OSM road network graph from raw data."""
    print("Loading road network graph...")
    G = ox.load_graphml("data/raw/road_graph.graphml")
    return G

def get_safe_evacuation_route(start_lat, start_lon, dest_type="assembly"):
    G = load_routing_graph()
    
    # Load destinations (assembly points or hospitals)
    filename = "data/raw/assembly_points.json" if dest_type == "assembly" else "data/raw/hospitals.json"
    with open(filename, "r") as f:
        destinations = json.load(f)
        
    if not destinations:
        raise ValueError(f"No destinations found in {filename}")
        
    # Find nearest nodes for origin and destination
    orig_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
    dest_node = ox.distance.nearest_nodes(G, X=destinations[0]["lon"], Y=destinations[0]["lat"])
    
    print(f"Computing shortest path to {dest_type} ({destinations[0]['name']})...")
    route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="length")
    
    # Extract coordinates for mapping
    route_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in route_nodes]
    
    print(f"Route successfully calculated with {len(route_nodes)} nodes.")
    return route_coords

if __name__ == "__main__":
    # Test route calculation for San Francisco downtown coordinate
    coords = get_safe_evacuation_route(37.7749, -122.4194, dest_type="assembly")
    print("Sample route coordinates (first 3):", coords[:3])