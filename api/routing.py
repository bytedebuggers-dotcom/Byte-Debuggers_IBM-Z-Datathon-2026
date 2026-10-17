"""
QuakeFix - Member 3: safe-route logic (NetworkX; OSMnx only used to load the graph if installed).

CONTRACT FUNCTIONS (docs/api_contract.md; what Member 1's API should call), near the bottom:
    nodes_in_zones(G, zones) -> set of node ids        zones = [{lon, lat, radius_m, hazard}]
    safe_route(G, origin_lonlat, dest_lonlat, blocked_nodes) -> [node ids] | None
    evacuation_route(G, origin_lonlat, assembly_points, blocked_nodes) -> (name, [node ids]) | (None, None)
    nodes_to_lonlat(G, nodes) -> [[lon, lat], ...]     for the "path" field the dashboard draws
    load_graph(path)                                   loads data/raw/road_graph.graphml
Run `python api/routing.py` for a self-test (prints ALL ROUTING TESTS PASSED).

Richer variants that return a dict with status/message/coords (optional to use):

    safe_route_emergency(origin_lonlat, dest_lonlat, hazard_nodes=(), G=None)
        Fastest path for an ambulance / fire engine from a station to an incident,
        never passing through any hazard node (roadblocks, active fires, flood zones).

    safe_route_evacuation(origin_lonlat, assembly_points, hazard_nodes=(), risk_nodes=(), G=None)
        Path for citizens from where they are to the nearest SAFE assembly point.
        hazard_nodes are impassable. risk_nodes (e.g. the predicted high-risk zone)
        stay passable but cost much more, so routes lead AWAY from the impact zone.

Both return a dict (never raise on "no route"):
    {"status": "ok" | "no_route",
     "route_nodes": [node ids] | None,     # contract field for directives ("route_nodes")
     "coords": [[lon, lat], ...] | None,   # ready for a pydeck PathLayer
     "length_m": float | None,
     "message": str}                       # plain text for the commander

Helpers:
    get_graph(path)              load data/raw/road_graph.graphml (cached)
    build_demo_graph()           small SYNTHETIC grid so tests/demos work with no map data
    nodes_in_radius(G, lon, lat, radius_m)   turn a fire/flood circle into hazard nodes

Coordinates are always (lon, lat), like OSMnx.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np

DEFAULT_GRAPH_PATH = os.getenv("QUAKEFIX_GRAPH", "data/raw/road_graph.graphml")
RISK_PENALTY = 5.0  # a road touching a risk node costs 5x its length in evacuation routing


# --------------------------------------------------------------------------- geometry
def _haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres; works on scalars or numpy arrays."""
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _node_arrays(G):
    nodes = list(G.nodes)
    xs = np.array([float(G.nodes[n]["x"]) for n in nodes])
    ys = np.array([float(G.nodes[n]["y"]) for n in nodes])
    return nodes, xs, ys


def nearest_node(G, lon: float, lat: float, exclude: Iterable = ()):
    """Nearest graph node to (lon, lat), skipping any node in `exclude`."""
    nodes, xs, ys = _node_arrays(G)
    d = _haversine_m(lon, lat, xs, ys)
    skip = set(exclude)
    if skip:
        mask = np.array([n in skip for n in nodes])
        if mask.all():
            return None
        d = np.where(mask, np.inf, d)
    return nodes[int(np.argmin(d))]


def nodes_in_radius(G, lon: float, lat: float, radius_m: float) -> set:
    """All nodes within radius_m of a point: use for a fire / flood / collapse zone."""
    nodes, xs, ys = _node_arrays(G)
    d = _haversine_m(lon, lat, xs, ys)
    return {n for n, dist in zip(nodes, d) if dist <= radius_m}


# --------------------------------------------------------------------------- graph
def _simplify(G) -> nx.DiGraph:
    """Collapse a (Multi)DiGraph to a DiGraph keeping the shortest parallel edge."""
    S = nx.DiGraph()
    for n, data in G.nodes(data=True):
        S.add_node(n, x=float(data["x"]), y=float(data["y"]))
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 0.0))
        if not S.has_edge(u, v) or length < S[u][v]["length"]:
            S.add_edge(u, v, length=length)
    return S


@lru_cache(maxsize=2)
def get_graph(path: str = DEFAULT_GRAPH_PATH) -> nx.DiGraph:
    """Load the road graph saved by Member 2 (OSMnx GraphML). Cached after first call."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Ask Member 2 for road_graph.graphml (data/raw is not in Git), "
            "or use build_demo_graph() for a synthetic test map.")
    try:
        import osmnx as ox
        G = ox.load_graphml(p)
    except ImportError:  # osmnx missing (e.g. on s390x): parse GraphML directly
        G = nx.read_graphml(p)
        G = nx.relabel_nodes(G, {n: int(n) if str(n).lstrip("-").isdigit() else n for n in G.nodes})
    return _simplify(G)


def build_demo_graph(rows: int = 7, cols: int = 7, origin=(18.42, -33.92), step_deg: float = 0.005) -> nx.DiGraph:
    """SYNTHETIC grid street map (two-way streets) for tests and offline demos. Not real roads."""
    G = nx.DiGraph(name="synthetic-demo-grid")
    nid = lambda r, c: r * cols + c
    for r in range(rows):
        for c in range(cols):
            G.add_node(nid(r, c), x=origin[0] + c * step_deg, y=origin[1] + r * step_deg)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 < rows and c2 < cols:
                    a, b = nid(r, c), nid(r2, c2)
                    length = float(_haversine_m(G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[b]["x"], G.nodes[b]["y"]))
                    G.add_edge(a, b, length=length)
                    G.add_edge(b, a, length=length)
    return G


# --------------------------------------------------------------------------- routing
def _result(G, path, length, message) -> dict:
    if path is None:
        return {"status": "no_route", "route_nodes": None, "coords": None, "length_m": None, "message": message}
    return {
        "status": "ok",
        "route_nodes": [int(n) if isinstance(n, (int, np.integer)) else n for n in path],
        "coords": [[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in path],
        "length_m": float(length),
        "message": message,
    }


def safe_route_emergency(origin_lonlat: Sequence[float], dest_lonlat: Sequence[float],
                         hazard_nodes: Iterable = (), G: Optional[nx.DiGraph] = None) -> dict:
    """Fastest (shortest) road path that avoids every hazard node."""
    G = G if G is not None else get_graph()
    hazards = {n for n in hazard_nodes if n in G}
    orig = nearest_node(G, origin_lonlat[0], origin_lonlat[1], exclude=hazards)
    dest = nearest_node(G, dest_lonlat[0], dest_lonlat[1], exclude=hazards)
    if orig is None or dest is None:
        return _result(G, None, None, "Every road node is blocked. Dispatch cannot route; use manual command.")
    H = nx.restricted_view(G, hazards, [])  # same graph without hazard nodes (no copy)
    try:
        length, path = nx.single_source_dijkstra(H, orig, dest, weight="length")
    except nx.NetworkXNoPath:
        return _result(G, None, None,
                       f"No safe route: destination is cut off by {len(hazards)} blocked node(s). "
                       "Commander decision needed (air support, foot team, or clear the road).")
    return _result(G, path, length, f"Route found avoiding {len(hazards)} blocked node(s).")


def safe_route_evacuation(origin_lonlat: Sequence[float], assembly_points: Sequence[Sequence[float]],
                          hazard_nodes: Iterable = (), risk_nodes: Iterable = (),
                          G: Optional[nx.DiGraph] = None) -> dict:
    """Route citizens to the best reachable assembly point.

    hazard_nodes: impassable (fire, flood, collapse, roadblock).
    risk_nodes:   predicted high-risk zone; passable but RISK_PENALTY x more costly, so the
                  route leaves the zone quickly and prefers assembly points outside it.
    Assembly points inside the hazard/risk zone are skipped (not safe). The result has the extra keys
    "assembly_index" and "assembly_node" saying which point was chosen.
    """
    G = G if G is not None else get_graph()
    hazards = {n for n in hazard_nodes if n in G}
    risky = {n for n in risk_nodes if n in G} - hazards
    orig = nearest_node(G, origin_lonlat[0], origin_lonlat[1], exclude=hazards)
    if orig is None:
        out = _result(G, None, None, "Every road node is blocked. Cannot plan an evacuation route.")
        return {**out, "assembly_index": None, "assembly_node": None}

    # candidate assembly nodes: the node nearest each point; a point that sits in the
    # hazard or risk zone is NOT a safe assembly point, so it is skipped (not re-snapped)
    candidates = {}
    for i, (lon, lat) in enumerate(assembly_points):
        node = nearest_node(G, lon, lat)
        if node is not None and node not in hazards and node not in risky:
            candidates.setdefault(node, i)

    H = nx.restricted_view(G, hazards, [])

    def weight(u, v, d):
        w = d["length"]
        return w * RISK_PENALTY if (u in risky or v in risky) else w

    dist, paths = nx.single_source_dijkstra(H, orig, weight=weight)
    reachable = [n for n in candidates if n in dist]
    if not reachable:
        out = _result(G, None, None,
                      "No reachable safe assembly point from this location. Shelter in place or request "
                      "manual evacuation by the commander.")
        return {**out, "assembly_index": None, "assembly_node": None}
    best = min(reachable, key=lambda n: dist[n])
    path = paths[best]
    true_len = sum(G[u][v]["length"] for u, v in zip(path[:-1], path[1:]))
    out = _result(G, path, true_len,
                  f"Evacuation route to assembly point {candidates[best]} found "
                  f"(avoiding {len(hazards)} blocked node(s), discouraging {len(risky)} risk-zone node(s)).")
    out["assembly_index"] = candidates[best]
    out["assembly_node"] = out["route_nodes"][-1]
    return out


# --------------------------------------------------------------------------- API contract functions
# These three are what docs/api_contract.md promises Member 1 ("api/routing.py provides:
# nodes_in_zones, safe_route, evacuation_route"). They accept either a raw OSMnx MultiDiGraph or
# a graph from get_graph()/build_demo_graph(). Path points for the dashboard are [lon, lat]:
# use nodes_to_lonlat(G, route_nodes).
_SIMPLE_CACHE: dict = {}


def _simple(G):
    """Return a simple DiGraph view of G (collapsing parallel edges once, then cached)."""
    if not G.is_multigraph():
        return G
    hit = _SIMPLE_CACHE.get(id(G))
    if hit is None or hit[0] is not G:
        hit = (G, _simplify(G))
        _SIMPLE_CACHE[id(G)] = hit
    return hit[1]


def load_graph(path: str = DEFAULT_GRAPH_PATH):
    """Load data/raw/road_graph.graphml (cached). Same as get_graph()."""
    return get_graph(str(path))


def nodes_in_zones(G, zones) -> set:
    """zones: list of dicts {lon, lat, radius_m, hazard}. Returns ids of road nodes inside any zone."""
    blocked: set = set()
    for z in zones:
        blocked |= nodes_in_radius(G, z["lon"], z["lat"], z["radius_m"])
    return blocked


def nodes_to_lonlat(G, nodes) -> list:
    """Node ids -> [[lon, lat], ...] for the API's route "path" field."""
    return [[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in nodes]


def safe_route(G, origin_lonlat, dest_lonlat, blocked_nodes):
    """Emergency (ambulance) route avoiding blocked nodes. Returns a list of node ids, or None.

    The start and the incident (destination) are always allowed even if they sit inside a
    hazard zone, so an ambulance can still be sent TO an incident in a blocked area; every
    road node in between that is blocked is avoided. None = no safe route: tell the commander.
    """
    S = _simple(G)
    orig = nearest_node(S, *origin_lonlat)
    dest = nearest_node(S, *dest_lonlat)
    hazards = {n for n in blocked_nodes if n in S} - {orig, dest}
    try:
        return nx.shortest_path(nx.restricted_view(S, hazards, []), orig, dest, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def evacuation_route(G, origin_lonlat, assembly_points, blocked_nodes, risk_nodes=()):
    """Citizen evacuation: the nearest SAFE assembly point reachable without crossing a hazard.

    assembly_points: list of dicts {name, lon, lat}.  Returns (name, node list) or (None, None).
    blocked_nodes are impassable. Optional risk_nodes (the wider predicted risk zone) stay
    passable but cost RISK_PENALTY x more, so routes leave the zone quickly. Assembly points that
    are blocked or inside the risk zone are skipped. The start may be inside a hazard zone.
    """
    S = _simple(G)
    orig = nearest_node(S, *origin_lonlat)
    blocked = set(blocked_nodes)
    hazards = {n for n in blocked if n in S} - {orig}
    risky = {n for n in risk_nodes if n in S} - hazards
    candidates: dict = {}
    for ap in assembly_points:
        node = nearest_node(S, ap["lon"], ap["lat"])
        if node in blocked or node in risky:
            continue  # the assembly point itself is in a hazard / risk zone
        candidates.setdefault(node, ap["name"])

    def weight(u, v, d):
        return d["length"] * RISK_PENALTY if (u in risky or v in risky) else d["length"]

    dist, paths = nx.single_source_dijkstra(nx.restricted_view(S, hazards, []), orig, weight=weight)
    reachable = [n for n in candidates if n in dist]
    if not reachable:
        return None, None
    best = min(reachable, key=lambda n: dist[n])
    return candidates[best], paths[best]


if __name__ == "__main__":
    # Self-test on a small fake road grid (same checks as the playbook). Run: python api/routing.py
    G = nx.MultiDiGraph()
    N = 10
    for i in range(N):
        for j in range(N):
            G.add_node(i * N + j, x=18.40 + i * 0.01, y=-33.90 - j * 0.01)
    for i in range(N):
        for j in range(N):
            for di, dj in [(1, 0), (0, 1)]:
                a, b = i * N + j, (i + di) * N + (j + dj)
                if i + di < N and j + dj < N:
                    G.add_edge(a, b, length=1000)
                    G.add_edge(b, a, length=1000)
    start, end = (18.40, -33.90), (18.49, -33.99)
    normal = safe_route(G, start, end, [])
    blocked = normal[len(normal) // 2: len(normal) // 2 + 1]
    detour = safe_route(G, start, end, blocked)
    print("normal route nodes:", len(normal), "| detour nodes:", len(detour) if detour else None)
    assert detour and detour != normal and blocked[0] not in detour, "blocked node must change the route"

    fire_zone = [{"lon": 18.45, "lat": -33.95, "radius_m": 2500, "hazard": "fire"}]
    zone_nodes = nodes_in_zones(G, fire_zone)
    aps = [{"name": "Assembly A", "lon": 18.49, "lat": -33.99},
           {"name": "Assembly B", "lon": 18.40, "lat": -33.99}]
    name, path = evacuation_route(G, (18.41, -33.91), aps, zone_nodes)
    print("zone nodes:", len(zone_nodes), "| evacuate to:", name, "| nodes:", len(path) if path else None)
    assert path and not (set(path) & zone_nodes), "evacuation route must avoid the fire zone"
    print("path points for the dashboard:", nodes_to_lonlat(G, detour)[:2], "...")
    print("ALL ROUTING TESTS PASSED")
