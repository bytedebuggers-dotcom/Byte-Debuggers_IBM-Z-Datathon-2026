"""
Routing tests. Run from the repo root:   python -m pytest api/test_routing.py -v
They use the SYNTHETIC grid from build_demo_graph(), so no map data is needed.
"""
import networkx as nx

try:
    from api.routing import (build_demo_graph, evacuation_route, nearest_node, nodes_in_radius,
                             nodes_in_zones, nodes_to_lonlat, safe_route, safe_route_emergency,
                             safe_route_evacuation)
except ImportError:  # run directly from inside api/
    from routing import (build_demo_graph, evacuation_route, nearest_node, nodes_in_radius,
                         nodes_in_zones, nodes_to_lonlat, safe_route, safe_route_emergency,
                         safe_route_evacuation)

G = build_demo_graph()  # 7x7 grid, node id = row*7 + col
START = (18.42, -33.92)   # node 0   (top-left)
END = (18.45, -33.89)     # node 42  (bottom-right)


def test_blocked_node_changes_the_route():
    normal = safe_route_emergency(START, END, [], G=G)
    assert normal["status"] == "ok"
    mid = normal["route_nodes"][len(normal["route_nodes"]) // 2]
    detour = safe_route_emergency(START, END, [mid], G=G)
    assert detour["status"] == "ok"
    assert mid not in detour["route_nodes"]
    assert detour["route_nodes"] != normal["route_nodes"]
    assert detour["length_m"] >= normal["length_m"] - 1e-6
    assert len(detour["coords"]) == len(detour["route_nodes"])


def test_destination_cut_off_returns_no_route_not_exception():
    dest = nearest_node(G, *END)
    neighbours = set(G.predecessors(dest))
    res = safe_route_emergency(START, END, neighbours, G=G)
    # the destination snaps to a free node, but with all its neighbours blocked it is isolated
    assert res["status"] == "no_route"
    assert res["route_nodes"] is None and "Commander" in res["message"]


def test_blocked_origin_snaps_to_free_node():
    res = safe_route_emergency(START, END, [0], G=G)
    assert res["status"] == "ok"
    assert 0 not in res["route_nodes"]


def test_everything_blocked():
    res = safe_route_emergency(START, END, set(G.nodes), G=G)
    assert res["status"] == "no_route"


def test_nodes_in_radius_turns_a_circle_into_hazard_nodes():
    lon, lat = G.nodes[24]["x"], G.nodes[24]["y"]   # centre node
    zone = nodes_in_radius(G, lon, lat, 700)        # grid spacing is ~460-550 m
    assert 24 in zone and len(zone) > 1
    res = safe_route_emergency(START, END, zone, G=G)
    assert res["status"] == "ok" and not (set(res["route_nodes"]) & zone)


def test_evacuation_picks_nearest_reachable_assembly_point():
    origin = (G.nodes[24]["x"], G.nodes[24]["y"])
    near = (G.nodes[26]["x"], G.nodes[26]["y"])      # 2 blocks away
    far = (G.nodes[48 - 6]["x"], G.nodes[48 - 6]["y"])  # node 42, further
    res = safe_route_evacuation(origin, [far, near], G=G)
    assert res["status"] == "ok" and res["assembly_index"] == 1


def test_evacuation_avoids_blocked_path_and_switches_assembly_point():
    origin = (G.nodes[24]["x"], G.nodes[24]["y"])
    near = (G.nodes[25]["x"], G.nodes[25]["y"])
    far = (G.nodes[3]["x"], G.nodes[3]["y"])
    first = safe_route_evacuation(origin, [near, far], G=G)
    assert first["assembly_index"] == 0
    # block the near assembly point's whole neighbourhood so it cannot be reached
    res = safe_route_evacuation(origin, [near, far], hazard_nodes={25, 18, 32, 26}, G=G)
    assert res["status"] == "ok"
    assert res["assembly_index"] == 1
    assert not ({25, 18, 32, 26} & set(res["route_nodes"]))


def test_risk_zone_pushes_route_and_assembly_choice_outside_it():
    origin = (G.nodes[0]["x"], G.nodes[0]["y"])
    inside = (G.nodes[2]["x"], G.nodes[2]["y"])     # nearer, but we will mark it risky
    outside = (G.nodes[14]["x"], G.nodes[14]["y"])
    risky = {1, 2, 3, 8, 9, 10}
    res = safe_route_evacuation(origin, [inside, outside], risk_nodes=risky, G=G)
    assert res["status"] == "ok" and res["assembly_index"] == 1
    assert res["assembly_node"] not in risky


def test_evacuation_no_safe_assembly_point():
    origin = (G.nodes[24]["x"], G.nodes[24]["y"])
    target = (G.nodes[26]["x"], G.nodes[26]["y"])
    ring = set(G.predecessors(nearest_node(G, *origin)))  # wall in the origin
    res = safe_route_evacuation(origin, [target], hazard_nodes=ring, G=G)
    assert res["status"] == "no_route" and res["assembly_index"] is None


# ---- the three functions in docs/api_contract.md, on a raw OSM-style MultiDiGraph ----------
def fake_osm_grid(n=10):
    """Like what ox.load_graphml returns: a MultiDiGraph with x, y and length attributes."""
    M = nx.MultiDiGraph()
    for i in range(n):
        for j in range(n):
            M.add_node(i * n + j, x=18.40 + i * 0.01, y=-33.90 - j * 0.01)
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < n and j + dj < n:
                    a, b = i * n + j, (i + di) * n + (j + dj)
                    M.add_edge(a, b, length=1000)
                    M.add_edge(b, a, length=1000)
                    M.add_edge(a, b, length=1500)  # a parallel, longer edge
    return M


M = fake_osm_grid()
M_START, M_END = (18.40, -33.90), (18.49, -33.99)


def test_contract_safe_route_detours_around_blocked_node():
    normal = safe_route(M, M_START, M_END, [])
    blocked = normal[len(normal) // 2: len(normal) // 2 + 1]
    detour = safe_route(M, M_START, M_END, blocked)
    assert detour and detour != normal and blocked[0] not in detour


def test_contract_safe_route_reaches_an_incident_inside_a_hazard_zone():
    zone = [{"lon": 18.49, "lat": -33.99, "radius_m": 500, "hazard": "fire"}]  # covers only the incident node
    blocked = nodes_in_zones(M, zone)
    dest = nearest_node(M, *M_END)
    assert dest in blocked
    route = safe_route(M, M_START, M_END, blocked)
    assert route and route[-1] == dest          # the incident itself is still reachable
    # a wider zone that walls the incident in cannot be crossed: no safe route
    walled = nodes_in_zones(M, [{"lon": 18.49, "lat": -33.99, "radius_m": 2500, "hazard": "fire"}])
    assert safe_route(M, M_START, M_END, walled) is None


def test_contract_safe_route_returns_none_when_cut_off():
    dest = nearest_node(M, *M_END)
    wall = set(M.predecessors(dest))
    assert safe_route(M, M_START, M_END, wall) is None


def test_contract_evacuation_avoids_zone_and_names_the_assembly_point():
    zone_nodes = nodes_in_zones(M, [{"lon": 18.45, "lat": -33.95, "radius_m": 2500, "hazard": "fire"}])
    aps = [{"name": "Assembly A", "lon": 18.49, "lat": -33.99},
           {"name": "Assembly B", "lon": 18.40, "lat": -33.99}]
    name, path = evacuation_route(M, (18.41, -33.91), aps, zone_nodes)
    assert name in ("Assembly A", "Assembly B") and path and not (set(path) & zone_nodes)


def test_contract_evacuation_skips_blocked_assembly_point_and_handles_none():
    aps = [{"name": "Near", "lon": 18.42, "lat": -33.91}, {"name": "Far", "lon": 18.49, "lat": -33.99}]
    near_node = nearest_node(M, 18.42, -33.91)
    name, path = evacuation_route(M, (18.41, -33.91), aps, {near_node})
    assert name == "Far" and near_node not in path
    assert evacuation_route(M, (18.41, -33.91), [aps[0]], {near_node}) == (None, None)


def test_contract_evacuation_prefers_leaving_the_risk_zone():
    origin = (18.45, -33.95)
    aps = [{"name": "InsideRisk", "lon": 18.46, "lat": -33.95}, {"name": "Outside", "lon": 18.40, "lat": -33.95}]
    risk = nodes_in_zones(M, [{"lon": 18.45, "lat": -33.95, "radius_m": 1500, "hazard": "flood"}])
    name, _ = evacuation_route(M, origin, aps, set(), risk_nodes=risk)
    assert name == "Outside"


def test_contract_evacuation_from_inside_a_hazard_zone_can_leave_it():
    origin = (18.45, -33.95)                                   # centre of the fire zone
    zone = nodes_in_zones(M, [{"lon": 18.45, "lat": -33.95, "radius_m": 2500, "hazard": "fire"}])
    assert nearest_node(M, *origin) in zone                    # we really start inside the zone
    side_closure = nodes_in_zones(M, [{"lon": 18.40, "lat": -33.98, "radius_m": 400, "hazard": "flood"}])
    out_node = nearest_node(M, 18.40, -33.99)
    assert side_closure and out_node not in side_closure and not (side_closure & zone)
    name, path = evacuation_route(M, origin, [{"name": "Out", "lon": 18.40, "lat": -33.99}],
                                  zone | side_closure)
    assert name == "Out" and path
    assert path[-1] == out_node and path[-1] not in zone       # it gets out and ends outside the zone
    assert not (set(path) & side_closure)                      # the separate closure is still avoided


def test_nodes_to_lonlat_gives_lon_lat_pairs():
    route = safe_route(M, M_START, M_END, [])
    pts = nodes_to_lonlat(M, route)
    assert pts[0] == [18.40, -33.90] and pts[-1] == [18.49, -33.99]
    assert all(len(p) == 2 and -180 <= p[0] <= 180 and -90 <= p[1] <= 90 for p in pts)
