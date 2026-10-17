"""
Routing tests. Run from the repo root:   python -m pytest api/test_routing.py -v
They use the SYNTHETIC grid from build_demo_graph(), so no map data is needed.
"""
try:
    from api.routing import (build_demo_graph, nearest_node, nodes_in_radius,
                             safe_route_emergency, safe_route_evacuation)
except ImportError:  # run directly from inside api/
    from routing import (build_demo_graph, nearest_node, nodes_in_radius,
                         safe_route_emergency, safe_route_evacuation)

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
