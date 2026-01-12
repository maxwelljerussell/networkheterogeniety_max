from typing import Tuple
import numpy as np
import networkx as nx
from network_generation.helpers.constraints_helper import get_network_extent
from network_generation.helpers.geometry import (
    do_line_segments_intersect,
    get_distance_between_point_and_line_segment,
)
from classes.network_constraints import (
    NetworkConstraints,
    default_network_constraints,
)


def check_network(
    network: nx.Graph,
    constraints: NetworkConstraints = default_network_constraints,
    return_all_issues: bool = False,
) -> Tuple[bool, str]:

    issues = []

    # Check if network is within max extent:
    extent = get_network_extent(network)
    if extent - 1e-6 > constraints.max_extent:
        issues.append(f"Network extent violation, extent: {extent}")
        if not return_all_issues:
            return False, issues[0]

    # Check edge lengths:
    for u, v, d in network.edges(data=True):
        pos_u = network.nodes[u]["position"]
        pos_v = network.nodes[v]["position"]
        length = np.linalg.norm(np.array(pos_u) - np.array(pos_v))

        d["length"] = length
    min_length = min([d["length"] for _, _, d in network.edges(data=True)])
    if min_length < constraints.min_edge_length:
        issues.append(f"Minimum edge length violation, length: {min_length}")
        if not return_all_issues:
            return False, issues[0]

    # Check node count:
    node_count = len(network.nodes)
    if node_count > constraints.max_node_count:
        issues.append(f"Maximum node count violation, count: {node_count}")
        if not return_all_issues:
            return False, issues[0]

    # Check edge count:
    edge_count = len(network.edges)
    if edge_count > constraints.max_edge_count:
        issues.append(f"Maximum edge count violation, count: {edge_count}")
        if not return_all_issues:
            return False, issues[0]

    # Check edge angles:
    for u in network.nodes:
        neighbors = network.neighbors(u)
        vectors = [
            network.nodes[v]["position"] - network.nodes[u]["position"]
            for v in neighbors
        ]
        vectors = [v / np.linalg.norm(v) for v in vectors if np.linalg.norm(v) > 0]

        for v1 in vectors:
            for v2 in vectors:
                if np.array_equal(v1, v2):
                    continue

                if np.dot(v1, v2) > np.cos(np.radians(constraints.min_edge_angle)):
                    angle = np.degrees(np.arccos(np.dot(v1, v2)))
                    issues.append(f"Edge angle violation, angle: {angle} degrees")
                    if not return_all_issues:
                        return False, issues[0]

    # Check distance between nodes and edges:
    for u in network.nodes:
        for v, w in network.edges:
            if u == v or u == w:
                continue

            # Make exception for dangling nodes:
            if (
                len(list(network.neighbors(u))) <= 1
                or len(list(network.neighbors(v))) <= 1
            ):
                continue

            distance = get_distance_between_point_and_line_segment(
                network.nodes[u]["position"],
                network.nodes[v]["position"],
                network.nodes[w]["position"],
            )
            if distance < constraints.min_distance_between_node_and_edge:
                issues.append(
                    f"Minimum distance between node and edge violation, distance: {distance}"
                )
                if not return_all_issues:
                    return False, issues[0]

    # Check for intersections:
    for u, v in network.edges:
        for w, x in network.edges:
            if u == w or u == x or v == w or v == x:
                continue

            if do_line_segments_intersect(
                network.nodes[u]["position"],
                network.nodes[v]["position"],
                network.nodes[w]["position"],
                network.nodes[x]["position"],
            ):
                issues.append("Edge intersection violation")
                if not return_all_issues:
                    return False, issues[0]

    # Check node degree:
    max_degree = max([len(list(network.neighbors(u))) for u in network.nodes])
    if max_degree > constraints.max_node_degree:
        issues.append(f"Maximum node degree violation, degree: {max_degree}")
        if not return_all_issues:
            return False, issues[0]

    # Check if there are any issues:
    if len(issues) > 0:
        return False, "\n".join(issues)
    else:
        return True, "Network is valid"
