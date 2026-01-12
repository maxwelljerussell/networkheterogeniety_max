from typing import Tuple
import networkx as nx
import numpy as np
from classes.network_constraints import (
    NetworkConstraints,
)
from network_generation.helpers.geometry import (
    do_line_segments_intersect,
    get_distance_between_point_and_line_segment,
)


def get_network_extent(network: nx.Graph) -> float:
    positions = np.array([node["position"] for node in network.nodes.values()])
    min_x, min_y = np.min(positions, axis=0)
    max_x, max_y = np.max(positions, axis=0)
    extent_x, extent_y = max_x - min_x, max_y - min_y
    extent = max(extent_x, extent_y)

    return extent


def exists_angle_violation(
    network: nx.Graph, u: int, v: int, constraints: NetworkConstraints
) -> bool:
    return not (
        (not _exists_angle_violation(network, u, v, constraints))
        and (not _exists_angle_violation(network, v, u, constraints))
    )


def _exists_angle_violation(
    network: nx.Graph, u: int, v: int, constraints: NetworkConstraints
) -> bool:
    angle_violation = False

    neighbors = network.neighbors(u)

    vectors = [
        network.nodes[w]["position"] - network.nodes[u]["position"] for w in neighbors
    ]
    vectors = [w / np.linalg.norm(w) for w in vectors if np.linalg.norm(w) > 0]

    new_vector = network.nodes[v]["position"] - network.nodes[u]["position"]
    new_vector = new_vector / np.linalg.norm(new_vector)

    for w in vectors:
        if np.dot(w, new_vector) > np.cos(np.radians(constraints.min_edge_angle)):
            angle_violation = True
            break

    return angle_violation


def can_add_edge(
    network: nx.Graph,
    u: int,
    v: int,
    constraints: NetworkConstraints,
    debug: bool = False,
) -> bool:
    u_pos = network.nodes[u]["position"]
    v_pos = network.nodes[v]["position"]

    # Return false if u == v or edge already exists
    if u == v or network.has_edge(u, v):
        if debug:
            print("u == v or edge already exists")
        return False

    # Return false if node degree is too high
    if (
        network.degree(u) >= constraints.max_node_degree
        or network.degree(v) >= constraints.max_node_degree
    ):
        if debug:
            print("node degree is too high")
        return False

    # Return false if edge length is too short
    if np.linalg.norm(u_pos - v_pos) < constraints.min_edge_length:
        if debug:
            print("edge length is too short")
        return False

    # Return false if edge count is too high
    if len(network.edges) >= constraints.max_edge_count:
        if debug:
            print("edge count is too high")
        return False

    # Return false if angle between edges is too small
    if exists_angle_violation(network, u, v, constraints):
        if debug:
            print("angle between edges is too small")
        return False

    # Return false if minimal distance between node and edge is too small
    distance_violoation = False
    for w in network.nodes:
        if u == w or v == w:
            continue

        w_pos = network.nodes[w]["position"]

        if (
            get_distance_between_point_and_line_segment(w_pos, u_pos, v_pos)
            < constraints.min_distance_between_node_and_edge
        ):
            distance_violoation = True
            break
    if distance_violoation:
        if debug:
            print("Minimal distance between node and edge is too small")
        return False

    # Return false if edge intersects with any other edge
    intersection = False
    for w, x in network.edges:
        if u == w or u == x or v == w or v == x:
            continue

        w_pos = network.nodes[w]["position"]
        x_pos = network.nodes[x]["position"]

        if do_line_segments_intersect(u_pos, v_pos, w_pos, x_pos):
            intersection = True
            break
    if intersection:
        if debug:
            print("edge intersects with any other edge")
        return False

    return True


def get_network_size(network: nx.Graph) -> Tuple[float, int, int]:
    """
    Calculate the total edge length, number of edges and number of nodes in the network.
    """

    total_length = sum(
        np.linalg.norm(
            np.array(network.nodes[u]["position"])
            - np.array(network.nodes[v]["position"])
        )
        for u, v in network.edges
    )
    num_edges = len(network.edges)
    num_nodes = len(network.nodes)

    return total_length, num_edges, num_nodes