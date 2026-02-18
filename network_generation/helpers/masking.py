from typing import Union, Literal
import copy
import networkx as nx
import numpy as np
from more_itertools import pairwise
from helpers.pixelation import pixelate_network
from network_generation.helpers.geometry import (
    get_portion_of_line_segment_within_circle,
    get_portion_of_line_segment_within_square,
)
from network_generation.helpers.postprocess import relabel_nodes


def _bool_array_to_freq_array(bool_array):
    """
    Convert a boolean array to a frequency array. For example:

    `[True, False, True, True, False] -> [{"freq": 1, "val": True}, {"freq": 1, "val": False}, {"freq": 2, "val": True}, {"freq": 1, "val": False}]`
    """
    freq_array = []
    for val in bool_array:
        if len(freq_array) == 0 or freq_array[-1]["val"] != val:
            freq_array.append({"freq": 1, "val": val})
        else:
            freq_array[-1]["freq"] += 1
    return freq_array


def _freq_array_to_bool_array(freq_array):
    """
    Convert a frequency array to a boolean array. For example:

    `[{"freq": 1, "val": True}, {"freq": 1, "val": False}, {"freq": 2, "val": True}, {"freq": 1, "val": False}] -> [True, False, True, True, False]`
    """
    bool_array = []
    for entry in freq_array:
        bool_array += [entry["val"]] * entry["freq"]
    return bool_array


def mask_network_to_arbitrary_shape(
    network: nx.Graph,
    mask: np.ndarray,
    remove_structure_below_pixel_size: Union[None, int] = None,
):
    # Mirror mask:
    mask = np.flipud(mask)

    width, height = mask.shape
    pixelated_network = pixelate_network(network, width, height, do_bookkeeping=True)
    pixel_assignment = pixelated_network.graph["pixel_assignment"]
    pixelated_edges = [edge for edge, _ in pixelated_network.edges.items()]

    should_keep_edges = np.zeros(len(pixelated_network.edges))
    for x in range(width):
        for y in range(height):
            if mask[y][x] == 0:
                continue

            for edge in pixel_assignment[y][x]:
                should_keep_edges[edge] = 1

    masked_network = pixelated_network.copy()
    bookkeeping_entries = pixelated_network.graph["original_edge_assignment"]
    for bookkeeping_entry in bookkeeping_entries:
        # Remove current edges from the network:
        for node1, node2 in pairwise(bookkeeping_entry):
            try:
                masked_network.remove_edge(node1, node2)
            except:
                masked_network.remove_edge(node2, node1)

        start_node = bookkeeping_entry[0]

        # Get edge indices:
        edge_indices = []
        for node1, node2 in pairwise(bookkeeping_entry):
            try:
                edge_indices.append(pixelated_edges.index((node1, node2)))
            except:
                edge_indices.append(pixelated_edges.index((node2, node1)))
        should_keep_edges_ = [
            should_keep_edges[edge_index] for edge_index in edge_indices
        ]

        # Remove small structure from the network:
        if remove_structure_below_pixel_size != None:
            freq_array = _bool_array_to_freq_array(should_keep_edges_)

            # Remove gaps:
            for i, entry in enumerate(freq_array[1:-1]):
                if entry["freq"] < remove_structure_below_pixel_size:
                    freq_array[i + 1]["val"] = True

            should_keep_edges_ = _freq_array_to_bool_array(freq_array)
            freq_array = _bool_array_to_freq_array(should_keep_edges_)

            # Remove islands:
            for i, entry in enumerate(freq_array[1:-1]):
                if entry["freq"] < remove_structure_below_pixel_size:
                    freq_array[i + 1]["val"] = False

            should_keep_edges_ = _freq_array_to_bool_array(freq_array)

        # Add edges to the network:
        for i, (node1, node2) in enumerate(pairwise(bookkeeping_entry)):
            keep_edge = should_keep_edges_[i]

            if keep_edge:
                if start_node == None:
                    start_node = node1
            else:
                if start_node != None:
                    if start_node != node1:
                        masked_network.add_edge(start_node, node1)
                    start_node = None

        if start_node != None:
            masked_network.add_edge(start_node, bookkeeping_entry[-1])

    return relabel_nodes(masked_network)


def mask_network(
    network: nx.Graph,
    radius: float,
    mask_type: Literal["circle", "square"] = "square",
    mode: Literal["stroke", "precision"] = "stroke",
    min_edge_length: Union[float, None] = None,
) -> nx.Graph:
    """
    Prune the  network to only include nodes and edges within a specified area.
    """

    # Stroke mode:
    if mode == "stroke":

        positions = [node["position"] for node in network.nodes.values()]

        for node, point in enumerate(positions):
            if mask_type == "circle":
                if np.linalg.norm(point) > radius:
                    network.remove_node(node)
            elif mask_type == "square":
                if np.abs(point[0]) > radius or np.abs(point[1]) > radius:
                    network.remove_node(node)
            else:
                raise ValueError("Invalid mask type.")

    # Precision mode:
    elif mode == "precision":

        if mask_type == "circle":

            edges = copy.deepcopy(list(network.edges))

            for u, v in edges:
                pos_u = network.nodes[u]["position"]
                pos_v = network.nodes[v]["position"]

                # Keep edge if both nodes are within the circle:
                if np.linalg.norm(pos_u) <= radius and np.linalg.norm(pos_v) <= radius:
                    continue
                else:
                    network.remove_edge(u, v)

                # Keep portion of edge that is within the circle:
                new_positions = get_portion_of_line_segment_within_circle(
                    np.array([0, 0]), radius, pos_u, pos_v
                )

                if new_positions == None:
                    continue

                # Add new nodes and edge:
                node1 = _add_node_if_not_existent(network, new_positions[0])
                node2 = _add_node_if_not_existent(network, new_positions[1])
                network.add_edge(node1, node2)

        elif mask_type == "square":

            edges = copy.deepcopy(list(network.edges))

            for u, v in edges:
                pos_u = network.nodes[u]["position"]
                pos_v = network.nodes[v]["position"]

                # Keep edge if both nodes are within the square:
                if (
                    np.abs(pos_u[0]) <= radius
                    and np.abs(pos_u[1]) <= radius
                    and np.abs(pos_v[0]) <= radius
                    and np.abs(pos_v[1]) <= radius
                ):
                    continue
                else:
                    network.remove_edge(u, v)

                # Keep portion of edge that is within the square:
                new_positions = get_portion_of_line_segment_within_square(
                    np.array([0, 0]), radius, pos_u, pos_v
                )

                if new_positions == None:
                    continue

                # Add new nodes and edge:
                node1 = _add_node_if_not_existent(network, new_positions[0])
                node2 = _add_node_if_not_existent(network, new_positions[1])
                network.add_edge(node1, node2)

        else:
            raise ValueError("Invalid mask type.")

    # Invalid mode:
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Remove dangling edges shorter than min_edge_length:
    if min_edge_length != None:
        for u, v in network.edges():
            if network.degree(u) > 1 and network.degree(v) > 1:
                continue

            pos_u = network.nodes[u]["position"]
            pos_v = network.nodes[v]["position"]
            length = np.linalg.norm(np.array(pos_u) - np.array(pos_v))

            if length < min_edge_length:
                network.remove_edge(u, v)

    return relabel_nodes(network)


def _add_node_if_not_existent(network: nx.Graph, position: np.ndarray):
    """
    Add a node to the network if it does not already exist.
    """
    for node, data in network.nodes(data=True):
        if np.allclose(data["position"], position):
            return node

    node = len(network.nodes)
    network.add_node(node, position=position)
    return node
