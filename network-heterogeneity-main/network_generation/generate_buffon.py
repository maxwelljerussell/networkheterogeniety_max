from typing import Union
import numpy as np
import networkx as nx
from network_generation.helpers.geometry import get_intersection_point
from network_generation.helpers.masking import mask_network
from classes.network_constraints import (
    NetworkConstraints,
    default_network_constraints,
)
from network_generation.quality_control.check_network import check_network
from network_generation.helpers.postprocess import (
    remove_dangling_edges,
)


def buffon_graph(
    num_needles: int = 20,
    constraints: NetworkConstraints = default_network_constraints,
    open: bool = False,
    seed: Union[int, None] = None,
    max_tries: Union[int, None] = 100,
):
    """
    Generate a Buffon graph.
    """

    tries = 0

    while max_tries is None or tries < max_tries:
        network = _buffon_graph(
            num_needles=num_needles,
            constraints=constraints,
            open=open,
            seed=seed,
        )

        tries += 1

        if check_network(network, constraints)[0]:
            return network

    raise ValueError("Could not generate a valid Buffon graph.")


def _buffon_graph(
    num_needles: int = 20,
    constraints: NetworkConstraints = default_network_constraints,
    open: bool = False,
    seed: Union[int, None] = None,
):
    """
    Generate a Buffon graph.
    """

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Initialize network
    nodes = []
    edges = []

    # Generate random needle centers and directions
    needle_centers = np.random.uniform(
        -constraints.max_extent / 2,
        constraints.max_extent / 2,
        (num_needles, 2),
    )
    needle_angles = np.random.uniform(0, np.pi, num_needles)
    needle_directions = np.array([np.cos(needle_angles), np.sin(needle_angles)]).T

    # Make sure the needles extend to the boundaries
    if open:
        additional_needle_centers = [
            [-constraints.max_extent, 0],
            [constraints.max_extent, 0],
            [0, constraints.max_extent],
            [0, -constraints.max_extent],
        ]
        additional_needle_directions = [
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ]
        needle_centers = np.concatenate(
            [needle_centers, additional_needle_centers], axis=0
        )
        needle_directions = np.concatenate(
            [needle_directions, additional_needle_directions], axis=0
        )

    # Iterate over all needles and check for intersections
    for i in range(num_needles):
        needle_center = needle_centers[i]
        needle_direction = needle_directions[i]

        intersection_points = []

        # Check for intersections with all other needles
        for j in range(num_needles):
            if i == j:
                continue

            other_needle_center = needle_centers[j]
            other_needle_direction = needle_directions[j]

            intersection = get_intersection_point(
                needle_center,
                needle_direction,
                other_needle_center,
                other_needle_direction,
            )

            intersection_points.append(intersection)

        # Sort intersection points by x-coordinate
        intersection_points = sorted(intersection_points, key=lambda x: x[0])

        # Add nodes to the network
        node_indices = []
        for point in intersection_points:
            if len(nodes) == 0:
                nodes.append(point)
                node_indices.append(0)
                continue

            distances = np.linalg.norm(np.array(nodes) - point, axis=1)
            min_distance = np.min(distances)

            if min_distance > constraints.min_edge_length:
                nodes.append(point)
                node_indices.append(len(nodes) - 1)

            else:
                node_indices.append(np.argmin(distances))

        # Add edges to the network
        for j in range(len(node_indices) - 1):
            if node_indices[j] != node_indices[j + 1]:
                edges.append((node_indices[j], node_indices[j + 1]))

    # Create network
    network = nx.Graph()
    network.add_nodes_from(range(len(nodes)))
    network.add_edges_from(edges)

    # Add positions to the nodes
    for u in network.nodes:
        network.nodes[u]["position"] = nodes[u]

    # Mask the network
    network = mask_network(
        network,
        constraints.max_extent / 2,
        mode="precision" if open else "stroke",
        min_edge_length=constraints.min_edge_length,
    )

    # Remove dangling edges if the network is not open
    if not open:
        network = remove_dangling_edges(network)

    return network
