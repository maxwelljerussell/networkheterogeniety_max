from typing import Union
import copy
import networkx as nx
import numpy as np
from classes.network_constraints import (
    NetworkConstraints,
    default_network_constraints,
)
from network_generation.helpers.constraints_helper import can_add_edge
from network_generation.helpers.masking import mask_network
from network_generation.helpers.postprocess import (
    remove_dangling_edges,
)


def generate_random_points(
    num_of_nodes: int = 50,
    constraints: NetworkConstraints = default_network_constraints,
):

    positions = [
        np.random.uniform(-constraints.max_extent / 2, constraints.max_extent / 2, 2)
    ]
    while len(positions) < num_of_nodes:
        position = np.random.uniform(
            -constraints.max_extent / 2, constraints.max_extent / 2, 2
        )
        distances = np.linalg.norm(np.array(positions) - position, axis=1)

        if np.min(distances) >= constraints.min_edge_length:
            positions.append(position)

    return positions


def generate_random_network(
    num_of_nodes: int = 50,
    num_of_edges: int = 105,
    length_scale: float = 2.5,
    constraints: NetworkConstraints = default_network_constraints,
    max_iterations: int = 5000,
    max_tries: Union[int, None] = 10,
    open: bool = False,
) -> nx.Graph:
    """
    Generate a random network with the given number of nodes and edges.

    Args:
        num_of_nodes: The number of nodes in the network.
        num_of_edges: The number of edges in the network.
        max_extent: The maximum extent of the network.

    Returns:
        A networkx graph representing the random network.
    """

    tries = 0

    while max_tries is None or tries < max_tries:
        try:
            return _generate_random_network(
                num_of_nodes=num_of_nodes,
                num_of_edges=num_of_edges,
                length_scale=length_scale,
                constraints=constraints,
                max_iterations=max_iterations,
                open=open,
            )
        except:
            tries += 1

    raise ValueError("Could not generate network with given constraints.")


def _generate_random_network(
    num_of_nodes: int = 50,
    num_of_edges: int = 105,
    length_scale: float = 2.5,
    constraints: NetworkConstraints = default_network_constraints,
    max_iterations: int = 5000,
    open: bool = False,
) -> nx.Graph:

    # Handle expand to boundaries
    EXPANSION_FACTOR = 1.2
    _constraints = copy.deepcopy(constraints)
    if open:
        _constraints.max_extent *= EXPANSION_FACTOR
        num_of_nodes = np.floor(num_of_nodes * EXPANSION_FACTOR**2)
        num_of_edges = np.floor(num_of_edges * EXPANSION_FACTOR**2)

    # Add nodes
    positions = generate_random_points(num_of_nodes, _constraints)

    network = nx.Graph()
    for i, pos in enumerate(positions):
        network.add_node(
            i,
            position=pos,
        )

    # Add edges
    iterations = 0
    while len(network.edges) < num_of_edges:
        # Keep track of iterations
        iterations += 1
        if iterations > max_iterations:
            raise ValueError("Could not generate network with given constraints.")

        # Choose first node
        u = np.random.choice(list(network.nodes))

        # Calculate possible edge lengths
        possible_edge_lengths = np.array(
            [
                np.linalg.norm(
                    network.nodes[u]["position"] - network.nodes[v]["position"]
                )
                for v in network.nodes
            ]
        )

        # Choose second node
        probabilities = np.exp(-possible_edge_lengths / length_scale)
        probabilities[possible_edge_lengths < constraints.min_edge_length] = 0
        for v in network.neighbors(u):
            probabilities[v] = 0
        probabilities /= np.sum(probabilities)
        v = np.random.choice(list(network.nodes), p=probabilities)

        # Skip if edge does not satisfy constraints
        if not can_add_edge(network, u, v, _constraints):
            continue

        # Add edge
        network.add_edge(u, v)

    # Mask network
    if open:
        network = mask_network(
            network,
            constraints.max_extent / 2,
            mode="precision" if open else "stroke",
            min_edge_length=constraints.min_edge_length,
        )

    # Remove dangling edges
    if not open:
        network = remove_dangling_edges(network)

    # Make sure the network is connected
    if not nx.is_connected(network):
        raise ValueError("Could not generate network with given constraints.")

    return network
