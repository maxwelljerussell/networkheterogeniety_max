from typing import Union
import numpy as np
import networkx as nx
from classes.network_constraints import (
    NetworkConstraints,
    default_network_constraints,
)
from network_generation.quality_control.check_network import check_network
import traceback as tb
from network_generation.helpers.masking import mask_network
from network_generation.helpers.postprocess import (
    remove_dangling_edges,
)


def generate_hex_network(
    length: float = 1.0,
    std_rel: float = 0.3,
    constraints: NetworkConstraints = default_network_constraints,
    max_tries: Union[int, None] = 400,
) -> nx.Graph:
    tries = 0

    while max_tries is None or tries < max_tries:
        try:
            return _generate_hex_network(length, std_rel, constraints)
        except Exception as e:
            tries += 1

    raise ValueError("Could not generate network with given constraints.")


def _generate_hex_network(
    length: float = 1.0,
    std_rel: float = 0.3,
    constraints: NetworkConstraints = default_network_constraints,
) -> nx.Graph:

    n_cols = int(2 * constraints.max_extent / length + 1)
    n_rows = int(2 * constraints.max_extent / length * 2 / np.sqrt(3) + 1)

    network = hex_graph(
        n_rows=n_rows,
        n_cols=n_cols,
        length=length,
        std_rel=std_rel,
    )

    network = mask_network(network, constraints.max_extent / 2)
    network = remove_dangling_edges(network)

    # Check if network is valid
    is_valid, _ = check_network(network, constraints)

    if is_valid:
        return network
    else:
        raise ValueError("Generated network does not meet the constraints.")


def truncated_normal(std, size):
    return np.tanh(np.random.normal(0, std, size=size))


def hex_graph(
    n_rows: int = 5,
    n_cols: int = 5,
    length: float = 1.0,
    std_rel: float = 0.3,
) -> nx.Graph:

    # Define the lattice vectors for hexagonal grid
    a = np.array([[np.sqrt(3), 0], [np.sqrt(3) / 2, 3 / 2]]) / np.sqrt(3)

    # Generate displacement vectors for each boundary node of hexagon
    dr = np.array([[np.cos(n * np.pi / 3), np.sin(n * np.pi / 3)] for n in range(6)])

    network = nx.Graph()
    index = 0

    for i in range(n_cols):
        for j in range(n_rows):
            center_position = i * length * a[0] + j * length * a[1]
            center_position[0] -= (n_cols - 1) * length * 3 / 4
            center_position[1] -= (n_rows - 1) / 2 * length * np.sqrt(3) / 2

            size = (length / 3) + (length / 6) * truncated_normal(std_rel, 1)[0]

            for k, vector in enumerate(dr):
                position = center_position + size * vector

                network.add_node(index + k, position=position)
                network.add_edge(index + k, index + (k + 1) % 6)

            if i > 0:
                network.add_edge(index - 6 * n_rows, index + 3)
            if j > 0:
                network.add_edge(index - 5, index + 4)
            if i > 0 and j > 0:
                network.add_edge(index - 6 * n_rows + 5, index - 4)

            index += 6

    return network
