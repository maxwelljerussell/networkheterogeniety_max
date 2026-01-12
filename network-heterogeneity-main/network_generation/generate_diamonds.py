from typing import Union
import numpy as np
import networkx as nx
from classes.network_constraints import (
    NetworkConstraints,
    default_network_constraints,
)
from network_generation.quality_control.check_network import check_network
import traceback as tb


def generate_diamonds_network(
    n_rows: int = 5,
    n_cols: int = 5,
    std_rel: float = 0.3,
    constraints: NetworkConstraints = default_network_constraints,
    max_tries: Union[int, None] = 400,
) -> nx.Graph:
    tries = 0

    while max_tries is None or tries < max_tries:
        try:
            return _generate_diamonds_network(n_rows, n_cols, std_rel, constraints)
        except Exception as e:
            tries += 1

    raise ValueError("Could not generate network with given constraints.")


def _generate_diamonds_network(
    n_rows: int = 5,
    n_cols: int = 5,
    std_rel: float = 0.3,
    constraints: NetworkConstraints = default_network_constraints,
) -> nx.Graph:

    network = diamond_graph(
        n_rows=n_rows,
        n_cols=n_cols,
        edge_length_x=constraints.max_extent / n_cols,
        edge_length_y=constraints.max_extent / n_rows,
        std_rel=std_rel,
    )

    # Check if network is valid
    is_valid, reason = check_network(network, constraints)

    if is_valid:
        return network
    else:
        raise ValueError("Generated network does not meet the constraints.")


def truncated_normal(std, size):
    return np.tanh(np.random.normal(0, std, size=size))


def diamond_graph(n_rows, n_cols, edge_length_x, edge_length_y, std_rel=0.3):
    network = nx.Graph()

    for row in range(n_rows):
        for col in range(n_cols):
            x = (col - (n_cols - 1) / 2) * edge_length_x
            y = (row - (n_rows - 1) / 2) * edge_length_y
            index = 4 * (row * n_cols + col)
            random_x = truncated_normal(std_rel, 2) * edge_length_x / 4
            random_y = truncated_normal(std_rel, 2) * edge_length_y / 4

            network.add_node(
                index, position=np.array([x, y + edge_length_y / 4 + random_y[0]])
            )
            network.add_node(
                index + 1, position=np.array([x + edge_length_x / 4 + random_x[0], y])
            )
            network.add_node(
                index + 2, position=np.array([x, y - edge_length_y / 4 + random_y[1]])
            )
            network.add_node(
                index + 3, position=np.array([x - edge_length_x / 4 + random_x[1], y])
            )

            network.add_edge(index, index + 1)
            network.add_edge(index + 1, index + 2)
            network.add_edge(index + 2, index + 3)
            network.add_edge(index + 3, index)

            if row > 0:
                network.add_edge(index + 2, index - 4 * n_cols)
            if col > 0:
                network.add_edge(index + 3, index - 4 + 1)

    return network
