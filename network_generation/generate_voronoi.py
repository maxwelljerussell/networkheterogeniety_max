from typing import Tuple, Union
import copy
import numpy as np
import networkx as nx
from scipy.spatial import Voronoi
from classes.network_metadata import NetworkMetadata
from classes.network_constraints import (
    NetworkConstraints,
    default_network_constraints,
)
from network_generation.helpers.masking import mask_network
from network_generation.helpers.create_grids import create_triangle_grid
from network_generation.helpers.clustered_points import generate_clustered_points
from network_generation.helpers.postprocess import (
    replace_tiny_edges_by_nodes,
    remove_dangling_edges,
    extract_largest_connected_component
)

from network_generation.quality_control.check_network import check_network


def add_disorder(points_list, alpha=0.8, seed=10):
    """
    Add small disorder to the coordinate positions.
    """

    np.random.seed(seed)

    # Add small randomness to the points
    rnd = np.random.rand(len(points_list), 2) - 0.5
    points_list = points_list + alpha * rnd

    return points_list


def voronoi_graph_from_points(points: list):
    """
    Generate a Voronoi graph from points.
    """

    vor = Voronoi(points)

    network = nx.Graph()

    for ridge in vor.ridge_vertices:
        # If the ridge is finite, add an edge
        if ridge[0] >= 0 and ridge[1] >= 0:
            network.add_edge(ridge[0], ridge[1])

    network_final = nx.Graph()
    network_final.add_nodes_from(sorted(network.nodes(data=True)))
    network_final.add_edges_from(network.edges(data=True))

    # Add the positions to the nodes:
    for u in network_final.nodes:
        network_final.nodes[u]["position"] = vor.vertices[u]

    return network_final


def generate_voronoi_network_with_small_disorder(
    radius=1.0,
    grid_size=3,
    disorder_value=0,
    seed=10,
    prune_radius: Union[int, None] = None,
) -> Tuple[nx.Graph, NetworkMetadata]:
    """
    Generate a Voronoi graph.
    """

    # Generate hexagonal grid points
    hex_grid_points = create_triangle_grid(radius, grid_size)

    # Add small random displacement to the points
    points = add_disorder(hex_grid_points, alpha=disorder_value, seed=seed)

    # Generate Voronoi diagram
    network = voronoi_graph_from_points(points)

    # Prune the graph to only include nodes and edges within a specified area
    if prune_radius is not None:
        network = mask_network(network, mask="square", radius=prune_radius)

    # Create metadata object:
    metadata = NetworkMetadata(
        generation_method="voronoi",
        generation_parameters={
            "radius": radius,
            "grid_size": grid_size,
            "disorder_value": disorder_value,
            "rseed": seed,
            "prune_radius": prune_radius,
        },
    )

    return network, metadata


def generate_voronoi_network(
    num_of_points: int = 60,
    constraints: NetworkConstraints = default_network_constraints,
    max_iterations: int = 100,
    max_tries: Union[int, None] = 100,
    open: bool = False,
) -> nx.Graph:
    tries = 0

    while max_tries is None or tries < max_tries:
        try:
            return _generate_voronoi_network(
                num_of_points=num_of_points,
                constraints=constraints,
                max_iterations=max_iterations,
                open=open,
            )
        except:
            tries += 1

    raise ValueError("Could not generate network with given constraints.")


def _generate_voronoi_network(
    num_of_points: int = 60,
    constraints: NetworkConstraints = default_network_constraints,
    max_iterations: int = 100,
    open: bool = False,
) -> nx.Graph:

    EXPANSION_FACTOR = 30 / 25

    effective_constraints = copy.deepcopy(constraints)
    effective_constraints.max_extent *= EXPANSION_FACTOR
    effective_constraints.max_node_count *= EXPANSION_FACTOR**2

    for _ in range(max_iterations):
        # Generate random network
        points = np.random.uniform(
            -effective_constraints.max_extent / 2,
            effective_constraints.max_extent / 2,
            (num_of_points, 2),
        )
        points = np.array(points)
        network = voronoi_graph_from_points(points)

        # Make changes to network
        network = mask_network(
            network,
            constraints.max_extent / 2,
            mode="precision" if open else "stroke",
            min_edge_length=constraints.min_edge_length,
        )
        network = replace_tiny_edges_by_nodes(
            network, threshold=constraints.min_edge_length
        )
        if not open:
            network = remove_dangling_edges(network)
        network = extract_largest_connected_component(network)

        # Check if network is valid
        is_valid, _ = check_network(network, constraints)

        if is_valid:
            return network

    raise ValueError("Could not generate network with given constraints.")

def generate_voronoi_network_clustered(
    num_of_points: int = 60,
    constraints: NetworkConstraints = default_network_constraints,
    max_iterations: int = 100,
    max_tries: Union[int, None] = 100,
    open: bool = False,
    seed: int = 0,
    n_clusters: int = 4,
    cluster_std: float = 0.08,
) -> nx.Graph:
    """
    Same as generate_voronoi_network, but seeds are clustered (mixture of Gaussians).
    Does NOT change original generate_voronoi_network behavior.
    """
    tries = 0
    while max_tries is None or tries < max_tries:
        try:
            return _generate_voronoi_network_clustered(
                num_of_points=num_of_points,
                constraints=constraints,
                max_iterations=max_iterations,
                open=open,
                seed=seed,
                n_clusters=n_clusters,
                cluster_std=cluster_std,
            )
        except Exception:
            tries += 1
    raise ValueError("Could not generate clustered Voronoi network with given constraints.")


def _generate_voronoi_network_clustered(
    num_of_points: int,
    constraints: NetworkConstraints,
    max_iterations: int,
    open: bool,
    seed: int,
    n_clusters: int,
    cluster_std: float,
) -> nx.Graph:
    EXPANSION_FACTOR = 30 / 25

    effective_constraints = copy.deepcopy(constraints)
    effective_constraints.max_extent *= EXPANSION_FACTOR
    effective_constraints.max_node_count *= EXPANSION_FACTOR**2

    for it in range(max_iterations):
        # clustered points inside expanded box
        points = generate_clustered_points(
            num_points=num_of_points,
            extent=effective_constraints.max_extent,
            n_clusters=n_clusters,
            cluster_std=cluster_std,
            seed=seed + it,  # vary per attempt
        )

        network = voronoi_graph_from_points(points)

        network = mask_network(
            network,
            constraints.max_extent / 2,
            mode="precision" if open else "stroke",
            min_edge_length=constraints.min_edge_length,
        )
        network = replace_tiny_edges_by_nodes(network, threshold=constraints.min_edge_length)
        if not open:
            network = remove_dangling_edges(network)
        network = extract_largest_connected_component(network)

        is_valid, _ = check_network(network, constraints)
        if is_valid:
            return network

    raise ValueError("Could not generate clustered Voronoi network.")
