import copy
import numpy as np
from scipy.spatial import Delaunay
import networkx as nx
from classes.network_constraints import (
    NetworkConstraints,
    default_network_constraints,
)
from network_generation.generate_random import generate_random_points
from network_generation.helpers.masking import mask_network
from network_generation.quality_control.check_network import check_network
from network_generation.helpers.clustered_points import generate_clustered_points
from network_generation.helpers.postprocess import (
    remove_dangling_edges,
)


def delaunay_graph_from_points(points: list):
    """
    Generate a Delaunay graph from points.
    """
    delaunay = Delaunay(points)

    # Create an empty graph
    network = nx.Graph()

    # Iterate over the simplices (triangles)
    for simplex in delaunay.simplices:
        # Add edges for each pair of vertices in the simplex
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                network.add_edge(simplex[i], simplex[j])

    network_final = nx.Graph()
    network_final.add_nodes_from(sorted(network.nodes(data=True)))
    network_final.add_edges_from(network.edges(data=True))

    # Add the positions to the nodes
    for idx, point in enumerate(points):
        network_final.nodes[idx]["position"] = point

    return network_final


def delaunay_graph(
    num_of_nodes: int = 60,
    constraints: NetworkConstraints = default_network_constraints,
    open: bool = False,
    max_tries: int = 400,
):
    """
    Generate a Delaunay graph from random points.
    """

    # Handle expand to boundaries
    EXPANSION_FACTOR = 2
    _constraints = copy.deepcopy(constraints)
    _constraints.max_extent *= EXPANSION_FACTOR
    num_of_nodes = np.floor(num_of_nodes * EXPANSION_FACTOR**2)

    for _ in range(max_tries):

        # Generate random points
        points = generate_random_points(
            num_of_nodes=num_of_nodes, constraints=_constraints
        )

        # Generate network
        network = delaunay_graph_from_points(points)

        # Meet max node degree constraint
        while True:
            max_degree = max(dict(network.degree()).values())
            if max_degree <= constraints.max_node_degree:
                break

            # Get the node with the highest degree
            node = max(dict(network.degree()).items(), key=lambda x: x[1])[0]

            # Get the neighbors of the node
            neighbors = list(network.neighbors(node))

            # Get the positions of the node and its neighbors
            node_pos = network.nodes[node]["position"]
            neighbor_positions = [
                network.nodes[neighbor]["position"] for neighbor in neighbors
            ]

            # Calculate the distances between the node and its neighbors
            distances = [
                np.linalg.norm(node_pos - neighbor_pos)
                for neighbor_pos in neighbor_positions
            ]

            # Get the neighbor with the smallest distance
            neighbor = neighbors[np.argmin(distances)]

            # Remove the edge between the node and its neighbor
            network.remove_edge(node, neighbor)

        # Mask network
        network = mask_network(
            network,
            constraints.max_extent / 2,
            mode="precision" if open else "stroke",
            min_edge_length=constraints.min_edge_length,
        )

        # Remove dangling edges
        if not open:
            network = remove_dangling_edges(network)

        # Check network
        valid, _ = check_network(network, constraints, return_all_issues=True)
        if valid:
            return network

    raise ValueError("Could not generate a network within the given constraints.")

def delaunay_graph_clustered(
    num_of_nodes: int = 60,
    constraints: NetworkConstraints = default_network_constraints,
    open: bool = False,
    max_tries: int = 400,
    seed: int = 0,
    n_clusters: int = 4,
    cluster_std: float = 0.08,
):
    """
    Same as delaunay_graph, but uses clustered points.
    Does NOT change original delaunay_graph behavior.
    """
    EXPANSION_FACTOR = 2
    _constraints = copy.deepcopy(constraints)
    _constraints.max_extent *= EXPANSION_FACTOR
    num_of_nodes_eff = int(np.floor(num_of_nodes * EXPANSION_FACTOR**2))

    for it in range(max_tries):
        points = generate_clustered_points(
            num_points=num_of_nodes_eff,
            extent=_constraints.max_extent,
            n_clusters=n_clusters,
            cluster_std=cluster_std,
            seed=seed + it,
        )

        network = delaunay_graph_from_points(points)

        # Meet max node degree constraint (unchanged)
        while True:
            max_degree = max(dict(network.degree()).values())
            if max_degree <= constraints.max_node_degree:
                break

            node = max(dict(network.degree()).items(), key=lambda x: x[1])[0]
            neighbors = list(network.neighbors(node))

            node_pos = network.nodes[node]["position"]
            neighbor_positions = [network.nodes[neighbor]["position"] for neighbor in neighbors]
            distances = [np.linalg.norm(node_pos - neighbor_pos) for neighbor_pos in neighbor_positions]

            neighbor = neighbors[int(np.argmin(distances))]
            network.remove_edge(node, neighbor)

        network = mask_network(
            network,
            constraints.max_extent / 2,
            mode="precision" if open else "stroke",
            min_edge_length=constraints.min_edge_length,
        )

        if not open:
            network = remove_dangling_edges(network)
        
        # Guard against empty / edge-less graphs (avoid numpy min([]))
        if network.number_of_nodes() == 0 or network.number_of_edges() == 0:
            continue


        valid, _ = check_network(network, constraints, return_all_issues=True)
        if valid:
            return network

    raise ValueError("Could not generate clustered Delaunay network within the given constraints.")

