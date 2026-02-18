import networkx as nx
import numpy as np


def relabel_nodes(network: nx.Graph) -> nx.Graph:
    """
    Relabel the nodes of the network with consecutive integers.
    """

    unused_nodes = [u for u in network.nodes if network.degree(u) == 0]
    network.remove_nodes_from(unused_nodes)
    mapping = {
        old_label: new_label for new_label, old_label in enumerate(network.nodes)
    }
    network = nx.relabel_nodes(network, mapping)

    return network


def needs_relabeling(network: nx.Graph) -> bool:
    """
    Check if the nodes of the network need to be relabeled.
    """

    return list(network.nodes) != list(range(len(network.nodes)))

def remove_dangling_edges(network: nx.Graph):
    """
    Remove all edges that are not connected to two nodes, consecutively.
    """
    new_network = network.copy()

    # Remove dangling edges
    while True:
        dangling_edge = None

        for u, v in new_network.edges():
            if u == v:
                continue

            if new_network.degree(u) == 1 or new_network.degree(v) == 1:
                dangling_edge = (u, v)
                break

        if dangling_edge is None:
            break
        else:
            u, v = dangling_edge
            new_network.remove_edge(u, v)

    return relabel_nodes(new_network)


def has_dangling_edges(network: nx.Graph) -> bool:
    """
    Check if the network has dangling edges.
    """
    for u, v in network.edges():
        if u == v:
            continue
        if network.degree(u) == 1 or network.degree(v) == 1:
            return True
    return False

def extract_largest_connected_component(network: nx.Graph):
    """
    Extract the largest connected component of a network.

    Args:
        network (nx.Graph): The network.

    Returns:
        nx.Graph: The largest connected component of the network.
    """
    # Get connected components:
    connected_components = list(nx.connected_components(network))

    # Get largest connected component:
    largest_connected_component = max(connected_components, key=len)
    largest_connected_component_network = network.subgraph(largest_connected_component)

    return relabel_nodes(largest_connected_component_network.copy())

def replace_tiny_edges_by_nodes(network: nx.Graph, threshold: float = 0.01) -> nx.Graph:
    """
    Replace all edges with a length smaller than the threshold by a node.
    The node will have the same edges as the original node.
    """

    new_network = network.copy()
    node_count = len(new_network.nodes)

    while True:
        critical_edge = None

        for u, v in new_network.edges():
            if u == v:
                continue

            pos_u = new_network.nodes[u]["position"]
            pos_v = new_network.nodes[v]["position"]
            length = np.linalg.norm(np.array(pos_u) - np.array(pos_v))

            if length < threshold * 1.001:  # Allow a small tolerance
                critical_edge = (u, v, (pos_u + pos_v) / 2)
                break

        if critical_edge is None:
            break
        else:
            u, v, new_pos = critical_edge

            # Remove edge
            new_network.remove_edge(u, v)

            # Add new node
            new_node = node_count
            node_count += 1
            new_network.add_node(new_node, position=new_pos)

            # Add edges to new node
            neighbors = set(new_network.neighbors(u)) - {u}
            for neighbor in neighbors:
                new_network.add_edge(new_node, neighbor)
            neighbors = set(new_network.neighbors(v)) - {v}
            for neighbor in neighbors:
                new_network.add_edge(new_node, neighbor)

            # Remove old nodes
            try:
                new_network.remove_node(u)
                new_network.remove_node(v)
            except nx.NetworkXError:
                pass

    return relabel_nodes(new_network)

def resize_to_max_extent(network: nx.Graph, max_extent: float = 50):

    positions = np.array([node["position"] for node in network.nodes.values()])

    min_x = np.min(positions[:, 0])
    max_x = np.max(positions[:, 0])

    min_y = np.min(positions[:, 1])
    max_y = np.max(positions[:, 1])

    extent_x = max_x - min_x
    extent_y = max_y - min_y

    extent = max(extent_x, extent_y)
    ratio = max_extent / extent

    for u in network:
        pos = network.nodes[u]["position"]
        network.nodes[u]["position"] = np.array(
            [
                (pos[0] - min_x - extent_x / 2) * ratio,
                (pos[1] - min_y - extent_y / 2) * ratio,
            ]
        )
    
    return network