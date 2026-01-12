import networkx as nx


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
