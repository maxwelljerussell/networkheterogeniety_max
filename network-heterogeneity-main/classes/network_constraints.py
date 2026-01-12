import numpy as np

class NetworkConstraints:
    """
    Class to store contraints made to a network.
    """

    def __init__(
        self,
        max_extent: float = 50,
        min_edge_length: float = 1.5,
        max_node_count: int = 100,
        max_edge_count: int = 200,
        min_edge_angle: float = 5,  # degrees
        max_node_degree: int = 7,
        min_distance_between_node_and_edge: float = 0.5,
    ):
        if min_distance_between_node_and_edge > min_edge_length:
            raise ValueError(
                "min_distance_between_node_and_edge and min_edge_length are not compatible."
            )

        self.max_extent = max_extent
        self.min_edge_length = min_edge_length
        self.max_node_count = max_node_count
        self.max_edge_count = max_edge_count
        self.min_edge_angle = min_edge_angle
        self.max_node_degree = max_node_degree
        self.min_distance_between_node_and_edge = min_distance_between_node_and_edge


default_network_constraints = NetworkConstraints()
default_network_constraints_2 = NetworkConstraints(
    max_edge_count=np.inf, max_node_count=np.inf
)
