from typing import Tuple
import numpy as np
import copy
import networkx as nx
from network_generation.helpers.constraints_helper import get_network_extent
from network_generation.helpers.geometry import (
    do_line_segments_intersect,
    get_distance_between_point_and_line_segment,
)
from classes.network_constraints import (
    NetworkConstraints,
    default_network_constraints,
)
from network_generation.quality_control.check_network import check_network


def _get_total_edge_length(network: nx.Graph) -> float:
    """Calculate the total edge length of the network based on node positions."""

    total_length = 0.0
    for u, v, d in network.edges(data=True):
        pos_u = network.nodes[u]["position"]
        pos_v = network.nodes[v]["position"]
        length = np.linalg.norm(np.array(pos_u) - np.array(pos_v))
        total_length += length

    return total_length


def renormalize_network(
    network: nx.Graph,
    min_total_edge_length: float,
    max_total_edge_length: float,
    #
    constraints: NetworkConstraints = default_network_constraints,
    #
    step_size: float = 0.1,
    max_iterations: int = 1000,
    max_iterations_for_constraint_matching: int = 1000,
) -> nx.Graph:
    # Check validity of inputs
    assert (
        min_total_edge_length < max_total_edge_length
    ), "Minimum total edge length must be less than maximum total edge length."

    # Check initial network validity
    is_valid, issue = check_network(network, constraints)
    if not is_valid:
        raise ValueError(f"Network is not valid: {issue}")

    # Create a copy of the network to avoid modifying the original
    original_network = network
    network = copy.deepcopy(network)
    for node in network.nodes:
        network.nodes[node]["position"] = np.array(network.nodes[node]["position"])
    num_nodes = len(network.nodes)

    # Start iterations:
    for iteration_count in range(1, max_iterations + 1):
        # Compare current total edge length with desired range
        total_edge_length = _get_total_edge_length(network)

        if (
            total_edge_length >= min_total_edge_length
            and total_edge_length <= max_total_edge_length
        ):
            # Check if the network is still valid
            is_valid, _ = check_network(network, constraints)

            if is_valid:
                return network
            else:
                # Restart the renormalization process with a smaller step size
                return renormalize_network(
                    original_network,
                    min_total_edge_length,
                    max_total_edge_length,
                    constraints,
                    step_size / 2,
                    max_iterations - iteration_count,
                    max_iterations_for_constraint_matching,
                )
        elif total_edge_length < min_total_edge_length:
            factor = -1
        else:
            factor = +1

        # Calculate the incremental adjustments needed
        dx = np.zeros((num_nodes, 2))

        for i, node in enumerate(network.nodes):
            pos1 = network.nodes[node]["position"]

            for neighbor in network.neighbors(node):
                pos2 = network.nodes[neighbor]["position"]

                dx[i] += (
                    factor * step_size * (pos2 - pos1) / np.linalg.norm(pos2 - pos1)
                )

        # Apply the adjustments to the node positions
        for i, node in enumerate(network.nodes):
            new_position = network.nodes[node]["position"] + dx[i]
            network.nodes[node]["position"] = new_position

        # Make sure the new positions are within the constraints
        for count in range(max_iterations_for_constraint_matching):
            changed_network = False

            # Maximum extent:
            for node in network.nodes:
                pos = network.nodes[node]["position"]
                for i in range(2):
                    if pos[i] < -constraints.max_extent:
                        pos[i] = -constraints.max_extent
                        changed_network = True
                    elif pos[i] > constraints.max_extent:
                        pos[i] = constraints.max_extent
                        changed_network = True

            # Minimum distance between nodes:
            for u, v in network.edges:
                pos_u = network.nodes[u]["position"]
                pos_v = network.nodes[v]["position"]
                distance = np.linalg.norm(pos_u - pos_v)

                if distance < constraints.min_edge_length:
                    # Move the nodes apart
                    direction = (pos_v - pos_u) / distance
                    move_distance = max(
                        0.01, (constraints.min_edge_length - distance) / 2 * 1.01
                    )
                    network.nodes[u]["position"] -= direction * move_distance
                    network.nodes[v]["position"] += direction * move_distance
                    changed_network = True

            # Minimum distance between nodes and edges:
            for u in network.nodes:
                pos_u = network.nodes[u]["position"]

                for v, w in network.edges:
                    if u == v or u == w:
                        continue

                    # Make exception for dangling nodes:
                    if len(list(network.neighbors(u))) <= 1:
                        continue

                    pos_v = network.nodes[v]["position"]
                    pos_w = network.nodes[w]["position"]

                    # Determine the position of u's projection on the line segment vw
                    line_vector = pos_w - pos_v
                    position_on_line_segment = (
                        np.dot(pos_u - pos_v, line_vector)
                        / np.linalg.norm(line_vector) ** 2
                    )
                    position_on_line_segment = np.clip(position_on_line_segment, 0, 1)
                    closest_point = pos_v + position_on_line_segment * line_vector
                    distance = np.linalg.norm(pos_u - closest_point)

                    if distance < constraints.min_distance_between_node_and_edge:
                        # Move u away from the closest point
                        direction = (pos_u - closest_point) / distance
                        move_distance = max(
                            0.01,
                            (constraints.min_distance_between_node_and_edge - distance)
                            * 1.01,
                        )
                        network.nodes[u]["position"] += direction * move_distance
                        changed_network = True

            # Minimum angle between edges:
            for node in network.nodes:
                neighbors = list(network.neighbors(node))
                num_neighbors = len(neighbors)

                for i in range(num_neighbors):
                    for j in range(i + 1, num_neighbors):
                        u = neighbors[i]
                        v = neighbors[j]

                        pos_node = network.nodes[node]["position"]
                        pos_u = network.nodes[u]["position"]
                        pos_v = network.nodes[v]["position"]

                        vec1 = pos_u - pos_node
                        vec2 = pos_v - pos_node

                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)

                        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                        angle = np.arccos(cos_angle)

                        if angle < np.radians(constraints.min_edge_angle):
                            # Move the nodes u and v to increase the angle
                            angle_diff = np.radians(constraints.min_edge_angle) - angle

                            # Determine the sign of the angle between vec1 and vec2 using the cross product (2D)
                            cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
                            sign = np.sign(cross)

                            # Rotate vec1 and vec2 away from each other by half the angle difference, in opposite directions
                            theta = sign * (angle_diff / 2)

                            rotation_matrix1 = np.array(
                                [
                                    [np.cos(theta), np.sin(theta)],
                                    [-np.sin(theta), np.cos(theta)],
                                ]
                            )
                            rotation_matrix2 = np.array(
                                [
                                    [np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)],
                                ]
                            )

                            new_vec1 = rotation_matrix1 @ vec1
                            new_vec2 = rotation_matrix2 @ vec2

                            network.nodes[u]["position"] = pos_node + new_vec1
                            network.nodes[v]["position"] = pos_node + new_vec2
                            changed_network = True

            # Check for intersections:
            for u, v in network.edges:
                for w, x in network.edges:
                    if u == w or u == x or v == w or v == x:
                        continue

                    if do_line_segments_intersect(
                        network.nodes[u]["position"],
                        network.nodes[v]["position"],
                        network.nodes[w]["position"],
                        network.nodes[x]["position"],
                    ):
                        # Start the renormalization process from the beginning with a smaller step
                        # size to avoid intersections to occur again
                        return renormalize_network(
                            original_network,
                            min_total_edge_length,
                            max_total_edge_length,
                            constraints,
                            step_size / 2,
                            max_iterations - iteration_count,
                            max_iterations_for_constraint_matching,
                        )

            # Check if the last iteration changed the network
            if not changed_network:
                break
            if count == max_iterations_for_constraint_matching - 1:
                raise ValueError(
                    "Could not get network within constraints after "
                    f"{max_iterations_for_constraint_matching} iterations."
                )

    total_edge_length = _get_total_edge_length(network)

    raise ValueError(
        f"Could not renormalise network within {max_iterations} iterations. "
        f"Current total edge length: {total_edge_length}, "
        f"desired range: [{min_total_edge_length}, {max_total_edge_length}]"
    )
