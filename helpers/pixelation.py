import numpy as np
import networkx as nx
from more_itertools import pairwise
from helpers.relabel_nodes import relabel_nodes


def _pixelation_coordinates(G: nx.Graph, nx: int, ny: int):
    """
    Compute the coordinates for pixelation grid based on the given graph.

    Args:
        G (networkx.Graph): Graph representing the image.
        nx (int): Number of pixels in the x-direction.
        ny (int): Number of pixels in the y-direction.

    Returns:
        tuple: x and y coordinates of the pixelation grid.

    """
    points = np.array([G.nodes[node]["position"] for node in G.nodes])
    # Extract the positions of the nodes in the graph

    xmin, ymin = np.min(points, 0)  # Coordinate limits
    xmax, ymax = np.max(points, 0)
    xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2  # Center coordinates
    Lx, Ly = (xmax - xmin), (ymax - ymin)  # Length along x and y
    dx, dy = Lx / nx, Ly / ny  # Step size along x and y
    d = max(dx, dy)  # Equalise step sizes

    x0, y0 = xc - d * nx / 2, yc - d * ny / 2  # Corner coordinates of pixel grid
    x = x0 + np.linspace(0, nx, nx + 1) * d  # x and y coordinates of pixel grid
    y = y0 + np.linspace(0, ny, ny + 1) * d

    return x, y


def _pixelate_network(
    network: nx.Graph,
    xpixels: np.ndarray,
    ypixels: np.ndarray,
    threshold: float = 1,
    numerical_threshold: float = 1e-1,
    do_bookkeeping: bool = False,
):
    """
    Pixelise the network based on the given x and y pixel positions.

    Args:
        network (networkx.Graph): The original network.
        xpixels (numpy.ndarray): Array of x pixel positions.
        ypixels (numpy.ndarray): Array of y pixel positions.
        threshold (float, optional): Threshold to ignore close points
        numerical_threshold (float, optional): Threshold for numerical comparisons.

    Returns:
        networkx.Graph: The pixelised graph.

    """
    _network = network.copy()
    _network = relabel_nodes(_network)
    network_pixelised = _network.copy()  # Create a copy of the original graph

    if do_bookkeeping:
        network_pixelised.graph["original_edge_assignment"] = []

    node_count = len(_network.nodes)
    # Number of nodes in the original graph

    def normalise(u, v):
        """
        Ensure that the endpoints of an edge are in ascending order.

        Args:
            u (int): First endpoint of the edge.
            v (int): Second endpoint of the edge.

        Returns:
            Tuple[int, int]: Endpoints in ascending order.

        """
        nonlocal _network
        x1, y1 = _network.nodes[u]["position"]
        x2, y2 = _network.nodes[v]["position"]

        if x1 < x2 or (x1 == x2 and y1 < y2):
            return u, v
        else:
            return v, u

    for u, v in list(_network.edges):
        network_pixelised.remove_edge(u, v)  # Remove the original edge
        u, v = normalise(u, v)  # Reorder endpoints to be ascending
        x1, y1 = _network.nodes[u]["position"]  # End point coordinates
        x2, y2 = _network.nodes[v]["position"]
        add = [(x1, y1, u), (x2, y2, v)]  # Nodes on edge: Start with endpoints

        length_initial = np.hypot(x2 - x1, y2 - y1)  # Initial length

        for x in xpixels:
            # Split edge at each x pixel position

            if (
                abs(x - x1) > numerical_threshold
                and abs(x - x2) > numerical_threshold
                and (x - x1) * (x - x2) < 0
            ):
                # Check if the x coordinate lies between endpoints
                # but isn't within threshold

                y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                # Compute the y coordinate based on the line equation

                length_1 = np.hypot(x - x1, y - y1)
                length_2 = np.hypot(x - x2, y - y2)
                # Compute the lengths of the two new edges

                if length_1 < threshold or length_2 < threshold:
                    continue
                # If either of the new edges is too short, skip

                network_pixelised.add_node(node_count, position=np.array([x, y]))
                # Add a new node to the pixelised graph

                add.append((x, y, node_count))  # Append new point to the point list
                node_count += 1  # and increment node count

        for y in ypixels:
            # And do the same for y pixel positions

            if (
                abs(y - y1) > numerical_threshold
                and abs(y - y2) > numerical_threshold
                and (y - y1) * (y - y2) < 0
            ):
                x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)

                length_1 = np.hypot(x - x1, y - y1)
                length_2 = np.hypot(x - x2, y - y2)

                if length_1 < threshold or length_2 < threshold:
                    continue

                network_pixelised.add_node(node_count, position=np.array([x, y]))
                add.append((x, y, node_count))
                node_count += 1

        add = sorted(add)  # Sort the added nodes based on their coordinates

        length_final = 0
        for (x1, y1, n1), (x2, y2, n2) in pairwise(add):
            network_pixelised.add_edge(n1, n2)  # Add edges between consecutive nodes
            length_final += np.hypot(x2 - x1, y2 - y1)  # and add to final length

        assert abs(length_final - length_initial) < numerical_threshold
        # Check that new length is not far from the original

        if do_bookkeeping:
            network_pixelised.graph["original_edge_assignment"].append(
                [n for _, _, n in add]
            )

    return network_pixelised


def _create_empty_matrix(width: int, height: int):
    """
    Create an empty matrix with the given width and height.

    Args:
        width (int): Width of the matrix.
        height (int): Height of the matrix.

    """
    matrix = [0] * height
    for i in range(height):
        matrix[i] = [0] * width
        for j in range(width):
            matrix[i][j] = []

    return matrix


def _get_pixel_assignment(
    network: nx.Graph,
    xpixels: np.ndarray,
    ypixels: np.ndarray,
):
    """
    Assign each pixel to the corresponding nodes in the network.

    Args:
        network (networkx.Graph): The original network.
        xpixels (numpy.ndarray): Array of x pixel positions.
        ypixels (numpy.ndarray): Array of y pixel positions.

    Returns:
        array: Array mapping pixels to nodes.

    """

    width = len(xpixels) - 1
    height = len(ypixels) - 1
    d = xpixels[1] - xpixels[0]  # Pixel size

    pixel_assignment = _create_empty_matrix(height, width)

    # Iterate over each edge in the graph
    for i, (u, v) in enumerate(network.edges):
        # Compute the centroid position of the current edge
        xc, yc = (network.nodes[u]["position"] + network.nodes[v]["position"]) / 2

        # Compute the x and y pixel for the centroid position
        x_id = int(np.floor((xc - xpixels[0]) / d))
        y_id = int(np.floor((yc - ypixels[0]) / d))

        # Handle edge case:
        if x_id == len(pixel_assignment):
            x_id = len(pixel_assignment) - 1
        if y_id == len(pixel_assignment):
            y_id = len(pixel_assignment) - 1

        # Assign the edge to the corresponding pixel:
        pixel_assignment[y_id][x_id].append(i)

    return pixel_assignment


def pixelate_network(
    network: nx.Graph,
    width: int,
    height: int,
    threshold: float = 1,
    numerical_threshold: float = 1e-1,
    do_bookkeeping: bool = False,
) -> nx.Graph:
    """
    Pixelate the network based on the given width and height.

    Args:
        network (networkx.Graph): The original network.
        width (int): Width of the image.
        height (int): Height of the image.
        threshold (float, optional): Threshold to ignore close points.
        numerical_threshold (float, optional): Threshold for numerical comparisons.

    Returns:
        networkx.Graph: The pixelated graph.

    """
    xpixels, ypixels = _pixelation_coordinates(network, width, height)
    # Compute the x and y pixel positions based on the network

    pixelised_network = _pixelate_network(
        network=network,
        xpixels=xpixels,
        ypixels=ypixels,
        threshold=threshold,
        numerical_threshold=numerical_threshold,
        do_bookkeeping=do_bookkeeping,
    )
    pixel_assignment = _get_pixel_assignment(pixelised_network, xpixels, ypixels)
    pixelised_network.graph["pixel_assignment"] = pixel_assignment
    pixelised_network.graph["pixel_width"] = width
    pixelised_network.graph["pixel_height"] = height

    return pixelised_network


def get_edge_weights(
    G: nx.Graph, xpixels: np.ndarray, ypixels: np.ndarray, image: np.ndarray
):
    """
    Compute edge weights based on corresponding pixel values in the image.

    Args:
        G (nx.Graph): The input graph.
        xpixels (np.ndarray): x-coordinates of the pixel grid.
        ypixels (np.ndarray): y-coordinates of the pixel grid.
        image (np.ndarray): 2D array representing the image.

    Returns:
        np.ndarray: List of weights corresponding to each edge in the graph.

    """
    d = xpixels[1] - xpixels[0]  # Pixel size
    weights = []  # Edge weights

    # Iterate over each edge in the graph
    for u, v in G.edges:
        # Compute the centroid position of the current edge
        xc, yc = (G.nodes[u]["position"] + G.nodes[v]["position"]) / 2

        # Compute the x and y pixel for the centroid position
        x_id = int(np.floor((xc - xpixels[0]) / d))
        y_id = int(np.floor((yc - ypixels[0]) / d))

        # Retrieve the weight from the image at the computed pixel
        weights.append(image[y_id][x_id])

    return weights


def total_weighted_length(G: nx.Graph, weights: np.ndarray = None):
    """
    Compute the total weighted length of edges in the graph.

    Args:
        G (networkx.Graph): Graph representing the image.
        weights (numpy.ndarray, optional): Array of edge weights.

    Returns:
        float: Total weighted length of edges.

    """
    if weights is None:
        weights = np.ones(len(G.edges))
    # If no weights are provided, initialize all weights to 1

    total = 0  # Total weighted length

    for i, (u, v) in enumerate(G.edges):
        # Iterate over each edge in the graph with endpoints u, v

        total += weights[i] * np.hypot(
            *(G.nodes[v]["position"] - G.nodes[u]["position"])
        )
        # Length of the edge is the Euclidean distance between endpoints
        # Multiply the length by the corresponding weight and add to total

    return total
