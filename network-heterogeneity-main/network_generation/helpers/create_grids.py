import numpy as np


def create_hex_grid(radius=1.0, grid_size=3):
    """
    Generates coordinates for a hexagonal grid.

    # Parameters
    radius = 1.0   # Distance from the center to a vertex of the hexagon
    grid_size = 3  # Defines the range of the grid in axial coordinates
    """

    hex_points = []

    # The vertical distance between adjacent rows (based on hexagon geometry)
    row_height = np.sqrt(3) * radius

    # Generate the grid of points
    for q in range(-grid_size, grid_size + 1):
        r_min = max(-grid_size, -q - grid_size)
        r_max = min(grid_size, -q + grid_size)
        for r in range(r_min, r_max + 1):
            if (q - r) % 3 == 0:
                continue

            # Convert axial coordinates (q, r) into 2D cartesian coordinates (x, y)
            x = radius * 3 / 2 * q
            y = row_height * (r + q / 2)
            hex_points.append([x, y])

    return np.array(hex_points)


def create_square_grid(radius=1.0, grid_size=5):
    """
    Generates coordinates for a square grid.

    # Define the grid size (N x N)
    eg. N = 5
    """

    # Generate x and y coordinates
    x = np.arange(grid_size) - (grid_size - 1) / 2
    y = np.arange(grid_size) - (grid_size - 1) / 2

    # Create a 2D grid of coordinates
    X, Y = np.meshgrid(x, y)
    X, Y = X * radius, Y * radius

    # Flatten the grids and combine the x, y coordinates into a list of tuples
    sq_points = list(zip(X.flatten(), Y.flatten()))

    return np.array(sq_points)


def create_triangle_grid(radius=1.0, grid_size=3):
    """
    Generates coordinates for a hexagonal grid.

    # Parameters
    radius = 1.0   # Distance from the center to a vertex of the hexagon
    grid_size = 3  # Defines the range of the grid in axial coordinates
    """

    hex_points = []

    # The vertical distance between adjacent rows (based on hexagon geometry)
    row_height = np.sqrt(3) * radius

    # Generate the grid of points
    for q in range(-grid_size, grid_size + 1):
        r_min = max(-grid_size, -q - grid_size)
        r_max = min(grid_size, -q + grid_size)
        for r in range(r_min, r_max + 1):
            # Convert axial coordinates (q, r) into 2D cartesian coordinates (x, y)
            x = radius * 3 / 2 * q
            y = row_height * (r + q / 2)
            hex_points.append([x, y])

    return np.array(hex_points)
