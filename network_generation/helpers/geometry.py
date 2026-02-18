from typing import Tuple, Union
import numpy as np


def do_line_segments_intersect(p1, p2, q1, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if np.abs(val) < 1e-10:
            return 0  # Collinear
        return 1 if val > 0 else -1  # Clockwise or Counterclockwise

    def on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[
            1
        ] <= max(p[1], r[1])

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False


def get_intersection_point(support1, direction1, support2, direction2):
    """
    Calculate the intersection point of two lines.
    """

    # Calculate the intersection point
    t = (
        (support1[0] - support2[0]) * direction2[1]
        - (support1[1] - support2[1]) * direction2[0]
    ) / (direction1[1] * direction2[0] - direction1[0] * direction2[1])

    intersection = support1 + t * direction1

    return np.array(intersection)


def get_closest_point_on_line(point, line_start, line_end):
    line_vector = line_end - line_start

    # Calculate the projection of the point onto the line
    projection = line_start + (
        (np.dot(point - line_start, line_vector))
        * line_vector
        / np.linalg.norm(line_vector) ** 2
    )

    return projection


def get_distance_between_point_and_line(point, line_start, line_end):
    """
    Calculate the distance between a point and a line.
    """

    projection = get_closest_point_on_line(point, line_start, line_end)
    distance = np.linalg.norm(point - projection)

    return distance


def get_closest_point_on_line_segment(point, line_start, line_end):

    # Determine the position of the projection on the line segment
    line_vector = line_end - line_start
    position_on_line_segment = (
        np.dot(point - line_start, line_vector) / np.linalg.norm(line_vector) ** 2
    )

    # Calculate the closest point on the line segment
    position_on_line_segment = np.clip(position_on_line_segment, 0, 1)
    closest_point = line_start + position_on_line_segment * line_vector

    return closest_point


def get_distance_between_point_and_line_segment(point, line_start, line_end):
    """
    Calculate the distance between a point and a line segment.
    """

    closest_point = get_closest_point_on_line_segment(point, line_start, line_end)
    distance = np.linalg.norm(point - closest_point)

    return distance


def get_intersection_points_of_circle_and_line(
    circle_center, radius, line_start, line_end
):
    """
    Calculate the intersection points of a circle and a line.
    """

    # Calculate the direction of the line
    line_direction = line_end - line_start

    # Calculate the vector from the line start to the circle center
    circle_to_line_start = circle_center - line_start

    # Calculate the projection of the circle center onto the line
    projection = line_start + (
        (np.dot(circle_to_line_start, line_direction))
        * line_direction
        / np.linalg.norm(line_direction) ** 2
    )

    # Calculate the distance between the circle center and the projection
    distance = np.linalg.norm(circle_center - projection)

    # Calculate the intersection points
    if distance > radius:
        return []

    if distance == radius:
        return [projection]

    # Calculate the distance between the projection and the intersection points
    distance_to_intersection = np.sqrt(radius**2 - distance**2)

    # Calculate the intersection points
    intersection1 = (
        projection
        + distance_to_intersection * line_direction / np.linalg.norm(line_direction)
    )
    intersection2 = (
        projection
        - distance_to_intersection * line_direction / np.linalg.norm(line_direction)
    )

    return [intersection1, intersection2]


def get_portion_of_line_segment_within_circle(
    circle_center, radius, line_start, line_end
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """
    Calculate the portion of a line segment that is within a circle.
    """

    # Calculate the intersection points of the circle and the line
    intersection_points = get_intersection_points_of_circle_and_line(
        circle_center, radius, line_start, line_end
    )

    if len(intersection_points) == 0:
        if np.linalg.norm(line_start) <= radius and np.linalg.norm(line_end) <= radius:
            return line_start, line_end
        else:
            return None

    if len(intersection_points) == 1:
        if np.abs(line_start) <= radius:
            intersection_points.append(line_start)
        else:
            intersection_points.append(line_end)

    line_direction = line_end - line_start

    intersection_points_position_on_line = [
        np.dot(intersection_point - line_start, line_direction)
        / np.linalg.norm(line_direction) ** 2
        for intersection_point in intersection_points
    ]
    intersection_points_position_on_line.sort()
    intersection_points_position_on_line = [
        np.clip(position, 0, 1) for position in intersection_points_position_on_line
    ]

    if (
        intersection_points_position_on_line[0]
        == intersection_points_position_on_line[1]
    ):
        return None

    new_line_start = (
        line_start + intersection_points_position_on_line[0] * line_direction
    )
    new_line_end = line_start + intersection_points_position_on_line[1] * line_direction

    return new_line_start, new_line_end


def get_intersection_points_of_square_and_line(
    square_center, radius, line_start, line_end
):

    # Calculate the direction of the line
    line_direction = line_end - line_start

    # List of square edges
    square_edges = [
        # (support, direction)
        (
            (square_center + np.array([radius, 0])),
            np.array([0, 1]),
        ),
        (
            (square_center + np.array([-radius, 0])),
            np.array([0, 1]),
        ),
        (
            (square_center + np.array([0, radius])),
            np.array([1, 0]),
        ),
        (
            (square_center + np.array([0, -radius])),
            np.array([1, 0]),
        ),
    ]

    # Get intersection points
    intersection_points = []
    for edge in square_edges:
        intersection = get_intersection_point(
            line_start, line_direction, edge[0], edge[1]
        )
        if (
            np.abs(intersection[0] - square_center[0]) - 1e-6 <= radius
            and np.abs(intersection[1] - square_center[1]) - 1e-6 <= radius
        ):
            intersection_points.append(intersection)

    return intersection_points


def get_portion_of_line_segment_within_square(
    square_center, radius, line_start, line_end
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """
    Calculate the portion of a line segment that is within a square.
    """

    # Calculate the intersection points of the square and the line
    intersection_points = get_intersection_points_of_square_and_line(
        square_center, radius, line_start, line_end
    )

    if len(intersection_points) == 0:
        if (
            np.abs(line_start[0]) <= radius
            and np.abs(line_start[1]) <= radius
            and np.abs(line_end[0]) <= radius
            and np.abs(line_end[1]) <= radius
        ):
            return line_start, line_end
        else:
            return None

    if len(intersection_points) == 1:
        if np.abs(line_start[0]) <= radius and np.abs(line_start[1]) <= radius:
            intersection_points.append(line_start)
        else:
            intersection_points.append(line_end)

    line_direction = line_end - line_start

    intersection_points_position_on_line = [
        np.dot(intersection_point - line_start, line_direction)
        / np.linalg.norm(line_direction) ** 2
        for intersection_point in intersection_points
    ]
    intersection_points_position_on_line.sort()
    intersection_points_position_on_line = [
        np.clip(position, 0, 1) for position in intersection_points_position_on_line
    ]

    if (
        intersection_points_position_on_line[0]
        == intersection_points_position_on_line[1]
    ):
        return None

    new_line_start = (
        line_start + intersection_points_position_on_line[0] * line_direction
    )
    new_line_end = line_start + intersection_points_position_on_line[1] * line_direction

    return new_line_start, new_line_end
