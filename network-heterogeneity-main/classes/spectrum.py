from typing import List
import numpy as np


class Spectrum:
    def __init__(
        self,
        frequencies: np.ndarray,
        intensities: np.ndarray,
        mode_indices: List[int],
        only_lasing_modes: bool,
    ):
        self.frequencies = frequencies
        self.intensities = intensities
        self.mode_indices = mode_indices
        self.only_lasing_modes = only_lasing_modes

    def distance_to(
        self, other, normalize: bool = True, accept_differing_lengths: bool = False
    ) -> float:
        if len(self.frequencies) != len(other.frequencies):
            if accept_differing_lengths:
                print("Warning: Spectra have different lengths.")
            else:
                return np.inf

        distance = 0

        my_points = np.dstack((self.frequencies, self.intensities))[0]
        other_points = np.dstack((other.frequencies, other.intensities))[0]

        if normalize:
            min_x = min(np.min(my_points[:, 0]), np.min(other_points[:, 0]))
            max_x = max(np.max(my_points[:, 0]), np.max(other_points[:, 0]))
            min_y = min(np.min(my_points[:, 1]), np.min(other_points[:, 1]))
            max_y = max(np.max(my_points[:, 1]), np.max(other_points[:, 1]))

            my_points[:, 0] = (my_points[:, 0] - min_x) / (max_x - min_x)
            my_points[:, 1] = (my_points[:, 1] - min_y) / (max_y - min_y)
            other_points[:, 0] = (other_points[:, 0] - min_x) / (max_x - min_x)
            other_points[:, 1] = (other_points[:, 1] - min_y) / (max_y - min_y)

        for p in my_points:
            if len(other_points) == 0:
                break

            # Find closest point in other spectrum
            closest = np.argmin(np.linalg.norm(other_points - p, axis=1))
            distance += np.linalg.norm(p - other_points[closest]) ** 2

            # Remove point from other spectrum
            other_points = np.delete(other_points, closest, axis=0)

        return np.sqrt(distance)

    def get_mode_indices_in_descending_order_of_intensity(self) -> List[int]:
        """
        Get the indices of the modes in descending order of intensity.
        """
        helper_indices = np.argsort(self.intensities)[::-1]
        return [self.mode_indices[i] for i in helper_indices]
