import numpy as np
from typing import Union

class Pattern:
    def __init__(self, array: np.ndarray, identifier: Union[str, None] = None):
        if not isinstance(array, np.ndarray) or array.ndim != 2:
            raise ValueError("Input must be a 2D numpy array")
        self.array = array
        self.identifier = identifier
        self.feature_map_mask = None

    def __eq__(self, other):
        if isinstance(other, Pattern):
            return np.array_equal(self.array, other.array)
        return False

    def __hash__(self):
        return hash(self.array.tobytes())

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __repr__(self):
        return f"Pattern(identifier={self.identifier}, array={self.array})"

    @property
    def width(self):
        return self.array.shape[1]

    @property
    def height(self):
        return self.array.shape[0]
    
    @property
    def size(self):
        return self.array.shape

    @property
    def average_brightness(self):
        return np.mean(self.array)

    @property
    def pixel_values(self):
        return self.array.flatten()

    def upscale_uneven(self, x_factor: int, y_factor: int):
        """
        Upscale the pattern with different factors for x and y.

        Args:
            x_factor (int): The upscale factor for x.
            y_factor (int): The upscale factor for y.

        Returns:
            Pattern: The upscaled pattern.
        """
        return Pattern(
            np.kron(self.array, np.ones((y_factor, x_factor))), self.identifier
        )

    def upscale(self, upscale_factor: int):
        """
        Upscale the pattern.

        Args:
            upscale_factor (int): The upscale factor.

        Returns:
            Pattern: The upscaled pattern.
        """
        return Pattern(
            np.kron(self.array, np.ones((upscale_factor, upscale_factor))),
            self.identifier,
        )

    def upscale_to_match(self, x_target: int, y_target: int):
        return self.upscale_uneven(
            x_target // self.array.shape[1], y_target // self.array.shape[0]
        )


all_on = Pattern(np.ones((1, 1)), "1x1/all_on")
all_off = Pattern(np.zeros((1, 1)), "1x1/all_off")
