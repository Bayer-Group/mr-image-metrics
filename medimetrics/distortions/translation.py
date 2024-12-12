from typing import Type

import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import shift

from medimetrics.base import Distortion


class Translation(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.name = "Translation :-)"
        self.parameter_ranges = {
            "translation_x": (0.01, 0.1),
            "translation_y": (0.01, 0.1),
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        tx = self.parameters["translation_x"] * image.shape[0]
        ty = self.parameters["translation_y"] * image.shape[1]
        """Px = int(np.ceil(tx)) py = int(np.ceil(ty)) padded_image =
        np.pad(image, pad_width = ((px, px), (py, py)), mode="wrap") points =
        np.meshgrid(image.shape) print("image.shape: ", image.shape)
        print("len(points; ", len(points)) points =
        np.meshgrid(padded_image.shape) print("len(points; ", len(points))

        values = padded_image trans_points = points.copy()
        trans_points[0] = trans_points[0] + tx trans_points[1] =
        trans_points[1] + ty distorted_image = interpn(points, values,
        xi=trans_points)
        """

        distorted_image = shift(image, (tx, ty), output=None, order=3, mode="wrap", cval=0.0, prefilter=True)

        return distorted_image
