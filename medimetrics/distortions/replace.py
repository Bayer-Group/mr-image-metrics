import numpy as np

from medimetrics.base import Distortion


class ReplaceArtifact(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {
            "fraction": (0.1, 1.0),
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        starty = 0
        # find first y, where the image is not all black
        while (image[starty, ...] == image[image.shape[0] - starty - 1, ...]).all():
            starty += 1

        endy = image.shape[0] // 2

        y = int(np.round((endy - starty) * self.parameters["fraction"]))
        reflected_part = np.flip(image[0 : starty + y, ...], axis=0)

        distorted_image = image.copy()
        distorted_image[image.shape[0] - y - starty : image.shape[0], ...] = reflected_part

        return distorted_image
