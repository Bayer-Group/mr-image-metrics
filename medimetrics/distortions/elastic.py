import elasticdeform
import numpy as np

from medimetrics.base import Distortion


class ElasticDeformation(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {"num_control_points": (18, 11), "max_rel_displacement": (0.03, 0.1)}

    def apply(self, image: np.ndarray) -> np.ndarray:
        max_shape = np.max(np.array(image.shape))
        sigma = max_shape / (self.parameters["num_control_points"]) * self.parameters["max_rel_displacement"]

        # apply deformation with a random 3 x 3 grid
        # elasticdeform uses spline interpolation
        # default is order 3, but order 1 guarantees values stay inside previous intensity range
        distorted_image = elasticdeform.deform_random_grid(
            image, sigma=sigma, points=self.parameters["num_control_points"], order=1
        )

        return distorted_image

        return
