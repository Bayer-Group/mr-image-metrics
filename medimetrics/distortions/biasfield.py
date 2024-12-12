import numpy as np

from medimetrics.base import Distortion


class BiasField(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {
            "coefficient": (0.5, 20.0),
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        x = np.linspace(0, 1, image.shape[0])
        y = np.linspace(0, 1, image.shape[1])

        bias_field = np.zeros(image.shape)

        field_x = 10 * (x - 1) * x * x
        field_y = (y - 0.5) * y * (y - 1)

        bias_field = np.matmul(field_x.reshape(-1, 1), field_y.reshape(1, -1))

        bias_field *= self.parameters["coefficient"]

        bias_field = np.exp(bias_field).astype(np.float32)

        return image * bias_field
