import numpy as np

from medimetrics.base import Distortion


class Stripes(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {"intensity": (0.05, 0.5), "k_radius": (0.3, 0.3), "phi": (0.0, 0.0)}

    def apply(self, image: np.ndarray) -> np.ndarray:
        intensity = self.parameters["intensity"]
        radius = self.parameters["k_radius"]
        phi = self.parameters["phi"]

        spikes_positions = [radius * np.cos(phi), radius * np.sin(phi)]

        # taken from torchio.RandomSpike implementation:
        # https://torchio.readthedocs.io/_modules/torchio/transforms/augmentation/intensity/random_spike.html#RandomSpike

        invert_transform = False

        transformed = np.fft.fftn(image)
        spectrum = np.fft.fftshift(transformed)

        shape = np.array(spectrum.shape)
        mid_shape = shape // 2
        indices = np.floor(spikes_positions * shape).astype(int)
        for index in indices:
            diff = index - mid_shape

            i, j = mid_shape + diff
            artifact = spectrum.max() * intensity
            if invert_transform:
                spectrum[i, j] -= artifact
            else:
                spectrum[i, j] += artifact

            # If we wanted to add a pure cosine, we should add spikes to both
            # sides of k-space. However, having only one is a better
            # representation og the actual cause of the artifact in real
            # scans. Therefore the next two lines have been removed.
            # #i, j, k = mid_shape - diff
            # #spectrum[i, j, k] = spectrum.max() * intensity_factor

        f_ishift = np.fft.ifftshift(spectrum)
        distorted_image = np.fft.ifftn(f_ishift).real

        return distorted_image.clip(image.min(), image.max())
