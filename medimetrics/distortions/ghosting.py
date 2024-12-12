import numpy as np

from medimetrics.base import Distortion


class Ghosting(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {
            "num_ghosts": (2, 2),
            "intensity": (0.05, 0.3),
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        num_ghosts = self.parameters["num_ghosts"]
        intensity = self.parameters["intensity"]

        transformed = np.fft.fftn(image)
        spectrum = np.fft.fftshift(transformed)

        # Variable "planes" is the part of the spectrum that will be modified
        # Variable "restore" is the part of the spectrum that will be restored
        axis = 1
        slices = [slice(None)] * spectrum.ndim
        slices[axis] = slice(None, None, num_ghosts)
        slices_tuple = tuple(slices)
        planes = spectrum[slices_tuple]

        restore_center = None
        dim_shape = spectrum.shape[axis]
        mid_idx = dim_shape // 2
        slices = [slice(None)] * spectrum.ndim
        if restore_center is None:
            slice_ = slice(mid_idx, mid_idx + 1)
        else:
            size_restore = int(np.round(restore_center * dim_shape))
            slice_ = slice(mid_idx - size_restore // 2, mid_idx + size_restore // 2)
        slices[axis] = slice_
        slices_tuple = tuple(slices)
        restore = spectrum[slices_tuple].copy()

        # Multiply by 0 if intensity is 1
        planes *= 1 - intensity

        # Restore the center of k-space to avoid extreme artifacts
        spectrum[slices_tuple] = restore

        f_ishift = np.fft.ifftshift(spectrum)
        ghosted_image = np.fft.ifftn(f_ishift).real

        return ghosted_image
