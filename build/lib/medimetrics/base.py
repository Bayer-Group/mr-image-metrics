import ABC from abc
import numpy as np



class FullRefMetric(ABC):

    def __init__():
        pass

    """
    Parameters:
    -----------
    image_true: np.array (H, W) or (H, W, D)
        Reference image
    image_test: np.array (H, W) or (H, W, D)
        Image to be evaluated against the reference image
    mask: 
        Optionally, restrict evaluation to masked region
    data_range:
        By default use joint maximum - joint minimum
    """
    def compute(self, image_true: np.array, image_test: np.array, mask: np.array=None, data_range: float=None) -> float:
        
        # assert consistent inputs:
        assert image_true.shape == image_test.shape, f"Reference image size {image_true.shape} does not test image size {image_test.shape}!"
        if mask is not None:
            assert mask.shape == image_true.shape, f"Mask size {mask.shape} does not match image size {image_true.shape}!"

        # check dimensionality:
        assert len(image_true.shape) == 2, f"Only 2 dimensions allowed! Input image has shape {image_true.shape}!"
        self.compute2D(image_true, image_test, mask, data_range)
        

    def compute_map(self, image_true: np.array, image_test: np.array, data_range: float=None) -> np.array:
        # assert consistent inputs:
        assert image_true.shape == image_test.shape

        # check dimensionality:
        # check dimensionality:
        assert len(image_true.shape) == 2, f"Only 2 dimensions allowed! Input image has shape {image_true.shape}!"
        self.compute2D_map(image_true, image_test, None, data_range)

    def compute2D(self, image_true: np.array, image_test: np.array, mask: np.array=None, data_range: float=None) -> float:
        pass

    def compute2D_map(self, image: np.array, data_range: float=None) -> np.array:
        pass
    

class NonRefMetric(ABC):

    def __init__():
        pass

    """
    Parameters:
    -----------
    image: np.array (H, W) or (H, W, D)
        Image to be evaluated
    mask: 
        Optionally, restrict evaluation to region with mask > 0
    data_range:
        By default use maximum - minimum
    """
    def compute(self, image: np.array, mask: np.array=None, data_range: float=None) -> float:
        
        # assert consistent inputs:
        if mask is not None:
            assert mask.shape == image.shape, f"Mask size {mask.shape} does not match image size {image.shape}!"

        # default data_range:
        if data_range is None:
            data_range = np.max(image) - np.min(image)
        
        # check dimensionality:
        assert len(image.shape) == 2, f"Only 2 dimensions allowed! Input image has shape {image.shape}!"
        self.compute2D(image, mask, data_range)

    def compute2D(self, image: np.array, mask: np.array=None, data_range: float=None) -> float:
        pass

