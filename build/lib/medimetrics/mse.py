from medimetrics.metrics import FullRefMetric
import numpy as np

class MSE(FullRefMetric):

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
        Optionally, restrict evaluation to region with mask > 0
    data_range:
        By default use joint maximum - joint minimum
    """
    
    def compute2D(self, image_true: np.array, image_test: np.array, mask: np.array=None, data_range: float=None) -> float:
        return np.mean(self.compute2D_map(image_true[mask>0], image_test[mask>0]), None)

    def compute2D_map(self, image_true: np.array, image_test: np.array, data_range: float=None) -> np.array:
        return np.power(image_true - image_test, 2)



    # 2D and 3D are the same
    def compute3D(self, image_true: np.array, image_test: np.array, mask: np.array=None, data_range: float=None) -> float:
        return self.compute2D(image_true, image_test, mask, data_range)

    def compute3D_map(self, image_true: np.array, image_test: np.array, data_range: float=None) -> np.array:
        return self.compute2D(image_true, image_test, data_range)
    