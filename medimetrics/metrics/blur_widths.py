from typing import Any, List

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.feature import canny

from medimetrics.base import NonRefMetric


def select_edges_by_dims(image: np.ndarray, binary_edges: np.ndarray) -> List[np.ndarray]:
    """Quantize angles of largest gradient to 45 Select x (angle = 0) and y
    (angle = +/- 180) directed edges. Currently only for 2D.

    Parameters:
    -----------
    image: NDArray
        image with intensities to calculate the widths of edges

    binary_edges: NDArray
        binary image with edges detected


    Returns:
    --------
    List[np.ndarray] (length is 2 for 2D)
        0: binary edges in x-direction
        1: binary edges in y-direction
    """

    row, column = image.shape

    # distiguish between horizontal and vertical edges:
    # find the gradient for the image
    gradient_y, gradient_x = np.gradient(image)

    # holds the angle information of the edges
    edge_angles = np.zeros(image.shape)

    # calculate the angle of the edges
    for imCol in range(1, column):
        for imRow in range(0, row):
            if gradient_x[imRow, imCol] != 0:
                edge_angles[imRow, imCol] = np.arctan2(gradient_y[imRow, imCol], gradient_x[imRow, imCol]) * (
                    180 / np.pi
                )
            elif gradient_x[imRow, imCol] == 0 and gradient_y[imRow, imCol] == 0:
                edge_angles[imRow, imCol] = 0
            elif gradient_x[imRow, imCol] == 0 and gradient_y[imRow, imCol] == np.pi / 2:
                edge_angles[imRow, imCol] = 90

    # quantize the angle
    quantized_angles = 45 * np.round(edge_angles / 45)

    # select x and y edges:
    edges_x = (quantized_angles == 0) & (binary_edges > 0)
    edges_y = ((quantized_angles == 180) | (quantized_angles == -180)) & (binary_edges > 0)

    return [edges_x, edges_y]


def calc_edge_widths(image: np.ndarray, binary_edges: np.ndarray, step: List[int]) -> np.ndarray:
    """Calculate the widths of the given edges in the image.
    Inspired, but strongly adapted from: https://github.com/affaalfiandy/Python-Blur-Metric/blob/main/nr_blur_np.py.

    Paper:
    Perceptual blur and ringing metrics: application to JPEG2000
    Marziliano, P; Dufaux, F; (...); Ebrahimi, T
    IEEE International Conference on Image Processing. Feb 2004
    SIGNAL PROCESSING-IMAGE COMMUNICATION 19 (2) , pp.163-172

    Parameters:
    -----------

    image: np.ndarray
        image with intensities to calculate the widths of edges

    binary_edges: np.ndarray
        binary image with edges detected

    step: List[int] (currently only len = 2, for 2D coords)
        directional vector in which edges are to be traced, e.g. [1, 0] for x-direction

    Returns:
    --------
    NDArray
        array of edge widths for each edge pixel
    """

    # keep edge widths for each pixel
    edge_widths = np.zeros_like(image)

    # currently only 2D:
    edge_coords = [(x, y) for x in range(image.shape[0]) for y in range(image.shape[1]) if binary_edges[x][y]]
    for x, y in edge_coords:
        current_pos = (x, y)
        edgePix = image[current_pos]

        # Tracing Back
        current_pos = (current_pos[0] - step[0], current_pos[1] - step[1])
        prevPix = edgePix
        backValue = 0
        # check if current position is inside image
        while (
            (current_pos[0] >= 0)
            and (current_pos[1] >= 0)
            and (current_pos[0] < image.shape[0])
            and (current_pos[1] < image.shape[1])
        ):
            curPix = image[current_pos]
            # check if image intensity values along the step direction are continuously
            # increasing or decreasing, otherwise stop
            if (
                (prevPix > edgePix and curPix <= prevPix)
                or (prevPix < edgePix and curPix >= prevPix)
                or (edgePix == curPix)
            ):
                break

            # next nackward step
            backValue += 1
            prevPix = curPix
            current_pos = current_pos = (current_pos[0] - step[0], current_pos[1] - step[1])

        # Tracing Forward
        current_pos = (x, y)
        edgePix = image[current_pos]
        current_pos = current_pos = (current_pos[0] + step[0], current_pos[1] + step[1])
        prevPix = edgePix
        forwardValue = 0
        # check if current position is inside image:
        while (
            (current_pos[0] >= 0)
            and (current_pos[1] >= 0)
            and (current_pos[0] < image.shape[0])
            and (current_pos[1] < image.shape[1])
        ):
            curPix = image[current_pos]
            # check if image intensity values along the step direction are continuously
            # increasing or decreasing, otherwise stop
            if (
                (prevPix > edgePix and curPix <= prevPix)
                or (prevPix < edgePix and curPix >= prevPix)
                or (edgePix == curPix)
            ):
                break

            # next forward step:
            forwardValue += 1
            prevPix = curPix
            current_pos = current_pos = (current_pos[0] + step[0], current_pos[1] + step[1])

        edge_widths[x, y] = float(backValue) + float(forwardValue)

    return edge_widths


def width_JNB(block_contrast: float, max_contrast: float) -> float:
    if block_contrast / max_contrast <= 50 / 255.0:
        return 5
    else:
        return 3


class BlurJNB(NonRefMetric):
    def __init__(self) -> None:
        pass

    def compute(
        self,
        image: np.ndarray,
        data_range: int = None,
        block_size: int = 64,
        edge_block_threshold: float = 0.002,
        beta: float = 3.6,
        **kwargs: Any
    ) -> float:
        """Calculates the sum of normalized edge widths in the edge width map
        of over all pixels.

        Parameters:
        -----------
        image: np.ndarray (H, W)
            Image to be evaluated
        data_range:
            By default use joint maximum - joint minimum
        block_size:
            edge length of processed blocks in pixels (default in original paper: 64)
        beta:
            constant for blur probability model (default in original paper: 3.6)
        edge_block_threshold:
            threshold for edge blocks (default in original paper: 0.002)
        """

        # set default data_range:
        if data_range is None:
            data_range = np.max(image) - np.min(image)

        canny_edges = canny(image, sigma=1, low_threshold=0.1 * data_range, high_threshold=0.2 * data_range)

        processed_blocks = 0

        x_chunks = int(np.ceil((image.shape[0]) / block_size))
        y_chunks = int(np.ceil((image.shape[1]) / block_size))

        block_distortions = []

        # scan each block
        for x in range(x_chunks):
            cx = x * block_size

            for y in range(y_chunks):
                cy = y * block_size

                edge_chunk = canny_edges[cy : cy + block_size, cx : cx + block_size]

                # process further if chunk is edge chunk
                if np.count_nonzero(edge_chunk) / (block_size * block_size) > edge_block_threshold:
                    processed_blocks += 1
                    lum_chunk = image[cy : cy + block_size, cx : cx + block_size]

                    # estimate local contrast of edge and compute corresponding edge width
                    block_contrast = lum_chunk.max() - lum_chunk.min()
                    jnb_width = width_JNB(block_contrast, data_range)

                    # for each edge compute corresponding edge width
                    # (only horizontal edges are considered)
                    edge_chunk_y = select_edges_by_dims(lum_chunk, edge_chunk)[1]
                    edge_widths = calc_edge_widths(lum_chunk, edge_chunk_y, step=[0, 1])

                    # compute block distortion
                    ratios = np.divide(edge_widths[edge_widths > 0], jnb_width)
                    edge_distortion = np.power(ratios, beta)

                    block_distortion = np.sum(edge_distortion)

                    block_distortions.append(block_distortion)

        block_distortions_array = np.array(block_distortions)
        block_distortions_array = np.abs(block_distortions_array) ** beta
        image_distortion = np.sum(block_distortions_array) ** (1.0 / beta)

        # return measures (blurriness and sharpness are simply inverse of each other)
        if processed_blocks != 0:
            blur_distortion = image_distortion / processed_blocks
        else:
            blur_distortion = np.inf

        # output the calculated value
        return blur_distortion


class BlurCPBD(NonRefMetric):
    def __init__(self) -> None:
        pass

    def compute(
        self,
        image: np.ndarray,
        data_range: int = None,
        block_size: int = 64,
        edge_block_threshold: float = 0.002,
        beta: float = 3.6,
        **kwargs: Any
    ) -> float:
        """Calculates the sum of normalized edge widths in the edge width map
        of over all pixels.

        Parameters:
        -----------
        image: np.ndarray (H, W)
            Image to be evaluated
        data_range:
            By default use joint maximum - joint minimum
        block_size:
            edge length of processed blocks in pixels (default in original paper: 64)
        beta:
            constant for blur probability model (default in original paper: 3.6)
        edge_block_threshold:
            threshold for edge blocks (default in original paper: 0.002)
        """

        # set default data_range:
        if data_range is None:
            data_range = np.max(image) - np.min(image)

        # create binary edge map with canny algorithm:
        canny_edges = canny(image, sigma=1, low_threshold=0.1 * data_range, high_threshold=0.2 * data_range)

        # determine horizontal edges
        edges_h = select_edges_by_dims(image, canny_edges)[1]

        # determine edge widths for each edge pixel:
        edge_widths = calc_edge_widths(image, binary_edges=edges_h, step=[0, 1])

        total_num_edges = 0

        # histogram for the probability of blur detection
        hist_pblur = np.zeros(101)

        # maximum block indices
        num_blocks_vertically = int(image.shape[0] / block_size)
        num_blocks_horizontally = int(image.shape[1] / block_size)

        #  loop over the blocks
        for i in range(num_blocks_vertically):
            for j in range(num_blocks_horizontally):
                # get the row and col indices for the block pixel positions
                rows = slice(block_size * i, block_size * (i + 1))
                cols = slice(block_size * j, block_size * (j + 1))

                # check if this an edge block based on threshold and the number of edges
                if (np.count_nonzero(canny_edges[rows, cols]) / canny_edges[rows, cols].size) > edge_block_threshold:
                    # get the (horizontal) edge widths in the block
                    block_widths = edge_widths[rows, cols]

                    # rotate block to simulate column-major boolean indexing
                    block_widths = np.rot90(np.flipud(block_widths), 3)
                    block_widths = block_widths[block_widths != 0]

                    block_contrast = np.max(image[rows, cols]) - np.min(image[rows, cols])
                    block_jnb = width_JNB(block_contrast, data_range)

                    # calculate the probability of blur detection at the edges
                    # detected in the block
                    prob_blur_detection = 1 - np.exp(-abs(block_widths / block_jnb) ** beta)

                    # update the statistics using the block information
                    for probability in prob_blur_detection:
                        bucket = int(round(probability * 100))
                        hist_pblur[bucket] += 1
                        total_num_edges += 1

        # normalize the pdf
        if total_num_edges > 0:
            hist_pblur = hist_pblur / total_num_edges

        # calculate the sharpness metric
        sharpness_value = np.sum(hist_pblur[:64])

        return sharpness_value


class BlurWidths(NonRefMetric):
    def __init__(self) -> None:
        pass

    def compute(self, image: np.ndarray, data_range: int = None, **kwargs: Any) -> float:
        """Calculates the sum of normalized edge widths in the edge width map
        of over all pixels.

        Parameters:
        -----------
        image: np.ndarray (H, W)
            Image to be evaluated
        data_range:
            By default use joint maximum - joint minimum
        """

        # set default data_range:
        if data_range is None:
            data_range = np.max(image) - np.min(image)

        # create binary edge map with canny algorithm:
        canny_edges = canny(image, sigma=1, low_threshold=0.1 * data_range, high_threshold=0.2 * data_range)

        # determine edges for x and y dimension:
        edges_per_dim = select_edges_by_dims(image, canny_edges)

        # determine edge widths for each edge pixel:
        edge_widths_x = calc_edge_widths(image, binary_edges=edges_per_dim[0], step=[1, 0])
        edge_widths_y = calc_edge_widths(image, binary_edges=edges_per_dim[1], step=[0, 1])

        # normalize by number of edges:
        norm_edge_widths = (
            edge_widths_x.astype(np.float64) / edges_per_dim[0].sum()
            + edge_widths_y.astype(np.float64) / edges_per_dim[1].sum()
        ) / 2.0

        return norm_edge_widths.sum()
