import logging
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

log = logging.getLogger(__name__)


def normalize(
    image: np.ndarray,
    normalization: str = "none",
    mask: np.ndarray = None,
    get_invert_params: bool = False,
    new_min: float = 0.0,
    new_max: float = 1.0,
    lower_p: float = None,
    upper_p: float = None,
    integer_dtype: Type[Any] = np.uint8,
    out_dtype: Optional[Type[Any]] = None,
    nbins: int = 256,
    mask_value: float = 0.0,
) -> Union[np.ndarray[Any, Any], Tuple[np.ndarray[Any, Any], Dict[str, Any]]]:
    r"""

    Parameters:
    -----------
    image:
        numpy image of 2 or 3 dimensions,

    normalization:
        one of the following methods:
        "minmax":      normalize percentile(lower_p)->new_min, percentile(upper_p)->new_max
                       clip at percentile(lower_p), percentile(upper_p)
        "zscore":      normalize mean -> 0.0, std -> 1.0
        "histeq":     rebin value range into 256 bins with approx. equal percentiles
        "quantile":    set median to 0 and IQR $(I_{75\%} - I_{25\%})$ to 1
        "from_int_range":  assume fixed integer range and scale fix_min and fix_max to new_min and new_max
        "binning":     bin values into nbins bins

    mask:
        calculate parameters only from unmasked image region,
        also normalizes intensity values which are masked out,
        but can take on values outside of resulting range inside mask

    mask_value:
        assign this value to masked out regions

    get_invert_params:
        whether to return a Dict of parameters to invert the normalization (as close as possible)

    new_min:
        lower range after normalization, also applies to minmax, from_int_range

    new_max:
        upper range after normalization

    lower_p
        lower percentile to detect for rescaling and clipping. If None, min is used

    upper_p:
        upper percentile to detect for rescaling and clipping. If None, max is used

    nbins:
        number of bins to use for histogramm equalization or binning

    integer_dtype:
        if normalization = "from_int_range", assume this integer type for input image

    out_dtype:
        optionally convert to this type after normalization
        if None & normalization = "binning" and nbins <= 256, convert to np.uint8
        if None & normalization = "binning" and nbins <= 65536, convert to np.uint16


    """
    if mask is None:
        image_values = image
    elif mask.sum() > 0:
        image_values = image[mask > 0]
    else:
        log.warn("Skipping normalization with all-zero mask!")
        normalization = "none"

    assert new_min < new_max

    # Alias for minmax5:
    if normalization == "minmax5":
        lower_p = 5
        upper_p = 95
        normalization = "minmax"

    inv_params: Dict[str, Any] = {"new_min": new_min, "new_max": new_max, "input_dtype": image.dtype}

    if normalization == "minmax":
        inv_params["min"] = image_values.min() if lower_p is None else np.percentile(image_values, lower_p)
        inv_params["max"] = image_values.max() if upper_p is None else np.percentile(image_values, upper_p)
        # optionally clip at percentiles:
        if lower_p is not None or upper_p is not None:
            image = image.clip(inv_params["min"], inv_params["max"])

        image = image - inv_params["min"]
        if inv_params["min"] != inv_params["max"]:
            # rescale to 0, 1:
            image = image / (inv_params["max"] - inv_params["min"])
        # rescale to new range:
        image = image * (new_max - new_min) + new_min

        if out_dtype is None:
            out_dtype = np.float32

    elif normalization == "histeq":
        inv_params["nbins"] = nbins
        bins = [np.percentile(image_values, p / (nbins - 1) * 100) for p in range(0, nbins)]
        inv_params["bins"] = bins
        image = np.digitize(image, bins=bins, right=True)

        if out_dtype is None:
            if nbins <= 256:
                out_dtype = np.uint8
            elif nbins <= 65536:
                out_dtype = np.uint16
            else:
                out_dtype = np.float32

    elif normalization == "zscore":
        inv_params["mean"] = image_values.mean()
        inv_params["std"] = image_values.std()
        image = image - inv_params["mean"]
        if inv_params["std"] != 0:
            image = image / inv_params["std"]
        else:
            image = np.nan * np.ones_like(image)

        if out_dtype is None:
            out_dtype = np.float32

    elif normalization == "from_int_range":
        inv_params["integer_dtype"] = integer_dtype
        fix_min = np.iinfo(integer_dtype).min
        fix_max = np.iinfo(integer_dtype).max
        image_min = image_values.min()
        image_max = image_values.max()
        if image_max > fix_max or image_min < fix_min:
            log.warn(
                f"Normalizing from type {integer_dtype}, but image"
                + f" (region) actually has range {image_min} to {image_max} outside of"
                + f" assumed type range {fix_min} to {fix_max}, clipping before normalization!"
            )
            image = image.clip(fix_min, fix_max)
        image = (image - fix_min) / (fix_max - fix_min) * (new_max - new_min) + new_min

        if out_dtype is None:
            out_dtype = np.float32

    elif normalization == "quantile":
        inv_params["median"] = np.median(image_values)
        if upper_p is None and lower_p is None:
            lower_p = 25
            upper_p = 75
        elif upper_p is None:
            upper_p = 100
        elif lower_p is None:
            lower_p = 0

        inv_params["min"] = np.percentile(image_values, lower_p)
        inv_params["max"] = np.percentile(image_values, upper_p)
        inv_params["IQR"] = inv_params["max"] - inv_params["min"]
        inv_params["new_min"] = new_min
        inv_params["new_max"] = new_max

        # set median to 0:
        image = image - inv_params["median"]
        if inv_params["IQR"] != 0:
            # rescale to 0, 1:
            image = image / inv_params["IQR"]
        # rescale to new range:
        image = image * (inv_params["new_max"] - inv_params["new_min"]) + inv_params["new_min"]

        if out_dtype is None:
            out_dtype = np.float32

    elif normalization == "binning":
        inv_params["nbins"] = nbins
        inv_params["min"] = image_values.min()
        inv_params["max"] = image_values.max()
        inv_params["new_min"] = 0
        inv_params["new_max"] = nbins - 1

        # set minimum to 0:
        image = image - inv_params["min"]

        # divide by range:
        if inv_params["min"] != inv_params["max"]:
            image = image / (inv_params["max"] - inv_params["min"])

        # multiply by new range:
        image = image * inv_params["nbins"]

        image = np.minimum(inv_params["nbins"] - 1, np.floor(image))

        if out_dtype is None:
            if nbins <= 256:
                out_dtype = np.uint8
            elif nbins <= 65536:
                out_dtype = np.uint16
            else:
                out_dtype = np.float32

    # else assuming "none"
    inv_params["normalization"] = normalization

    # set masked out regions to mask_value:
    if inv_params["normalization"] != "none":
        image[mask == 0] = mask_value

    # convert to desired output type:
    if out_dtype is not None:
        image = image.astype(out_dtype)

    if get_invert_params:
        return image, inv_params
    else:
        return image


def invert_normalize(image: np.ndarray, params: Dict) -> np.ndarray:
    r"""
    Parameters:
    -----------
    image:
        numpy image of 2 or 3 dimensions,

    params:
        A dict including keys:
        "normalization":
            type of normalization, e.g.:

            "minmax":      normalize min->params["new_min"], max->["new_max"]
                        if params["lower_p"] or params["upper_p"] are not None, clip before rescaling
            "zscore":      normalize mean -> 0.0, std -> 1.0
            "histeq":      rebin value range into params["bins"] bins with approx. equal percentiles
            "from_int_range":  Assume fixed range integer type defined by np.dtype in params["int_dtype"]
            "quantile":     set median to 0 and IQR $(I_{75\%} - I_{25\%})$ to 1
            "binning":     bin values into params["nbins"] bins
        "input_dtype":
            original input dtype, which should be restored after inverted normalization
        "new_min", "new_max", "min", "max", "mean", "std", "median":
            parameters used to invert normalization

    """

    if params["normalization"] == "minmax":
        if params["max"] != params["min"]:
            image = (image - params["new_min"]) / (params["new_max"] - params["new_min"]) * (
                params["max"] - params["min"]
            ) + params["min"]
        # reverse values for constant images:
        elif params["max"] > params["new_max"]:
            image = image / params["new_max"] * params["max"]
        elif params["min"] < params["new_min"]:
            image = image / params["new_min"] * params["min"]

    elif params["normalization"] == "histeq":
        result_image = image.copy()
        for bin_idx in range(params["nbins"]):
            result_image[image == bin_idx] = params["bins"][bin_idx]
        image = result_image

    elif params["normalization"] == "zscore":
        image = image * params["std"] + params["mean"]

    elif params["normalization"] == "from_int_range":
        fix_min = np.iinfo(params["integer_dtype"]).min
        fix_max = np.iinfo(params["integer_dtype"]).max
        # scale to 0, 1:
        image = (image - params["new_min"]) / (params["new_max"] - params["new_min"])
        # scale to fixed range:
        image = np.round(image * (fix_max - fix_min) - fix_min)
        # clip to assure fixed range:
        image = image.clip(fix_min, fix_max).astype(params["integer_dtype"])

    elif params["normalization"] == "quantile":
        # rescale to 0, 1:
        if params["new_max"] != params["new_min"]:
            image = image * (params["new_max"] - params["new_min"])
        image = image + params["new_min"]

        # rescale to previous range:
        image = image * params["IQR"]
        image = image + params["median"]

    if params["normalization"] == "binning":
        # divide by new range:
        image = image / params["nbins"]

        # multiply by old range:
        if params["min"] != params["max"]:
            image = image * (params["max"] - params["min"])

        # restore minimum:
        image = image + params["min"]

    # else assuming params["normalization"] = "none"

    if "input_dtype" in params.keys():
        if params["input_dtype"].issctype():
            image = np.round(image)
        image = image.astype(params["input_dtype"])

    return image
