import glob
import os
import time
from typing import List, Type

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from medimetrics.base import FullRefMetric
from medimetrics.metrics import DSS, FSIM, GMSD, IWSSIM, MDSI, VIF, VSI, HaarPSI, PieApp
from medimetrics.utils import normalize

result_path = "/home/melanie.dohmen/experiments/evaluation_20240419/piq_metrics/results.csv"

reference_paths = sorted(glob.glob("/home/melanie.dohmen/experiments/reference_imagest1c_20240419/*.nii.gz"))
distorted_paths = sorted(glob.glob("/home/melanie.dohmen/experiments/distorted_imagest1c_20240419/*_*/"))

start_image_index = 0
end_image_index = 99
nr_images = end_image_index - start_image_index + 1

reference_images = [reference_paths[i] for i in range(end_image_index + 1)]

distorted_images = {}
for distortion_folder in distorted_paths:
    for i in range(nr_images):
        single_distortion_paths = sorted(glob.glob(distortion_folder + "/*.nii.gz"))
        distorted_images[distortion_folder.split("/")[-2]] = [
            single_distortion_paths[i] for i in range(end_image_index + 1)
        ]

norms: List[str] = ["none", "binning", "minmax", "minmax5", "zscore", "quantile"]
metrics: List[Type[FullRefMetric]] = [VIF, VSI, IWSSIM, DSS, FSIM, GMSD, HaarPSI, MDSI, PieApp]
image_idxs = range(nr_images)


if os.path.exists(result_path):
    write_mode = "a"
else:
    write_mode = "w"

with open(result_path, write_mode) as f:
    if write_mode == "w":
        f.write("generation_method,image,metric,norm,value,time\n")
    for metric in metrics:
        print("Calculating metric: ", metric.__name__)
        metric_inst = metric()

        for norm in norms:
            print("Normalizing method: ", norm)

            for image_idx in tqdm(range(start_image_index, end_image_index + 1)):
                ref = nib.load(reference_images[image_idx]).get_fdata()
                for norm in norms:
                    norm_ref = normalize(ref, norm)
                    for distortion in distorted_images.keys():
                        dist = nib.load(distorted_images[distortion][image_idx]).get_fdata()

                        norm_dist = normalize(dist, norm)
                        start = time.time()
                        try:
                            if isinstance(norm_ref, np.ndarray) and isinstance(norm_dist, np.ndarray):
                                metric_value = metric_inst.compute(norm_ref, norm_dist)
                                f.write(
                                    f"{distortion},{image_idx},{metric.__name__},{norm},{metric_value},{time.time() - start}\n"
                                )
                        except Exception as e:
                            print(
                                "Cound not calculate metric: ",
                                metric.__name__,
                                " for image: ",
                                image_idx,
                                " with norm: ",
                                norm,
                                " and distortion: ",
                                distortion,
                            )
                            print(e)
