# Similarity and Quality Metrics for MR Images

This repository provides the metrics and distortions, that were presented in the paper:

M. Dohmen, M. Klemens, T. Truong, I. Baltruschat, M. Lenga: "Similarity and Quality Metrics for MR Image-to-Image Translation"
submitted to Nature Scientific Reports
[(see preprint)](https://arxiv.org/abs/2405.08431)

# Installation

Clone repository

```
cd into mr-image-metrics
pip install .
```

# Example Usage

```
import numpy as np

from medimetrics.metrics import MSE, BlurEffect
from medimetrics.distortions import GaussianBlur, GammaHigh


blur = GaussianBlur(max_strength=5)
darker = GammaHigh(max_strength=5)

image_true = np.linspace(0.0, 1.0, 128*128).reshape(128, 128)
image_test_blurred = blur(image_true, strength=5)
image_test_darker = darker(image_true, strength=5)

mse = MSE()
print(mse.compute(image_true, image_test_darker))
> 0.0555207631635955

blur_effect = BlurEffect()
print(blur_effect.compute(image_test_blurred))
> 0.9772262051422484

```

# Metrics

## Reference Metrics

- __MSE__: Mean Squared Error
- __SSIM__: Structural Similarity Index Measure
  - Wang, Z., Bovik, A. C., Sheikh, H. R. & Simoncelli, E. P. Image quality assessment: from error visibility to structural
    similarity. IEEE Transactions on Image Process. 13, 600–12 (2004).
  - Implementation adapted from [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/image/structural__similarity.html)
- __MAE__: Mean Absolute Error
- __MS-SSIM__: Multi-Scale SSIM
  - Wang, Z., Simoncelli, E. & Bovik, A. Multiscale structural similarity for image quality assessment. In The Thirty-Seventh Asilomar Conference on Signals, Systems and Computers, 2003, vol. 2, 1398–1402 Vol.2, DOI: 10.1109/ACSSC.2003.1292216 (2003).
  - Implementation adapted from [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/image/multi__scale__structural__similarity.html)
- __CW-SSIM__: Complex-Wavelet SSIM
  - Sampat, M. P., Wang, Z., Gupta, S., Bovik, A. C. & Markey, M. K. Complex wavelet structural similarity: A new image
    similarity index. IEEE Transactions on Image Process. 18, 2385–2401, DOI: 10.1109/TIP.2009.2025923 (2009).
  - Implementation adapted from [github/dingkeyan93](https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/CW_SSIM.py)
- __PSNR__ : Peak-Signal-to-Noise-Ratio
  - Korhonen, J. & You, J. Peak signal-to-noise ratio revisited: Is simple beautiful? In 2012 Fourth International Workshop on Quality of Multimedia Experience, 37–38, DOI: 10.1109/QoMEX.2012.6263880 (2012)
- __LPIPS__ : Learned Perceptual Image Patch Similarity
  - Zhang, R., Isola, P., Efros, A. A., Shechtman, E. & Wang, O. The unreasonable effectiveness of deep features as a
    perceptual metric. CoRR ayn/1801.03924 (2018). 1801.03924.
  - Implementation integrated from [pypi](https://pypi.org/project/lpips/)
- __DISTS__ : Deep Image Stucture and Texture Similarity
  - Ding, K., Ma, K., Wang, S. & Simoncelli, E. P. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis Mach. Intell. 44, 2567–2581, DOI: 10.1109/TPAMI.2020.3045810 (2022).
  - Implementation adapted from [github/dingkeyan93](https://github.com/dingkeyan93/DISTS)
- __PCC__ : Pearson Correlation Coefficient
  - Implementation integrated from [numpy](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html)
- __NMI__ : Normalized Mutual Information
  - Maes, F., Collignon, A., Vandermeulen, D., Marchal, G. & Suetens, P. Multimodality image registration by maximization of mutual information. IEEE Transactions on Med. Imaging 16, 187–198, DOI: 10.1109/42.563664 (1997).
  - Implementation adapted from [scikit-image](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.normalized__mutual__information)

## Non-Reference Metrics

- __BlurWidths__:
  - Marziliano, P., Dufaux, F., Winkler, S. & Ebrahimi, T. A no-reference perceptual blur metric. In Proceedings. International Conference on Image Processing, vol. 3, III–III, DOI: 10.1109/ICIP.2002.1038902 (2002).
  - Implementation inspired by: [github/affaalfiandy](https://github.com/affaalfiandy/Python-Blur-Metric/blob/main/nr_blur_np.py)
- __BlurJNB__:
  - Ferzli, R. & Karam, L. J. A no-reference objective image sharpness metric based on the notion of just noticeable blur
    (jnb). IEEE Transactions on Image Process. 18, 717–728, DOI: 10.1109/TIP.2008.2011760 (2009).
  - Implementation adapted from: [github/davidatroberts](https://github.com/davidatroberts/No-Reference-Sharpness-Metric/blob/master/src/main.cpp)
- __BlurCPBD__:
  - Narvekar, N. D. & Karam, L. J. A no-reference image blur metric based on the cumulative probability of blur detection
    (cpbd). IEEE Transactions on Image Process. 20, 2678–2683, DOI: 10.1109/TIP.2011.2131660 (2011).
  - Implementation adapted from: [github/x64746b](https://github.com/0x64746b/python-cpbd)
- __BlurEffect__:
  - Crété-Roffet, F., Dolmiere, T., Ladret, P. & Nicolas, M. The Blur Effect: Perception and Estimation with a New No-
    Reference Perceptual Blur Metric. In SPIE Electronic Imaging Symposium Conf Human Vision and Electronic Imaging,
    vol. XII, EI 6492–16 (San Jose, United States, 2007).
  - Implementation adapted from [scikit-image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.blur_effect)
- __BlurRatio / MeanBlur__:
  - Choi, M. G., Jung, J. H. & Jeon, J. W. No-reference image quality assessment using blur and noise. Int. J. Electr. Comput. Eng. 3, 184–188 (2009).
  - Own implementation
- __BRISQUE__:
  - Mittal, A., Moorthy, A. K. & Bovik, A. C. No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Process. 21, 4695–4708, DOI: 10.1109/TIP.2012.2214050 (2012).
  - Implementation integrated from [pypi](https://pypi.org/project/brisque/)
- __NIQE__: Natural Image Quality Estimator
  - Mittal, A., Soundararajan, R. & Bovik, A. C. Making a “completely blind” image quality analyzer. IEEE Signal Process. Lett. 20, 209–212, DOI: 10.1109/LSP.2012.2227726 (2013).
  - Implementation adapted from [github/guptapraful](https://github.com/guptapraful/niqe)
- __VarLaplace__: Variance of Laplacian
  Pech-Pacheco, J., Cristobal, G., Chamorro-Martinez, J. & Fernandez-Valdivia, J. Diatom autofocusing in brightfield
  microscopy: a comparative study. In Proceedings 15th International Conference on Pattern Recognition. ICPR-2000,
  vol. 3, 314–317 vol.3, DOI: 10.1109/ICPR.2000.903548 (2000).
- __MeanTotalVar__: Mean Total Variation
  - Rudin, L. I., Osher, S. & Fatemi, E. Nonlinear total variation based noise removal algorithms. Phys. D: Nonlinear Phenom. 60, 259–268, DOI: https://doi.org/10.1016/0167-2789(92)90242-F (1992)
  - Own implementation
- __MLC / MSLC__: Mean Line Correlation / Mean Shifted Line Correlation
  - Schuppert et al., C. Whole-body magnetic resonance imaging in the large population-based german national cohort study: Predictive capability of automated image quality assessment for protocol repetitions. Investig. Radiol. 57 (2022).
  - Own implementation

## Segmentation

- __DICE__ : DICE
  - Dice, L. R. Measures of the amount of ecologic association between species. Ecology 26, 297–302, DOI: https://doi.org/10.2307/1932409 (1945).
  - Own implementation

## Testing Reference Metrics (Branch: New Metrics)

- __FSIM__: Feature Similarity
  - Zhang, L., Zhang, L., Mou ,X., Zhang, D. FSIM: A feature similarity index for image quality assessment. IEEE Transactions on Image Processing 24 (2015) 2579-2591
- __IWSSIM__: Information Content-Weighted SSIM
- __GMSD__: Gradient Magnitude Similarity Deviation
  - Xue, W., Zhang, L., Mou, X., Bovik, A.C. Gradient magnitude similarity deviation: A highly efficient perceptual image quality index. IEEE Transactions on Image Processing 23 (2013) 684-695
- __VIF__: Visual Information Fidelity
  - H.R. Sheikh, A.C. Bovik and G. de Veciana, "An information fidelity criterion for image quality assessment using natural scene statistics," IEEE Transactions on Image Processing , vol.14, no.12pp. 2117- 2128, Dec. 2005.
    (Sheikh, H.R., Bovik, A.C. Image Information and Visual Quality. IEEE Transactions on Image Processing 15 (2006) 430-444)
  - https://github.com/pavancm/Visual-Information-Fidelity---Python
- __VSI__: Visual Saliency-Induced Index
  - Zhang, L., Shen, Y., Li, H. VSI: A visual saliency-induced index for perceptual image quality assessment. IEEE Transactions on Image Processing 23 (2014) 4270-4281

# Distortions

- __Stripes__ Scales up the intensity of a single point in k-space, which leads to stripe artifacts
- __Bias Field__ Applies a synthetic bias field to the image
- __Ghosting__ Applies 2 ghosts in y-direction to the image
- __Gaussian Blur__ Applies a Gaussian filter to the image, which leads to blurring
- __Gaussian Noise__ Adds Gaussian noise to the image
- __Replace Artifact__ Mirrors a fraction of the upper half of the image to the lower half and replaces this part.
- __Shift Intensity__ Adds a fraction of the data range to all pixels
- __Gamma High__ Normalizes the image to range (0, 1). Takes all pixels to the power of gamma, with gamma > 1, and then scales back to the original data range. All image intensities become lower, the center of the histogram is effected the most.
- __Gamma Low__ Normalizes the image to range (0, 1). Takes all pixels to the power of gamma, with gamma \< 1, and then scales back to the original data range. All image intensities become higher, the center of the histogram is effected the most.
- __Translation__ Shifts the image by a fraction of the image shape in x- and y-direction
- __Elastic Deform__ Randomly displaces nodes of a grid on the image and linearly interpolates the image between the nodes, which creates an elastic deformation
