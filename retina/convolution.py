from functools import lru_cache
from typing import Any, Callable, Union
import numpy as np
import numpy.typing as npt
import cv2 as cv

import retina
from retina.size import Point, Rectangle

GAUSSIAN_3X3_KERNEL = np.array([
  [1, 2, 1],
  [2, 4, 2],
  [1, 2, 1]
], dtype=float) * (1/16)

GAUSSIAN_5X5_KERNEL = np.array([
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1],
]) / 256

SHARPEN_KERNEL = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])

UNSHARP_MASKING_KERNEL = np.array([
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, -476, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1],
]) * -1 / 256

PREWITT_HORIZONTAL_KERNEL = np.array([
  [1, 1, 1],
  [0, 0, 0],
  [-1, -1, -1]
])
PREWITT_VERTICAL_KERNEL = np.array([
  [1, 0, -1],
  [1, 0, -1],
  [1, 0, -1]
])

SOBEL_HORIZONTAL_KERNEL = np.array([
  [1, 2, 1],
  [0, 0, 0],
  [-1, -2, -1]
])
SOBEL_VERTICAL_KERNEL = np.array([
  [1, 0, -1],
  [2, 0, -2],
  [1, 0, -1]
])

def prewitt(img: npt.NDArray):
  if len(img.shape) > 2:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  # https://en.wikipedia.org/wiki/Prewitt_operator
  # Float point conversion is necessary to avoid values overflowing
  float_img = img.astype(float)
  Gx:npt.NDArray = cv.filter2D(float_img, -1, PREWITT_VERTICAL_KERNEL, borderType=cv.BORDER_REFLECT)
  Gy:npt.NDArray = cv.filter2D(float_img, -1, PREWITT_HORIZONTAL_KERNEL, borderType=cv.BORDER_REFLECT)

  G = np.sqrt((Gx * Gx) + (Gy * Gy)).clip(0, 255).astype(np.uint8)
  # G: npt.NDArray = Gx + Gy
  return G, np.arctan2(Gy, Gx)

def sobel(img: npt.NDArray):
  if len(img.shape) > 2:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  # https://en.wikipedia.org/wiki/Sobel_operator
  float_img = img.astype(float)
  Gx:npt.NDArray = cv.filter2D(float_img, -1, SOBEL_VERTICAL_KERNEL, borderType=cv.BORDER_REFLECT).astype(float)
  Gy:npt.NDArray = cv.filter2D(float_img, -1, SOBEL_HORIZONTAL_KERNEL, borderType=cv.BORDER_REFLECT).astype(float)
  G = np.sqrt((Gx * Gx) + (Gy * Gy)).clip(0, 255).astype(np.uint8)
  # G: npt.NDArray = Gx + Gy
  return G, np.arctan2(Gy, Gx)

@lru_cache(1)
def gabor_kernel():
  # https://www.freedomvc.com/index.php/2021/10/16/gabor-filter-in-edge-detection/

  filters = []
  num_filters = 16
  ksize = 35 # The local area to evaluate
  sigma = 3.0  # Larger Values produce more edges
  lambd = 10.0
  gamma = 0.5
  psi = 0  # Offset value - lower generates cleaner results
  for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
      kern = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_64F)
      kern /= 1.0 * kern.sum()  # Brightness normalization
      filters.append(kern)
  return filters

def gabor(img):
  # https://www.freedomvc.com/index.php/2021/10/16/gabor-filter-in-edge-detection/
  # Gabor is used in this https://worldscientific.com/doi/epdf/10.1142/S0219691320500034

  filters = gabor_kernel()

  # This general function is designed to apply filters to our image
  # First create a numpy array the same size as our input image
  newimage = np.zeros_like(img)
     
  # Starting with a blank image, we loop through the images and apply our Gabor Filter
  # On each iteration, we take the highest value (super impose), until we have the max value across all filters
  # The final image is returned
  depth = -1 # remain depth same as original image
     
  for kern in filters:  # Loop through the kernels in our GaborFilter
    image_filter = cv.filter2D(img, depth, kern)  # Apply filter to image
    # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
    np.maximum(newimage, image_filter, newimage)
  return newimage
