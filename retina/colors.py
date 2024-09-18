import json
import math
import cv2 as cv
import numpy as np
import numpy.typing as npt

from retina import log

def contrast(img: cv.typing.MatLike, contrast: float, brightness: float=1):
  # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
  return cv.addWeighted(img, contrast, img, 0, brightness)

def get_autocontrast_values(img: cv.typing.MatLike)->tuple[float, float]:
  # https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape/56909036
  # https://stackoverflow.com/questions/9744255/instagram-lux-effect/9761841#9761841
  
  # Get intensity/brightness/value channel
  img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
  hist: npt.NDArray = img[:,:,2].flatten()
  # mean, std
  mean = hist.mean()
  std = hist.std()
  # Get 5th or 95th percentile
  min, max = mean - std * 2, mean + std * 2

  # solve linear equations a * min + b = 0 and a * max + b = 255
  lineq_solution = min / max
  beta = (-255 * lineq_solution) / (1 - lineq_solution)
  alpha = (255 - beta) / max

  return (alpha, beta)

def get_autobrightness_values(img:cv.typing.MatLike, *, debug: bool = False)->float:
  # https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv
  grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  if debug:
    log.matshow(grey)
    print(f"Sum: {grey.sum()}")
  brightness = grey.sum() / (255 * img.shape[0] * img.shape[1])
  min_brightness = 0.5
  if brightness == 0:
    return 1
  ratio = min_brightness / brightness
  if debug:
    print(f"Brightness: {brightness}", f"Min Brightness: {min_brightness}", f"Ratio: {ratio}", sep='\n')
  return ratio

def gamma_correction(img: npt.NDArray, gamma: float = 1):
  # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
  # https://learnopengl.com/Advanced-Lighting/Gamma-Correction
  # https://blog.johnnovak.net/2016/09/21/what-every-coder-should-know-about-gamma/
  lookup_table = np.zeros((1,256), np.uint8)
  for i in range(256):
    lookup_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
  return cv.LUT(img, lookup_table)

def hist_match(source: npt.NDArray, template: npt.NDArray):
  # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

  oldshape = source.shape
  source = source.ravel()
  template = template.ravel()

  # get the set of unique pixel values and their corresponding indices and
  # counts
  s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                          return_counts=True)
  t_values, t_counts = np.unique(template, return_counts=True)

  # take the cumsum of the counts and normalize by the number of pixels to
  # get the empirical cumulative distribution functions for the source and
  # template images (maps pixel value --> quantile)
  s_quantiles = np.cumsum(s_counts).astype(np.float64)
  s_quantiles /= s_quantiles[-1]
  t_quantiles = np.cumsum(t_counts).astype(np.float64)
  t_quantiles /= t_quantiles[-1]

  # interpolate linearly to find the pixel values in the template image
  # that correspond most closely to the quantiles in the source image
  interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

  result = interp_t_values[bin_idx].reshape(oldshape).clip(0, 255).astype(np.uint8)
  result.flat[source == 0] = 0
  return result