from typing import Callable, Sequence
import cv2 as cv
import numpy.typing as npt
import numpy as np

from retina import convolution
from retina.cvutil import mask_matrix
from retina.size import *

BOX_KERNEL_3X3 = np.ones((3, 3))
BOX_KERNEL_5X5 = np.ones((5, 5))
ROUND_KERNEL_3X3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
ROUND_KERNEL_5X5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

def open(img: cv.typing.MatLike, dilate_kernel: cv.typing.MatLike = ROUND_KERNEL_3X3, erode_kernel: cv.typing.MatLike = ROUND_KERNEL_3X3, *, dilate_iteration: int = 1, erode_iteration: int = 1):
  # https://www.geeksforgeeks.org/python-opencv-morphological-operations/
  img = cv.erode(img, erode_kernel, iterations=erode_iteration)
  img = cv.dilate(img, dilate_kernel, iterations=dilate_iteration)
  return img

def close(img: cv.typing.MatLike, dilate_kernel: cv.typing.MatLike = ROUND_KERNEL_3X3, erode_kernel: cv.typing.MatLike = ROUND_KERNEL_3X3, *, dilate_iteration: int = 1, erode_iteration: int = 1):
  img = cv.dilate(img, dilate_kernel, iterations=dilate_iteration)
  img = cv.erode(img, erode_kernel, iterations=erode_iteration)
  return img

def gradient(img: cv.typing.MatLike, dilate_kernel: cv.typing.MatLike = ROUND_KERNEL_3X3, erode_kernel: cv.typing.MatLike = ROUND_KERNEL_3X3, *, dilate_iteration: int = 1, erode_iteration: int = 1):
  img_dilate: npt.NDArray = cv.dilate(img, dilate_kernel, iterations=dilate_iteration)
  img_erode: npt.NDArray = cv.erode(img, erode_kernel, iterations=erode_iteration)
  return np.abs(img_dilate - img_erode)

def harris_corner(img: cv.typing.MatLike):
  # https://www.geeksforgeeks.org/python-corner-detection-with-harris-corner-detection-method-using-opencv/
  # https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
  corners: npt.NDArray = cv.cornerHarris(img, 2, 3, 0.058)
  return cv.dilate(corners, BOX_KERNEL_3X3)

def fill_contours(contours: Sequence[cv.typing.MatLike], shape: Sequence[int], *, min_dims: Optional[Dimension] = None):
  result = np.zeros(shape, dtype=np.uint8)
  # https://stackoverflow.com/questions/1716274/fill-the-holes-in-opencv
  for idx, contour in enumerate(contours):
    if min_dims is not None:
      bbox = Rectangle.from_tuple(cv.boundingRect(contour)).dimensions
      # No need to fill in contours that are too small
      if not bbox.can_encapsulate(min_dims):
        continue
    cv.drawContours(result, contours, idx, 255.0, -1) # type: ignore
  # Remove noise
  result = cv.medianBlur(result, 5)
  result = close(result)
  return result

def fill_holes(img: npt.NDArray, seed: Point):
  # https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
  # https://stackoverflow.com/questions/16705721/opencv-floodfill-with-mask
  flood_flags = 4 | cv.FLOODFILL_MASK_ONLY | (255 << 8)
  # Size needs to be w+2 x h+2
  mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
  mask[1:img.shape[0]+1, 1:img.shape[1]+1] = img
  cv.floodFill(img, mask, (seed.x, seed.y), (255,), (10,), (10,), flood_flags)
  output = img | cv.bitwise_not(mask[1:img.shape[0]+1, 1:img.shape[1]+1])
  return output

def optimal_fill_holes(img: npt.NDArray, skips: tuple[int, int]):
  total_pixels = Dimension.from_shape(img.shape).area
  # Black pixels that has a potential to be filled
  available_pixels = total_pixels - cv.countNonZero(img)
  # There's no need to sample EVERY point. Get them in intervals according to ``SKIP``
  for r in range(0, img.shape[0], skips[0]):
    for c in range(0, img.shape[1], skips[1]):
      if img[r, c] != 0:
        continue
      result = fill_holes(img, Point.cell(r, c))
      # Remaining black pixels, hopefully the background
      leftover_pixels = total_pixels - cv.countNonZero(result)

      # Only up to 30% of the available pixels can be filled
      if leftover_pixels >= available_pixels * 0.7:
        result = cv.erode(result, ROUND_KERNEL_5X5)
        return result
  return img

def skeletonize(img: cv.typing.MatLike):
  return cv.ximgproc.thinning(img, thinningType=cv.ximgproc.THINNING_GUOHALL)