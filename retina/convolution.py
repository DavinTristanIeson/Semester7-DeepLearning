from typing import Any, Callable, Union
import numpy as np
import numpy.typing as npt
import cv2 as cv

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