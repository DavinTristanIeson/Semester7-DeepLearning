import argparse
from typing import Callable, ClassVar
import cv2 as cv
import sys

import numpy as np
from retina.size import *

IS_PREVIEW = "--preview" in sys.argv
IS_SAVE = "--save" in sys.argv

def wait_until_esc():
  while (cv.waitKey(5) != 27): pass

def splice_matrix(dest: cv.typing.MatLike, src: cv.typing.MatLike, rect: Rectangle):
  dest_rect = rect.clamp(Dimension.from_shape(dest.shape))
  src_rect = Rectangle.with_dimensions(Dimension.from_shape(src.shape))\
    .clamp(dest_rect.dimensions)\
    .translate(dest_rect.x0 - rect.x0, dest_rect.y0 - rect.y0)
  dest[dest_rect.slice] = src[src_rect.slice]

def mask_matrix(dest: cv.typing.MatLike, src: cv.typing.MatLike, rect: Rectangle):
  dest_rect = rect.clamp(Dimension.from_shape(dest.shape))
  src_rect = Rectangle.with_dimensions(Dimension.from_shape(src.shape))\
    .clamp(dest_rect.dimensions)\
    .translate(dest_rect.x0 - rect.x0, dest_rect.y0 - rect.y0)
  src_mask = src[src_rect.slice].astype(bool)
  dest_mask = np.zeros(dest.shape, dtype=bool)
  dest_mask[dest_rect.slice] = src_mask
  return dest_mask

def resize_image(img: cv.typing.MatLike, target_dims: Dimension):
  dimensions = Dimension(img.shape[1], img.shape[0])\
    .resize(width=target_dims.width, height=target_dims.height)
  img = cv.resize(img, dimensions.tuple, interpolation=cv.INTER_LINEAR)
  dimensions = Dimension(img.shape[1], img.shape[0])
  rectangle = Rectangle.around(dimensions.center, target_dims)
  return img[rectangle.slice]

def rotate_image(img: cv.typing.MatLike, angle: float):
  dims = Dimension.from_shape(img.shape)
  center = dims.center
  rot_mat = cv.getRotationMatrix2D(center.tuple, angle, 1.0)
  result = cv.warpAffine(img, rot_mat, dims.tuple, flags=cv.INTER_LINEAR)
  return result

def translate_image(img: cv.typing.MatLike, vector: tuple[float, float]):
  # https://www.geeksforgeeks.org/image-translation-using-opencv-python/
  dims = Dimension.from_shape(img.shape)    
  T = np.array([[1, 0, vector[0]], [0, 1, vector[1]]], dtype=np.float32) 
  return cv.warpAffine(img, T, dims.tuple)