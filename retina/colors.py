from functools import lru_cache
import json
import math
import cv2 as cv
import numpy as np
import numpy.typing as npt

from retina import log

@lru_cache(maxsize=1)
def get_clahe():
  return cv.createCLAHE(tileGridSize=(16, 16), clipLimit=2.0)

def clahe(img: cv.typing.MatLike):
  clahe = get_clahe()
  return clahe.apply(img)

def hist_equalize(img: cv.typing.MatLike):
  return cv.equalizeHist(img)

  