from functools import lru_cache
import cv2 as cv


@lru_cache(maxsize=1)
def get_clahe():
  return cv.createCLAHE(tileGridSize=(8, 8), clipLimit=2.0)

def clahe(img: cv.typing.MatLike):
  clahe = get_clahe()
  return clahe.apply(img)

def brighten(img: cv.typing.MatLike, brightness: int):
  return cv.convertScaleAbs(img, img, 1, brightness)

def hist_equalize(img: cv.typing.MatLike):
  return cv.equalizeHist(img)

  