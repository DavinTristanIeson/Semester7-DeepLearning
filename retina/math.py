import math
import cv2 as cv
import numpy as np


def clamp(value: int, min_value: int, max_value: int):
  return max(min_value, min(max_value, value))
def deg2rad(deg: float):
  return deg * 3.14 / 180.0
def rad2deg(rad: float):
  return rad * 180.0 / 3.14

def lerp(a: float, b: float, t: float):
  return a * (1.0 - t) + b * t

def blerp(dy: float, dx: float, rect: tuple[float, float, float, float]):
  # https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20172018/LectureNotes/IP/Images/ImageInterpolation.html
  rc, rc2, r2c, r2c2 = rect
  A = (1 - dy) * (1 - dx) * rc
  B = (1 - dy) * dx * rc2
  C = dy * (1 - dx) * r2c
  D = dy * dx * r2c2
  return A + B + C + D

def euclidean_distance(a: cv.typing.Point2f, b: cv.typing.Point2f):
  return math.sqrt(pow(a[0] - b[0], 2) * pow(a[1] - b[1], 2))
  

# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(z):
  """Compute softmax values for each sets of scores in x."""
  return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)  