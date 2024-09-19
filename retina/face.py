from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence
import cv2 as cv

import retina
from retina.size import Rectangle

CASCADE_CLASSIFIER_PATH = "retina/haarcascade_frontalface_default.xml"

@lru_cache(maxsize=1)
def get_face_haar_classifier():
  return cv.CascadeClassifier(CASCADE_CLASSIFIER_PATH)

def haar_detect(img: cv.typing.MatLike)->Sequence[Rectangle]:
  # Source: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
  face_cascade = get_face_haar_classifier()
  face_coordinates = face_cascade.detectMultiScale(img, 1.1, 4)

  rectangles = list(Rectangle.from_tuple(coords) for coords in face_coordinates)
  rectangles.sort(key=lambda x: x.area)
  return rectangles
