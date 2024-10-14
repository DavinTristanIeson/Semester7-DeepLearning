from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import os
from typing import ClassVar, Sequence
import cv2 as cv
import urllib.request

import numpy as np
import numpy.typing as npt

from retina.log import Ansi
from retina.size import Dimension, FloatingPoint, Point, Rectangle

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

LBFmodel_path = "retina/lbfmodel.yaml"
@lru_cache(1)
def get_face_landmark_detector()->cv.face.Facemark:
  # https://medium.com/analytics-vidhya/facial-landmarks-and-face-detection-in-python-with-opencv-73979391f30e
  if not os.path.exists(LBFmodel_path):
    print(f"{Ansi.Warning}Cannot find any LBFmodel installation. Installing to {LBFmodel_path}{Ansi.End}")
    urllib.request.urlretrieve("https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml", LBFmodel_path)

  landmark = cv.face.createFacemarkLBF()
  landmark.loadModel(LBFmodel_path)
  return landmark

def many_face_landmark_detection(img: cv.typing.MatLike, faces: Sequence[Rectangle])->Sequence[Sequence[Point]]:
  # Source: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
  landmark_detector = get_face_landmark_detector()
  _, face_landmarks = landmark_detector.fit(img, np.array(list(map(lambda face: face.tuple, faces))))
  print(face_landmarks)

  wrapped_landmarks: list[list[Point]] = []
  for face in face_landmarks:
    points: list[Point] = []
    for point in face[0]:
      points.append(Point(point[0], point[1]))
    wrapped_landmarks.append(points)

  return wrapped_landmarks

@dataclass
class FaceLandmark:
  face_shape: list[Point]
  eyes: list[Point]
  nose: list[Point]
  lips: list[Point]
  dims: Dimension

  
  @property
  def points(self):
    return [*self.face_shape, *self.eyes, *self.nose, *self.lips]

  
  def as_feature_vector(self, normalize: bool)->npt.NDArray:
    features = []
    for point in self.points:
      if normalize:
        point = point.normalized(self.dims)
      features.append(point.x)
      features.append(point.y)
    return np.array(features)
  
  def draw_on(self, img: cv.typing.MatLike)->npt.NDArray:
    canvas = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for idx, point in enumerate(self.points):
      cv.circle(canvas, (int(point.x), int(point.y)), 1, (0, 255, 0), -1)
    return canvas

def face_landmark_detection(img: cv.typing.MatLike)->FaceLandmark:
  # Source: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
  landmark_detector = get_face_landmark_detector()
  _, face_landmarks = landmark_detector.fit(img, np.array(((0, 0, img.shape[0], img.shape[1]),)))

  points: list[Point] = []
  for point in face_landmarks[0][0]:
    points.append(Point(point[0], point[1]))

  return FaceLandmark(
    face_shape=points[:17],
    eyes=points[17:-32],
    nose=points[-32:-20],
    lips=points[-20:],
    dims=Dimension.from_shape(img.shape)
  )

class FacialExpressionLabel(Enum):
  Angry = 0
  Disgusted = 1
  Happy = 2
  Neutral = 3
  Sad = 4
  Surprised = 5


FACIAL_EXPRESSION_MAPPER: dict[str, FacialExpressionLabel] = {
  "angry": FacialExpressionLabel.Angry,
  "disgusted": FacialExpressionLabel.Disgusted,
  "happy": FacialExpressionLabel.Happy,
  "neutral": FacialExpressionLabel.Neutral,
  "sad": FacialExpressionLabel.Sad,
  "surprised": FacialExpressionLabel.Surprised,
}