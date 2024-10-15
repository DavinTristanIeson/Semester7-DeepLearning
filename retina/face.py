from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import os
from typing import ClassVar, Sequence
import cv2 as cv
import urllib.request

import numpy as np
import numpy.typing as npt
import scipy
import scipy.spatial

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
  face_shape: list[npt.NDArray]
  eyes: list[npt.NDArray]
  eyebrows: list[npt.NDArray]
  nose: list[npt.NDArray]
  lips: list[npt.NDArray]
  dims: Dimension

  
  @property
  def feature_points(self)->npt.NDArray:
    return np.vstack([self.eyes, self.nose, self.lips])

  def as_feature_vector(self)->npt.NDArray:
    # https://arxiv.org/pdf/1812.04510
    # 17 points are dedicated for the shape of the face, which we don't really need.
    normalized_points = self.feature_points / np.array((self.dims.width, self.dims.height))
    interdistance_map = scipy.spatial.distance.cdist(normalized_points, normalized_points, "euclidean").flatten()
    # All diagonal values are excluded
    excluded_points = np.eye(len(self.feature_points)).flatten() == 1

    interdistance_map = interdistance_map[~excluded_points]

    # Also calculate the distance to the average point in the face
    average_point = normalized_points.mean(axis=0)
    distances_to_center = scipy.spatial.distance.cdist(np.array([average_point]), normalized_points, "euclidean")[0]

    feature_vector = np.hstack((interdistance_map, distances_to_center))

    return feature_vector


  EYE_COLOR: ClassVar[tuple[int, int, int]] = (0, 0, 255)
  LIP_COLOR: ClassVar[tuple[int, int, int]] = (0, 255, 0)
  NOSE_COLOR: ClassVar[tuple[int, int, int]] = (255, 0, 0)
  FACE_SHAPE_COLOR: ClassVar[tuple[int, int, int]] = (255, 255, 0)
  EYEBROW_COLOR: ClassVar[tuple[int, int, int]] = (0, 255, 255)
  def draw_on(self, img: cv.typing.MatLike, *, with_distances: bool = False)->npt.NDArray:
    canvas = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for point in self.eyes:
      cv.circle(canvas, tuple(point.astype(np.int32)), 1, self.EYE_COLOR, -1)
    for point in self.lips:
      cv.circle(canvas, tuple(point.astype(np.int32)), 1, self.LIP_COLOR, -1)
    for point in self.nose:
      cv.circle(canvas, tuple(point.astype(np.int32)), 1, self.NOSE_COLOR, -1)
    for point in self.face_shape:
      cv.circle(canvas, tuple(point.astype(np.int32)), 1, self.FACE_SHAPE_COLOR, -1)
    for point in self.eyebrows:
      cv.circle(canvas, tuple(point.astype(np.int32)), 1, self.EYEBROW_COLOR, -1)

    if with_distances:
      for i, point in enumerate(self.feature_points):
        for j, other in enumerate(self.feature_points):
          if i == j:
            continue
          cv.line(canvas, tuple(point.astype(np.int32)), tuple(other.astype(np.int32)), self.EYE_COLOR, 1)

    return canvas

def face_landmark_detection(img: cv.typing.MatLike)->FaceLandmark:
  # Source: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
  landmark_detector = get_face_landmark_detector()
  _, face_landmarks = landmark_detector.fit(img, np.array(((0, 0, img.shape[0], img.shape[1]),)))

  points: list[npt.NDArray] = face_landmarks[0][0]

  return FaceLandmark(
    face_shape=points[:17],
    eyebrows=points[17:27],
    nose=points[27:36],
    eyes=points[36:48],
    lips=points[48:],
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