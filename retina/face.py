from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import os
import pickle
from typing import ClassVar, Optional, Sequence
import cv2 as cv
import urllib.request

import numpy as np
import numpy.typing as npt
import scipy
import scipy.spatial
import sklearn
import sklearn.decomposition

import retina
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
    return np.vstack([self.eyes, self.eyebrows, self.nose, self.lips])

  def as_feature_vector(self)->npt.NDArray:
    # https://arxiv.org/pdf/1812.04510
    # 17 points are dedicated for the shape of the face, which we don't really need.
    normalized_points = self.feature_points / np.array((self.dims.width, self.dims.height))
    interdistance_map = scipy.spatial.distance.cdist(normalized_points, normalized_points, "euclidean").flatten()
    # All diagonal values are excluded
    excluded_points = np.eye(len(self.feature_points)).flatten() == 1

    # Square the interdistance map to make larger differences more prominent
    interdistance_map = np.power(interdistance_map[~excluded_points], 2)

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
  def draw_on(self, img: cv.typing.MatLike, *, offset: Point = Point(0, 0)):
    offsetnp = offset.nparray
    for point in self.eyes:
      cv.circle(img, tuple(point.astype(np.int32) + offsetnp), 1, self.EYE_COLOR, -1)
    for point in self.lips:
      cv.circle(img, tuple(point.astype(np.int32) + offsetnp), 1, self.LIP_COLOR, -1)
    for point in self.nose:
      cv.circle(img, tuple(point.astype(np.int32) + offsetnp), 1, self.NOSE_COLOR, -1)
    for point in self.face_shape:
      cv.circle(img, tuple(point.astype(np.int32) + offsetnp), 1, self.FACE_SHAPE_COLOR, -1)
    for point in self.eyebrows:
      cv.circle(img, tuple(point.astype(np.int32) + offsetnp), 1, self.EYEBROW_COLOR, -1)

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
  
  @staticmethod
  def target_names():
    return tuple(map(lambda x: x.name, sorted(FacialExpressionLabel.__members__.values(), key=lambda x: x.value)))



FACIAL_EXPRESSION_MAPPER: dict[str, FacialExpressionLabel] = {
  "angry": FacialExpressionLabel.Angry,
  "disgusted": FacialExpressionLabel.Disgusted,
  "happy": FacialExpressionLabel.Happy,
  "neutral": FacialExpressionLabel.Neutral,
  "sad": FacialExpressionLabel.Sad,
  "surprised": FacialExpressionLabel.Surprised,
}

INVERSE_FACIAL_EXPRESSION_MAPPER: dict[FacialExpressionLabel, str] = {v:k for k, v in FACIAL_EXPRESSION_MAPPER.items()}

def extract_faces(img: cv.typing.MatLike, *, canvas: Optional[cv.typing.MatLike] = None)->tuple[Sequence[cv.typing.MatLike], Sequence[Rectangle]]:
  face_positions = haar_detect(img)
  faces = tuple(
    img[pos.slice]
    for pos in face_positions
  )

  # retina.debug.imdebug(retina.debug.draw_rectangles(img, face_positions))

  saved_faces: list[cv.typing.MatLike] = []
  saved_face_positions: list[Rectangle] = []
  for i in range(len(faces)):
    face = faces[i]
    rect = face_positions[i]
    is_overlapping = False
    for j in range(0, i):
      other_rect = face_positions[j]
      # Don't grab overlapping squares
      IOU = rect.intersection_with_union(other_rect)
      if IOU > 0.4:
        is_overlapping = True
        break

    if not is_overlapping:
      saved_faces.append(face)
      saved_face_positions.append(rect)

  if canvas is not None:
    for facepos in saved_face_positions:
      cv.rectangle(canvas, facepos.tuple, (0, 255, 0), 1)
  return saved_faces, saved_face_positions

def extract_face_landmarks(img: cv.typing.MatLike, *, canvas: Optional[cv.typing.MatLike] = None, offset: Point = Point(0, 0)):
  face = cv.filter2D(img, -1, retina.convolution.GAUSSIAN_3X3_KERNEL)
  face = cv.filter2D(img, -1, retina.convolution.SHARPEN_KERNEL)
  face_landmarks = retina.face.face_landmark_detection(face)
  
  feature_vector = face_landmarks.as_feature_vector()
  if canvas is not None:
    face_landmarks.draw_on(canvas, offset=offset)

  return feature_vector

FEATURE_DIMENSIONS = 20

@lru_cache(maxsize=1)
def get_trained_pca_model()->sklearn.decomposition.PCA:
  if not os.path.exists(retina.filesys.PCA_MODEL_PATH):
    raise Exception("Model has not been trained yet. Check")
  with open(retina.filesys.PCA_MODEL_PATH, 'rb') as f:
    return pickle.load(f)

def face2vec(original: cv.typing.MatLike, *, canvas: Optional[cv.typing.MatLike] = None)->Optional[npt.NDArray]:
  img = retina.cvutil.resize_image(original, retina.size.STANDARD_DIMENSIONS) # Resize
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Grayscale
  img = retina.colors.clahe(img) # Contrast adjustment
  pca_model = get_trained_pca_model()


  faces, face_rects = extract_faces(img, canvas=canvas)

  landmarks: list[npt.NDArray] = []
  for face, face_rect in zip(faces, face_rects):
    landmark = extract_face_landmarks(face, canvas=canvas, offset=face_rect.p0)
    landmarks.append(landmark)

  if len(landmarks) == 0:
    return None
  
  feature_vectors = pca_model.transform(np.array(landmarks))
    
  return feature_vectors