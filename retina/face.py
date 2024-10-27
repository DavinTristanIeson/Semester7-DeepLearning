from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import os
from typing import ClassVar, Optional, Sequence, cast

import scipy.ndimage
import cv2 as cv
import skimage
import urllib.request

import numpy as np
import numpy.typing as npt
import scipy
import scipy.spatial

import retina
from retina.log import Ansi
from retina.size import Dimension, Point, Rectangle

CASCADE_CLASSIFIER_PATH = "retina/haarcascade_frontalface_default.xml"

@lru_cache(maxsize=1)
def get_face_haar_classifier():
  return cv.CascadeClassifier(CASCADE_CLASSIFIER_PATH)

def haar_detect(img: cv.typing.MatLike)->Sequence[Rectangle]:
  # Source: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
  face_cascade = get_face_haar_classifier()
  face_coordinates = face_cascade.detectMultiScale(img)

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
  face_shape: npt.NDArray
  eyes: npt.NDArray
  eyebrows: npt.NDArray
  nose: npt.NDArray
  lips: npt.NDArray
  dims: Dimension

  @property
  def feature_points(self):
    return np.vstack([
      self.eyes[0],
      np.array([self.eyes[1], self.eyes[2]]).mean(axis=0),
      self.eyes[3],
      np.array([self.eyes[4], self.eyes[5]]).mean(axis=0),

      self.eyes[6],
      np.array([self.eyes[7], self.eyes[8]]).mean(axis=0),
      self.eyes[9],
      np.array([self.eyes[10], self.eyes[11]]).mean(axis=0),

      *self.eyebrows[0:5:2],
      *self.eyebrows[5:10:2],
      self.lips[0],
      self.lips[3],
      self.lips[6],
      self.lips[9],
    ])

  def as_feature_vector(self)->npt.NDArray:
    # https://arxiv.org/pdf/1812.04510
    normalized_points = self.feature_points / np.array((self.dims.width, self.dims.height))
    interdistance_map = scipy.spatial.distance.cdist(normalized_points, normalized_points, "euclidean").flatten()
    interdistance_map = np.power(interdistance_map, 2)
    # # All diagonal values are excluded
    excluded_points = np.eye(len(normalized_points)).flatten() == 1

    # # Square the interdistance map to make larger differences more prominent
    return interdistance_map.flatten()[~excluded_points]


  EYE_COLOR: ClassVar[tuple[int, int, int]] = (0, 0, 255)
  LIP_COLOR: ClassVar[tuple[int, int, int]] = (0, 255, 0)
  NOSE_COLOR: ClassVar[tuple[int, int, int]] = (255, 0, 0)
  FACE_SHAPE_COLOR: ClassVar[tuple[int, int, int]] = (255, 255, 0)
  EYEBROW_COLOR: ClassVar[tuple[int, int, int]] = (0, 255, 255)

  def project_point(self, point: npt.NDArray, rect: Optional[Rectangle] = None)->Sequence[int]:
    if rect is None:
      return cast(Sequence[int], point.astype(np.int32))
    projected_point = point * rect.dimensions.ndarray / self.dims.ndarray
    return cast(Sequence[int], (projected_point + rect.p0.ndarray).astype(np.int32))
  
  def draw_on(self, img: cv.typing.MatLike, *, offset: Optional[Rectangle] = None):
    for point in self.eyes:
      cv.circle(img, self.project_point(point, offset), 1, self.EYE_COLOR, -1)
    for point in self.lips:
      cv.circle(img, self.project_point(point, offset), 1, self.LIP_COLOR, -1)
    for point in self.nose:
      cv.circle(img, self.project_point(point, offset), 1, self.NOSE_COLOR, -1)
    for point in self.face_shape:
      cv.circle(img, self.project_point(point, offset), 1, self.FACE_SHAPE_COLOR, -1)
    for point in self.eyebrows:
      cv.circle(img, self.project_point(point, offset), 1, self.EYEBROW_COLOR, -1)

def face_landmark_detection(img: cv.typing.MatLike)->FaceLandmark:
  # Source: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
  landmark_detector = get_face_landmark_detector()
  _, face_landmarks = landmark_detector.fit(img, np.array(((0, 0, img.shape[0], img.shape[1]),)))

  points: npt.NDArray = face_landmarks[0][0]

  return FaceLandmark(
    face_shape=points[:17],
    eyebrows=points[17:27],
    nose=points[27:36],
    eyes=points[36:48],
    lips=points[48:],
    dims=Dimension.from_shape(img.shape)
  )

def lbp_histograms(img: cv.typing.MatLike, rectangles: Sequence[Rectangle], *, canvas: Optional[cv.typing.MatLike] = None, offset: Optional[Rectangle] = None)->npt.NDArray:
  if canvas is not None:
    retina.debug.draw_rectangles(canvas, rectangles, offset=offset.p0 if offset else None)
  
  histograms: list[npt.NDArray] = []
  BIN_COUNT = 6
  for rect in rectangles:
    chunk = img[rect.slice]
    radius = 1

    if chunk.size == 0:
      histograms.append(np.full((BIN_COUNT,), 0))
      continue
    lbp: npt.NDArray = skimage.feature.local_binary_pattern(chunk, 8 * radius, radius)
    histograms.append(scipy.ndimage.histogram(lbp, 0, 255, BIN_COUNT) / lbp.size)

  return np.hstack(histograms)

def face_lbp(img: cv.typing.MatLike, landmark: FaceLandmark, *, canvas: Optional[cv.typing.MatLike] = None, offset: Optional[Rectangle] = None):
  prominent_points = [
    landmark.eyes[0], # left eye left corner
    landmark.eyes[3], # left eye right corner
    landmark.eyes[6], # right eye left corner
    landmark.eyes[9], # right eye right corner
    landmark.eyebrows[0], # Eyebrow left left corner,
    landmark.eyebrows[4], # Eyebrow left right corner,
    landmark.eyebrows[-4], # Eyebrow right left corner,
    landmark.eyebrows[-1], # Eyebrow right right corner
    landmark.lips[0], # Lips left corner
    landmark.lips[6], # Lips right corner
    landmark.lips[3], # Lips top corner
    landmark.lips[9], # Lips bottom corner
  ]
  prominent_rects = tuple(map(
    lambda x: Rectangle.around(Point(int(x[0]), int(x[1])), Dimension(12, 12)),
    prominent_points
  ))
  return lbp_histograms(img, prominent_rects, canvas=canvas, offset=offset)

def grid_lbp(img: cv.typing.MatLike, *, canvas: Optional[cv.typing.MatLike] = None, offset: Optional[Rectangle] = None):
  dims = Dimension.from_shape(img.shape)
  grid_rects = dims.partition(8, 8)
  return lbp_histograms(img, grid_rects, canvas=canvas, offset=offset)

class FacialExpressionLabels:
  Ours = ["angry", "disgusted", "happy", "neutral", "sad", "surprised"]
  Fer2013 = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"] 
  CkPlus = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral", "Contempt"]
  
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

def face_alignment(img: cv.typing.MatLike, landmark: FaceLandmark):
  # https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
  desired_left_eye = retina.size.FACE_DIMENSIONS.sample(0.22, 0.25)
  desired_right_eye_x = retina.size.FACE_DIMENSIONS.width - desired_left_eye.x

  left_eye_avg = landmark.eyes[0:6].mean(axis=0)
  right_eye_avg = landmark.eyes[6:].mean(axis=0)

  delta = right_eye_avg - left_eye_avg
  angle = np.degrees(np.arctan2(delta[1], delta[0]))

  dist = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
  desired_dist = desired_right_eye_x - desired_left_eye.x
  scale = desired_dist / dist

  eyes_center = np.array([left_eye_avg, right_eye_avg]).mean(axis=0)
  rotation_matrix = cv.getRotationMatrix2D(eyes_center, angle, scale)

  translation_x = retina.size.FACE_DIMENSIONS.width * 0.5
  translation_y = desired_left_eye.y
  rotation_matrix[0, 2] += (translation_x - eyes_center[0])
  rotation_matrix[1, 2] += (translation_y - eyes_center[1])

  img = cv.warpAffine(img, rotation_matrix, retina.size.FACE_DIMENSIONS.tuple, flags=cv.INTER_CUBIC)

  return img

def extract_face_landmarks(img: cv.typing.MatLike, *, canvas: Optional[cv.typing.MatLike] = None, offset: Optional[Rectangle] = None):
  face = cv.filter2D(img, -1, retina.convolution.GAUSSIAN_3X3_KERNEL)
  face = cv.filter2D(img, -1, retina.convolution.SHARPEN_KERNEL)
  face_landmarks = retina.face.face_landmark_detection(face)
  
  if canvas is not None:
    face_landmarks.draw_on(canvas, offset=offset)

  return face_landmarks

def preprocess_face_image(original: cv.typing.MatLike):
  img = retina.cvutil.resize_image(original, retina.size.STANDARD_DIMENSIONS) # Resize
  if img.ndim > 2:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Grayscale
  img = retina.colors.clahe(img) # Contrast adjustment
  return img

def face2vec(original: cv.typing.MatLike, *, canvas: Optional[cv.typing.MatLike] = None, skip_face_detection: bool = False, skip_face_landmarking: bool = False)->Optional[npt.NDArray]:
  img = preprocess_face_image(original)
  if not skip_face_detection:
    faces, face_rects = extract_faces(img, canvas=canvas)
  else:
    faces = [img]
    face_rects = [Rectangle.with_dimensions(Dimension.from_shape(original.shape))]

  features: list[npt.NDArray] = []
  for face, face_rect in zip(faces, face_rects):
    face = cv.resize(face, retina.size.FACE_DIMENSIONS.tuple, interpolation=cv.INTER_CUBIC)
    if not skip_face_landmarking:
      landmark = extract_face_landmarks(face, canvas=canvas, offset=face_rect)
      face = face_alignment(face, landmark)
    feature_vector = grid_lbp(face)
    feature_vector = feature_vector / feature_vector.sum() # Normalization
    features.append(feature_vector)

  if len(features) == 0:
    return None
  
  return np.array(features)