from dataclasses import dataclass
import os
import sys
from typing import Sequence, cast


from retina.face import FACIAL_EXPRESSION_MAPPER, FacialExpressionLabel
from retina.log import Ansi
from retina.size import Dimension

sys.path.append(os.getcwd())

import shutil
import zipfile

import tqdm
import cv2 as cv
import pandas as pd
import tensorflow as tf
import keras

import retina

if (os.path.exists(retina.filesys.DATA_DIR_PATH)):
  shutil.rmtree(retina.filesys.DATA_DIR_PATH)
with zipfile.ZipFile(retina.filesys.DATASET_PATH, 'r') as zip_ref:
  zip_ref.extractall(retina.filesys.DATA_DIR_PATH)

def preprocess_image(img: cv.typing.MatLike):
  img = retina.cvutil.resize_image(original, retina.size.STANDARD_DIMENSIONS) # Resize
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Grayscale
  img = retina.colors.clahe(img) # Contrast adjustment
  return img

def extract_faces(img: cv.typing.MatLike)->Sequence[cv.typing.MatLike]:
  face_positions = retina.face.haar_detect(img)
  faces = tuple(
    img[pos.slice]
    for pos in face_positions
  )

  retina.debug.imdebug(retina.debug.draw_rectangles(img, face_positions))

  saved_faces: list[cv.typing.MatLike] = []
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
  return faces

def extract_face_landmarks(img: cv.typing.MatLike):
  face = cv.filter2D(img, -1, retina.convolution.GAUSSIAN_3X3_KERNEL)
  face = cv.filter2D(img, -1, retina.convolution.SHARPEN_KERNEL)
  face_landmarks = retina.face.face_landmark_detection(face)
  retina.debug.imdebug(retina.face.draw_face_landmarks(face, face_landmarks))

  # Normalize points
  dims = Dimension.from_shape(face.shape)
  normalized_face_landmarks = tuple(map(lambda point: point.normalized(dims),  face_landmarks))

  return normalized_face_landmarks

@dataclass
class TrainDataEntry:
  path: str
  label: FacialExpressionLabel


entries: list[TrainDataEntry] = []
for folder in os.scandir(retina.filesys.DATA_DIR_PATH):
  if not folder.is_dir():
    continue
  expression = FACIAL_EXPRESSION_MAPPER.get(folder.name, None)
  if expression is None:
    print(f"{Ansi.Error}Skipping the inclusion of {folder.name} in the dataset.{Ansi.End}")
    continue
  entries.extend(map(
    lambda fpath: TrainDataEntry(path=fpath, label=cast(FacialExpressionLabel, expression)),
    retina.filesys.get_files_in_folder(folder.path)
  ))

# To be considered
data_augmentation = keras.Sequential([
  keras.layers.RandomFlip("horizontal"),
  keras.layers.RandomRotation(0.05),
  keras.layers.RandomTranslation(0.1, 0.1),
  keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
  keras.layers.RandomContrast(0.1),
])

rows: list[list[float]] = []
for entry in tqdm.tqdm(entries, desc="Building dataset from images"):
  original = cv.imread(entry.path)
  img = preprocess_image(original)
  faces = extract_faces(img)

  for original_face in faces:
    for i in range(10):
      face = data_augmentation(original_face, training=True)
      print(i)
      landmarks = list(map(extract_face_landmarks, faces))
      for landmark in landmarks:
        row: list[float] = [entry.label.value]
        for idx, point in enumerate(landmark):
          row.append(point.x)
          row.append(point.y)
        rows.append(row)
  
if len(rows) == 0:
  print(f"{Ansi.Error}No images were successfully processed into the dataset.{Ansi.End}")
  exit(1)

df = pd.DataFrame(rows, columns=[
  "label",
  *map(lambda idx: f'x{idx // 2}' if idx % 2 == 0 else f'y{idx // 2}', range(len(rows[0]) - 1))
])

df.to_csv(retina.filesys.DATA_CSV_PATH, index=False)

