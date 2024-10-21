from dataclasses import dataclass
import os
import pickle
import random
import sys
from typing import Sequence, cast

import numpy as np


from retina.face import FACIAL_EXPRESSION_MAPPER, FacialExpressionLabel
from retina.log import Ansi
from retina.size import Dimension

sys.path.append(os.getcwd())

import shutil
import zipfile

import tqdm
import cv2 as cv
import pandas as pd
import numpy.typing as npt
import sklearn.decomposition

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

list_data: list[npt.NDArray] = []
list_labels: list[int] = []
for entry in tqdm.tqdm(entries, desc="Building dataset from images"):
  original = cv.imread(entry.path)
  img = preprocess_image(original)
  canvas = img.copy()
  faces, _ = retina.face.extract_faces(img)

  if len(faces) == 0:
    continue

  for face in faces:
    face = cv.resize(face, retina.size.FACE_DIMENSIONS.tuple, interpolation=cv.INTER_CUBIC)
    canvas = cv.cvtColor(face, cv.COLOR_GRAY2BGR)
    landmark = retina.face.extract_face_landmarks(face, canvas=canvas)
    texture = retina.face.face_lbp(face, landmark, canvas=canvas)
    retina.debug.imdebug(canvas)

    list_data.append(texture)
    list_labels.append(entry.label.value)
  
if len(list_data) == 0:
  print(f"{Ansi.Error}No images were successfully processed into the dataset.{Ansi.End}")
  exit(1)

data = np.array(list_data)
# Normalization
data = data / data.sum(axis=1).reshape((-1, 1))

labels = np.array(list_labels).reshape((-1, 1))

dfdata = np.hstack((labels, data))


df = pd.DataFrame(dfdata, columns=[
  "label",
  *map(lambda idx: f'texture-{idx + 1}', range(data.shape[1])),
])

df.to_csv(retina.filesys.TRAINING_DATA_CSV_PATH, index=False)

