from dataclasses import dataclass
import os
import sys
from typing import cast

import numpy as np


from retina.face import FacialExpressionLabels
from retina.log import Ansi

sys.path.append(os.getcwd())

import shutil
import zipfile

import tqdm
import cv2 as cv
import pandas as pd
import numpy.typing as npt

import retina

if "--unzip" in sys.argv:
  if (os.path.exists(retina.filesys.DATA_DIR_PATH)):
    shutil.rmtree(retina.filesys.DATA_DIR_PATH)
  with zipfile.ZipFile("fer2013.zip", 'r') as zip_ref:
    zip_ref.extractall(retina.filesys.DATA_DIR_PATH)

@dataclass
class TrainDataEntry:
  path: str
  label: int


entries: list[TrainDataEntry] = []
for folder in os.scandir(os.path.join(retina.filesys.DATA_DIR_PATH, "test")):
  if not folder.is_dir():
    continue

  try:
    expression = FacialExpressionLabels.Fer2013.index(folder.name)
  except ValueError:
    print(f"{Ansi.Error}Skipping the inclusion of {folder.name} in the dataset.{Ansi.End}")
    continue

  entries.extend(map(
    lambda fpath: TrainDataEntry(path=fpath, label=expression),
    retina.filesys.get_files_in_folder(folder.path)
  ))

retina.face.get_face_landmark_detector()

skipped = 0
list_data: list[npt.NDArray] = []
list_labels: list[int] = []
for entry in tqdm.tqdm(entries, desc="Building dataset from images"):
  original = cv.imread(entry.path)
  canvas = cv.cvtColor(retina.face.preprocess_face_image(original), cv.COLOR_GRAY2BGR)
  features = retina.face.face2vec(original, canvas=canvas, skip_face_detection=True)
  retina.debug.imdebug(canvas)
  if features is None:
    print(f"\n{Ansi.Error}Skipping {entry.path} because no faces were found in the image.{Ansi.End}")
    skipped += 1
    continue
  for feature in features:
    list_data.append(feature)
    list_labels.append(entry.label)
  
if len(list_data) == 0:
  print(f"{Ansi.Error}No images were successfully processed into the dataset.{Ansi.End}")
  exit(1)

print(f"{Ansi.Warning}Skipped over {skipped} images because no faces were found in the images.")

data = np.array(list_data)
labels = np.array(list_labels).reshape((-1, 1))
dfdata = np.hstack((labels, data))
df = pd.DataFrame(dfdata, columns=[
  "label",
  *map(lambda idx: f'feature-{idx + 1}', range(data.shape[1])),
])

df.to_csv(retina.filesys.TRAINING_DATA_CSV_PATH, index=False)

