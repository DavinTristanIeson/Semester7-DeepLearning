from dataclasses import dataclass
import os
import numpy as np
import numpy.typing as npt

RAW_FACES_DATASET_PATH = "raw_faces.zip"
DATA_PATH = "data"
LABELS_PATH = "labels.json"
PLAYGROUND_PATH = "temp"

def get_files_in_folder(basepath: str):
  if not os.path.exists(basepath):
    return []
  file_paths = [os.path.join(basepath, filename) for filename in os.listdir(basepath)]
  return [path for path in file_paths if os.path.isfile(path)]