from dataclasses import dataclass
import os
import numpy as np
import numpy.typing as npt

DATASET_PATH = "dataset.zip"
DATA_DIR_PATH = "data"
DATA_CSV_PATH = os.path.join(DATA_DIR_PATH, "data.csv")

def get_files_in_folder(basepath: str):
  if not os.path.exists(basepath):
    return []
  file_paths = [os.path.join(basepath, filename) for filename in os.listdir(basepath)]
  return [path for path in file_paths if os.path.isfile(path)]