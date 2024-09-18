from dataclasses import dataclass
import os
import numpy as np
import numpy.typing as npt

DATASET_PATH = "scripts/dataset.zip"
PLAYGROUND_PATH = "temp"
TRAINING_PATH = os.path.join(PLAYGROUND_PATH, "train")
TESTING_PATH = os.path.join(PLAYGROUND_PATH, "test")
FILE_TRANSFORM_SEPARATOR = "__tf__"

def get_files_in_folder(basepath: str):
  if not os.path.exists(basepath):
    return []
  file_paths = [os.path.join(basepath, filename) for filename in os.listdir(basepath)]
  return [path for path in file_paths if os.path.isfile(path)]

@dataclass
class Dataset:
  data: npt.NDArray
  labels: npt.NDArray
  files: list[str]
  classes: set[str]
  def __len__(self):
    return len(self.data)
  
  def shuffle(self):    
    shuffler = np.arange(0, len(self.data))
    np.random.shuffle(shuffler)    
    for i, idx in enumerate(shuffler):
      self.data[i], self.data[idx] = self.data[idx], self.data[i]
      self.labels[i], self.labels[idx] = self.labels[idx], self.labels[i]
      self.files[i], self.files[idx] = self.files[idx], self.files[i]
