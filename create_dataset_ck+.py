import retina
import pandas as pd
import numpy as np
import numpy.typing as npt
import zipfile
import tqdm

# Sourced from https://www.kaggle.com/datasets/davilsena/ckdataset/data
with zipfile.ZipFile("ck+.zip") as zipf:
  csv = zipf.open("ckextended.csv")
  df = pd.read_csv(csv)

labels = df["emotion"]
data = np.array(list(map(
  lambda x: np.array(list(map(
    int,
    x.split(' ')
  ))).astype(np.uint8).reshape((48, 48)),
  df["pixels"]
)))

retina.face.get_face_landmark_detector()
list_data: list[npt.NDArray] = []
for img in tqdm.tqdm(data, desc="Creating dataset from CK+ dataset"):
  ftbatch = retina.face.face2vec(img, skip_face_detection=True, skip_face_landmarking=True)
  if ftbatch is None:
    continue
  list_data.append(ftbatch[0])
data = np.array(list_data)
dfdata = np.hstack((labels.to_numpy().reshape((-1, 1)), data))
df = pd.DataFrame(dfdata, columns=[
  "label",
  *map(lambda idx: f'feature-{idx + 1}', range(data.shape[1])),
])

df.to_csv(retina.filesys.TRAINING_DATA_CSV_PATH, index=False)

