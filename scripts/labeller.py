import os
import sys
sys.path.append(os.getcwd())

import tqdm
import retina
import cv2 as cv

TEMP_RESULT_PATH = os.path.join(retina.filesys.PLAYGROUND_PATH, 'results')

LABELS = [
  "Very Positive",
  "Positive",
  "Neutral",
  "Negative"
]

for label in LABELS:
  folder_path = os.path.join(retina.filesys.DATA_PATH, label)
  if not os.path.exists(folder_path):
    os.mkdir(folder_path)

for file_path in tqdm.tqdm(retina.filesys.get_files_in_folder(TEMP_RESULT_PATH), desc="Labeling"):
  img = cv.imread(file_path)
  cv.imshow("Labeling", retina.cvutil.resize_image(img, retina.size.PREVIEW_DIMENSIONS))
  chosen_label = -1
  while True:
    key = cv.waitKey(5)
    if key == 27:
      break

    label = key - ord('1')
    if label < 0 or label >= len(LABELS):
      continue
    chosen_label = label
    break

  if chosen_label == -1:
    continue
  destination_path = os.path.join(retina.filesys.DATA_PATH, LABELS[chosen_label], os.path.basename(file_path))
  os.rename(file_path, destination_path)
