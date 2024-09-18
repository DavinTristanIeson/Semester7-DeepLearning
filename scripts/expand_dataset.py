import os
import sys

import tqdm
sys.path.append(os.getcwd())
import retina
import cv2 as cv

if "--pipe" in sys.argv:
  import scripts.preprocess

FOLDERNAME = retina.filesys.PLAYGROUND_PATH
files = retina.filesys.get_files_in_folder(FOLDERNAME)
for idx, file_path in enumerate(tqdm.tqdm(files, desc="Expanding Dataset")):
  original = cv.imread(file_path)

  filename = os.path.basename(file_path)
  def namefile(prefix: str):
    path = os.path.join(FOLDERNAME, f"{prefix}{retina.filesys.FILE_TRANSFORM_SEPARATOR}{filename}")
    return path

  
  dims = retina.size.Dimension.from_shape(original.shape)

  # Rotation
  ANGLE_SLICES_COUNT = 10
  for i in range(1, ANGLE_SLICES_COUNT):
    rotation = (i * 360 / ANGLE_SLICES_COUNT)
    rotated = retina.size.rotate_image(original, rotation)
    retina.cvutil.finish_process(rotated, before=original, path=namefile(f"Rotated{i}"))
    # Flip. Vertical is 1, Horizontal is 0
    for j in range(0, 2):
      flipped = cv.flip(rotated, j)
      retina.cvutil.finish_process(flipped, before=original, path=namefile(f"Rotated{i}-Flip{j}"))
      

