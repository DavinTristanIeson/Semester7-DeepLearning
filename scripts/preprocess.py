import os
import sys

import retina.debug
from retina.log import Ansi
from retina.size import FACE_DIMENSIONS
sys.path.append(os.getcwd())
import retina
import cv2 as cv
import tqdm

if "--pipe" in sys.argv:
  import scripts.unzip

files = retina.filesys.get_files_in_folder(retina.filesys.PLAYGROUND_PATH)
for idx, file_path in enumerate(tqdm.tqdm(files, desc="Transforming Images")):
  original = cv.imread(file_path)
  
  # Perform transformation here
  img = retina.size.resize_image(original, retina.size.STANDARD_DIMENSIONS) # Resize
  brightness = retina.colors.get_autobrightness_values(img)
  img = retina.colors.contrast(img, 1, brightness)
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Grayscale

  # img, _ = retina.convolution.sobel(img)
  face_positions = retina.face.haar_detect(img)
  faces = tuple(
    retina.size.resize_image(img[pos.slice], FACE_DIMENSIONS)
    for pos in face_positions
  )

  img = retina.debug.draw_rectangles(img, face_positions)
  retina.cvutil.finish_process(img, before=original)

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
      retina.cvutil.finish_process(face, before=img, path=file_path)
  
  if len(faces) == 0:
    print(f"{Ansi.Error}Removed {file_path} because it doesn't contain any faces!{Ansi.End}")
    os.remove(file_path)

  
