import os
import sys
sys.path.append(os.getcwd())

import shutil
import zipfile

import tqdm
import cv2 as cv

import retina

TEMP_ORIGINAL_PATH = os.path.join(retina.filesys.PLAYGROUND_PATH, 'original')
TEMP_RESULT_PATH = os.path.join(retina.filesys.PLAYGROUND_PATH, 'results')

if (os.path.exists(retina.filesys.PLAYGROUND_PATH)):
  shutil.rmtree(retina.filesys.PLAYGROUND_PATH)
with zipfile.ZipFile(retina.filesys.RAW_FACES_DATASET_PATH, 'r') as zip_ref:
  zip_ref.extractall(TEMP_ORIGINAL_PATH)
os.mkdir(TEMP_RESULT_PATH)

for file_path in tqdm.tqdm(retina.filesys.get_files_in_folder(TEMP_ORIGINAL_PATH), desc="Transforming Images"):
  original = cv.imread(file_path)
  
  # Perform transformation here
  img = retina.cvutil.resize_image(original, retina.size.STANDARD_DIMENSIONS) # Resize
  brightness = retina.colors.get_autobrightness_values(img)
  img = retina.colors.contrast(img, 1, brightness)
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Grayscale

  # img, _ = retina.convolution.sobel(img)
  face_positions = retina.face.haar_detect(img)
  faces = tuple(
    retina.cvutil.resize_image(img[pos.slice], retina.size.FACE_DIMENSIONS)
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

  filename = os.path.basename(file_path)
  filename, fileext = os.path.splitext(filename)
  for idx, face in enumerate(saved_faces):
    result_filename = filename + (str(idx) if idx > 0 else '') + fileext
    result_path = os.path.join(TEMP_RESULT_PATH, result_filename)
    cv.imwrite(result_path, face)



