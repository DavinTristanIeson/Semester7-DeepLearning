import os
import sys
from typing import Sequence

sys.path.append(os.getcwd())

import shutil
import zipfile

import tqdm
import cv2 as cv

import retina
from retina.size import Point

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
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Grayscale
  img = retina.colors.clahe(img) # Contrast adjustment

  # img, _ = retina.convolution.sobel(img)
  face_positions = retina.face.haar_detect(img)
  faces = tuple(
    img[pos.slice]
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

  for idx, face in enumerate(saved_faces):
    retina.debug.imdebug(retina.cvutil.resize_image(face, retina.size.FACE_DIMENSIONS))
    # face = retina.cvutil.resize_image(face, retina.size.Dimension(256, 256))
    # face = retina.convolution.gabor(face)
    face = retina.cvutil.resize_image(face, retina.size.FACE_DIMENSIONS)
    face = cv.filter2D(face, -1, retina.convolution.GAUSSIAN_3X3_KERNEL)
    face = cv.filter2D(face, -1, retina.convolution.SHARPEN_KERNEL)
    face_landmarks = retina.face.face_landmark_detection(face)

    retina.debug.imdebug(retina.face.draw_face_landmarks(face, face_landmarks))
    saved_faces[idx] = face

  filename = os.path.basename(file_path)
  filename, fileext = os.path.splitext(filename)
  for idx, face in enumerate(saved_faces):    
    result_filename = filename + (str(idx) if idx > 0 else '') + fileext
    result_path = os.path.join(TEMP_RESULT_PATH, result_filename)
    cv.imwrite(result_path, face)


