import os
import time
import cv2 as cv
import numpy as np

import retina
from retina.log import Ansi

class_names = sorted(retina.face.FacialExpressionLabel.__members__.values(), key=lambda x: x.value)

camera = cv.VideoCapture(0)
if not os.path.exists(retina.filesys.DATA_DIR_PATH):
  print(f"{Ansi.Warning}Creating {retina.filesys.DATA_DIR_PATH} as it doesn't exist yet.{Ansi.End}")
  os.mkdir(retina.filesys.DATA_DIR_PATH)

for idx, cls in enumerate(class_names):
  print(f"{idx + 1}. {cls.name}")
  dirpath = os.path.join(retina.filesys.DATA_DIR_PATH, retina.face.INVERSE_FACIAL_EXPRESSION_MAPPER[cls])
  if not os.path.exists(dirpath):
    print(f"{Ansi.Warning}Creating {dirpath} as it doesn't exist yet.{Ansi.End}")
    os.mkdir(dirpath)
while True:
  ret, frame = camera.read()
  if not ret:
    break

  canvas = retina.cvutil.resize_image(frame, retina.size.PREVIEW_DIMENSIONS)
  cv.imshow("Camera Feed", canvas)

  key = cv.waitKey(5)

  if key == 27:
    break

  label = key - ord('1')
  if 0 <= label < len(class_names):
    cls = class_names[label]
    pathname = os.path.join(retina.filesys.DATA_DIR_PATH, retina.face.INVERSE_FACIAL_EXPRESSION_MAPPER[cls], f"Frame {time.time()}.png")
    cv.imwrite(pathname, canvas)
    print(f"{Ansi.Success}Captured {pathname} with label {cls.name}{Ansi.End}")
