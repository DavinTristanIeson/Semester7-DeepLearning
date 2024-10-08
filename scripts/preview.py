import os
import sys
sys.path.append(os.getcwd())

import retina
from retina.size import *

files = []
if os.path.exists(retina.filesys.PLAYGROUND_PATH):
  for dirpath in os.scandir(retina.filesys.PLAYGROUND_PATH):
    if os.path.isdir(dirpath.path):
      files.extend(retina.filesys.get_files_in_folder(dirpath.path))
if os.path.exists(retina.filesys.DATA_PATH):
  for dirpath in os.scandir(retina.filesys.DATA_PATH):
    if os.path.isdir(dirpath.path):
      files.extend(retina.filesys.get_files_in_folder(dirpath.path))

for image in retina.debug.collage_images(files, Dimension(6, 5)):
  retina.debug.imdebug(image)