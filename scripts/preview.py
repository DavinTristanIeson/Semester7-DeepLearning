import os
import sys
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np

import retina
from retina.size import *

files = retina.filesys.get_files_in_folder(retina.filesys.PLAYGROUND_PATH) +\
  retina.filesys.get_files_in_folder(retina.filesys.TRAINING_PATH) +\
  retina.filesys.get_files_in_folder(retina.filesys.TESTING_PATH)

for image in retina.debug.collage_images(files, Dimension(6, 5)):
  retina.cvutil.finish_process(image, force_preview=True)