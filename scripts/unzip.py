import zipfile
import os
import sys
sys.path.append(os.getcwd())
import shutil
import retina

if (os.path.exists(retina.filesys.PLAYGROUND_PATH)):
  shutil.rmtree(retina.filesys.PLAYGROUND_PATH)
with zipfile.ZipFile(retina.filesys.DATASET_PATH, 'r') as zip_ref:
  zip_ref.extractall(retina.filesys.PLAYGROUND_PATH)
