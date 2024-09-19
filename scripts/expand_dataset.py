import os
import sys

import tqdm
sys.path.append(os.getcwd())
import retina
import cv2 as cv

if "--pipe" in sys.argv:
  import scripts.preprocess

def namefile(prefix: str, file_path: str):
  filename = os.path.basename(file_path)
  path = os.path.join(FOLDERNAME, f"{prefix}{retina.filesys.FILE_TRANSFORM_SEPARATOR}{filename}")
  return path


FOLDERNAME = retina.filesys.PLAYGROUND_PATH
files = retina.filesys.get_files_in_folder(FOLDERNAME)
for idx, file_path in enumerate(tqdm.tqdm(files, desc="Expanding Dataset")):
  original = cv.imread(file_path)

  flipped = cv.flip(original, 1)
  retina.cvutil.finish_process(flipped, before=original, path=namefile(f"FlipHorizontal", file_path))
    
      

