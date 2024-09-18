import os
import sys
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

  retina.cvutil.finish_process(original, before=original, path=file_path, force_save=not retina.cvutil.IS_PREVIEW)

  
