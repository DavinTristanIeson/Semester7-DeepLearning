import os
import sys
sys.path.append(os.getcwd())
import retina
import sklearn.model_selection

if "--pipe" in sys.argv:
  import scripts.expand_dataset

import shutil

files = retina.filesys.get_files_in_folder(retina.filesys.PLAYGROUND_PATH)
datasets = sklearn.model_selection.train_test_split(files, test_size=0.4, random_state=2024, shuffle=True)
FOLDERS = (retina.filesys.TRAINING_PATH, retina.filesys.TESTING_PATH)
for folder in FOLDERS:
  if not os.path.exists(folder):
    os.mkdir(folder)

for i, basepath in enumerate(FOLDERS):
  retina.log.show_progress("Moving Dataset", FOLDERS[i], i, len(FOLDERS))
  for path in datasets[i]:
    filename = os.path.basename(path)
    newpath = os.path.join(basepath, filename)
    shutil.move(path, newpath)