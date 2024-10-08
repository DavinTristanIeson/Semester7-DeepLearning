import os
import sys

import tqdm
sys.path.append(os.getcwd())

import argparse
import retina
import cv2 as cv
import zipfile

from retina.size import Dimension

parser = argparse.ArgumentParser(
  prog="Pack dataset",
  description="This script is used to compress images and gather them inside a zipped file",
)
parser.add_argument("src", help="Specify the source folder that contains all of the images. The folder should only contain images.", required=True)
parser.add_argument("dest", help="Specify the name of the zip file. It should include the extension.", required=False)
args = parser.parse_args()

if not os.path.exists(args.src):
  raise Exception("Dataset directory doesn't exist!")

with zipfile.ZipFile(args.dest or "faces.zip", 'w') as zipf:
  progress = tqdm.tqdm(os.scandir(args.src))
  for file in progress:
    img = cv.imread(file.path)
    img_dims = Dimension.from_shape(img.shape)
    ratio = 1_000_000 / img_dims.area

    if ratio >= 1:
      continue

    img_dims = img_dims.scale(ratio)
    progress.set_description(file.path + ' ' + str(img_dims))

    img = retina.cvutil.resize_image(img, img_dims)

    with zipf.open(os.path.basename(file.path), 'w') as f:
      file_contents = cv.imencode('.png', img)[1].tobytes()
      f.write(file_contents)
