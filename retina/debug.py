from dataclasses import dataclass
import sys
from typing import Callable, Iterable, Sequence, Union, Optional
import numpy as np
import numpy.typing as npt
import cv2 as cv

import retina
from retina.size import PREVIEW_DIMENSIONS, Dimension, Point, Rectangle, STANDARD_DIMENSIONS
from retina.log import Ansi
import matplotlib.pyplot as plt

IS_QUIET = "--quiet" in sys.argv


@dataclass
class MatshowParams:
  value: Union[int, float]
  pos: Point
  image: npt.NDArray

def matshow(matrix: npt.NDArray, *, area: Optional[Rectangle] = None, condition: Optional[Callable[[MatshowParams], Union[bool, str]]] = None, title: Optional[str] = None):
  if title is not None:
    print(f"{Ansi.Bold}{title}{Ansi.End}")
  for row_idx, row in enumerate(matrix):
    str_row = []
    for col_idx, col in enumerate(row):
      str_cell = str(col).rjust(3)
      modifier = Ansi.Grey
      if area is not None and area.has_point(Point.cell(row_idx, col_idx), cell=True):
        modifier = Ansi.Bold
      if condition is not None:
        result = condition(MatshowParams(col, Point(col_idx, row_idx), matrix))
        if isinstance(result, str):
          modifier = result
        elif result:
          modifier = Ansi.Grey
      str_row.append(f"{modifier}{str_cell}{Ansi.End}")
    print(' '.join(str_row))

def draw_rectangles(img: cv.typing.MatLike, rectangles: Iterable[Rectangle], offset: Optional[Point] = None):
  if img.ndim == 2:
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
  for rect in rectangles:
    if offset is not None:
      rect = rect.translate(offset.x, offset.y)
    cv.rectangle(img, rect.p0.tuple, rect.p1.tuple, color=(0, 255, 0))
  return img

def imdebug(img: cv.typing.MatLike):
  if IS_QUIET:
    return
  dimensions = Dimension.from_shape(img.shape).resize(height=PREVIEW_DIMENSIONS.height)
  cv.imshow("Debug", cv.resize(img, dimensions.tuple, interpolation=cv.INTER_CUBIC))
  retina.cvutil.wait_until_esc()

def collage_images(files: Sequence[Union[str, cv.typing.MatLike]], arrangement: Dimension, size: Dimension = Dimension.sized(64)):
  for i in range(0, len(files), arrangement.area):
    displayed = [cv.resize(cv.imread(path) if isinstance(path, str) else path, size.tuple) for path in files[i:i+arrangement.area]]
    for i in range(len(displayed)):
      img = displayed[i]
      if img.ndim == 2:
        displayed[i] = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    while len(displayed) < arrangement.area:
      displayed.append(np.zeros((*size.tuple, 3), dtype=np.uint8))

    rows = []
    for j in range(0, len(displayed), arrangement.width):
      row_proto = displayed[j:j+arrangement.width]
      row = np.concatenate(row_proto, axis=1)
      rows.append(row)
    image = np.concatenate(rows, axis=0)
    yield image

def collage_images_plt(imgs: Sequence[cv.typing.MatLike], labels: Sequence[str], shape: tuple[int, int]):
  fig, axs = plt.subplots(shape[0], shape[1], figsize=(shape[0] * 2, shape[1] * 2))
  fig.tight_layout()
  axs = axs.flatten()
  for img, ax, label in zip(imgs, axs, labels):
    ax.imshow(img)
    ax.set_xlabel(label)
  plt.show()