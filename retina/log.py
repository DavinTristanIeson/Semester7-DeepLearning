from dataclasses import dataclass
import sys
from typing import Callable, Union, Optional
import numpy.typing as npt

from retina.size import Point, Rectangle

class Ansi:
  Error = '\033[91m'
  Warning = '\033[92m'
  Success = '\033[93m'
  End = '\033[0m'
  Grey = "\033[38;5;243m"
  Bold = "\033[1m"
  Underline = "\033[4m"

def show_progress(action: str, message: str, step: int, total: int):
  print(f"{Ansi.Bold}{action}{Ansi.End}{Ansi.Grey}: {message} ({step + 1}/{total}){Ansi.End}")

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
