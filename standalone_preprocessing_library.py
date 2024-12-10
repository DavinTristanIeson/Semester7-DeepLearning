from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import numpy.typing as npt
import numpy as np
import math
import cv2 as cv

def clamp(value: int, min_value: int, max_value: int):
  return max(min_value, min(max_value, value))
def deg2rad(deg: float):
  return deg * 3.14 / 180.0
def rad2deg(rad: float):
  return rad * 180.0 / 3.14

@dataclass
class Point:
  x: int
  y: int
  @property
  def row(self):
    return self.y
  @property
  def col(self):
    return self.x
  @property
  def tuple(self)->tuple[int, int]:
    return (int(self.x), int(self.y))
  @property
  def ndarray(self)->npt.NDArray:
    return np.array(self.tuple)
  @property
  def as_cell(self)->Tuple[int, int]:
    return (self.row, self.col)
  @staticmethod
  def cell(row: int, col:int):
    return Point(col, row)
  def forward(self, angle: float, shift: float)->"Point":
    # https://stackoverflow.com/questions/22252438/draw-a-line-using-an-angle-and-a-point-in-opencv
    angle = deg2rad(angle)
    return Point(
      int(self.x + shift * math.cos(angle)),
      int(self.y + shift * math.sin(angle))
    )
  def translate(self, dx: int, dy: int):
    return Point(self.x + dx, self.y + dy)
  def __eq__(self, value: object) -> bool:
    if isinstance(value, Point):
      return self.x == value.x and self.y == value.y
    return False
  def radians_to(self, point: "Point"):
    return np.arctan2(point.x - self.x, point.y - self.y)
  
  @staticmethod
  def from_tuple(src: Union[Tuple[int, int], npt.NDArray]):
    return Point(src[0], src[1])
  
  def project_to(self, from_rect: "Rectangle", to_rect: "Rectangle")->"Point":
    x_new = int(self.x * to_rect.width / from_rect.width)
    y_new = int(self.y * to_rect.height / from_rect.height)
    return Point(x_new, y_new)

@dataclass
class Dimension:
  width: int
  height: int
  @property
  def tuple(self)->tuple[int, int]:
    return (self.width, self.height)
  @property
  def ndarray(self)->npt.NDArray:
    return np.array((self.width, self.height))
  @property
  def area(self):
    return self.width * self.height
  def resize(self, *, width: Optional[int] = None, height: Optional[int] = None)->"Dimension":
    if width is not None and height is not None:
      ratio = max(width / self.width, height / self.height)
      return Dimension(round(self.width * ratio), round(self.height * ratio))
    elif width is not None:
      ratio = width / self.width
      return Dimension(width, round(self.height * ratio))
    elif height is not None:
      ratio = height / self.height
      return Dimension(round(self.width * ratio), height)
    else:
      return Dimension(self.width, self.height)
  @property
  def center(self)->Point:
    return Point(
      self.width // 2,
      self.height // 2
    )

  @staticmethod
  def sized(value: int)->"Dimension":
    return Dimension(value, value)
  @staticmethod
  def from_shape(shape: Sequence[int])->"Dimension":
    return Dimension(shape[1], shape[0])
  def can_encapsulate(self, rect: Union["Rectangle", "Dimension"])->bool:
    return (self.width >= rect.width and self.height >= rect.height) or (self.width >= rect.height and self.height >= rect.width)
  def scale(self, scale: float)->"Dimension":
    return Dimension(int(self.width * scale), int(self.height * scale))


@dataclass
class Rectangle:
  x0: int
  y0: int
  x1: int
  y1: int
  @property
  def width(self):
    return self.x1 - self.x0
  @property
  def height(self):
    return self.y1 - self.y0
  @property
  def p0(self):
    return Point(self.x0, self.y0)
  @property
  def p1(self):
    return Point(self.x1, self.y1)
  @property
  def center(self)->Point:
    return Point(
      self.x0 + (self.width // 2),
      self.y0 + (self.height // 2)
    )
  @property
  def slice(self)->tuple[slice, slice]:
    return slice(int(self.y0), int(self.y0 + self.height)), slice(int(self.x0), int(self.x0 + self.width))
  @property
  def dimensions(self)->Dimension:
    return Dimension(self.width, self.height)
  @property
  def area(self):
    return self.width * self.height
  @property
  def dict(self):
    return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

  def start_zero(self)->"Rectangle":
    return Rectangle(0, 0, self.width, self.height)
  
  def clamp(self, rect: Union["Rectangle", Dimension])->"Rectangle":
    if isinstance(rect, Rectangle):
      min_x = rect.x0
      max_x = rect.x1
      min_y = rect.y0
      max_y = rect.y1
    else:
      min_x = 0
      max_x = rect.width
      min_y = 0
      max_y = rect.height
    return Rectangle(
      clamp(self.x0, min_x, max_x),
      clamp(self.y0, min_y, max_y),
      clamp(self.x1, min_x, max_x),
      clamp(self.y1, min_y, max_y),
    )
  def translate(self, dx: int, dy: int)->"Rectangle":
    return Rectangle(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
  
  def expand(self, dx: int, dy: int)->"Rectangle":
    return Rectangle(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)
  
  def intersection(self, other: "Rectangle")->"Rectangle":
    # https://machinelearningspace.com/intersection-over-union-iou-a-comprehensive-guide/
    return Rectangle(
      max(self.x0, other.x0),
      max(self.y0, other.y0),
      min(self.x1, other.x1),
      min(self.y1, other.y1),
    )
  def intersection_with_union(self, other: "Rectangle")->"float":
    intersection_area = self.intersection(other).area
    union_area = self.area + other.area - intersection_area
    return intersection_area / union_area

  @staticmethod
  def around(pt: Point, dimension: Dimension)->"Rectangle":
    halfwidth = dimension.width // 2
    halfheight = dimension.height // 2
    return Rectangle(
      pt.x - halfwidth, pt.y - halfheight,
      pt.x + dimension.width - halfwidth,
      pt.y + dimension.height - halfheight
    )
  @staticmethod
  def with_dimensions(dimension: Dimension, starting_point: Optional[Point] = None):
    x = starting_point.x if starting_point is not None else 0
    y = starting_point.y if starting_point is not None else 0
    return Rectangle(
      x, y, x + dimension.width, y + dimension.height
    )
  @staticmethod
  def from_tuple(tuple: Sequence[int])->"Rectangle":
    return Rectangle(tuple[0], tuple[1], tuple[0] + tuple[2], tuple[1] + tuple[3])
  @property
  def tuple(self)->tuple[int,int,int,int]:
    return (self.x0, self.y0, self.width, self.height)

  @staticmethod
  def min_bbox(points: npt.NDArray):
    x0 = x1 = points[0][0]
    y0 = y1 = points[0][1]
    for point in points:
      x0 = min(x0, point[0])
      y0 = min(y0, point[1])
      x1 = max(x1, point[0])
      y1 = max(y1, point[1])
    return Rectangle(x0, y0, x1, y1)

def resize_image(img: cv.typing.MatLike, target_dims: Dimension):
  dimensions = Dimension(img.shape[1], img.shape[0])\
    .resize(width=target_dims.width, height=target_dims.height)
  img = cv.resize(img, dimensions.tuple, interpolation=cv.INTER_LINEAR)
  dimensions = Dimension(img.shape[1], img.shape[0])
  rectangle = Rectangle.around(dimensions.center, target_dims)
  return img[rectangle.slice]

GAUSSIAN_3X3_KERNEL = np.array([
  [1, 2, 1],
  [2, 4, 2],
  [1, 2, 1]
], dtype=float) * (1/16)

SHARPEN_KERNEL = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])