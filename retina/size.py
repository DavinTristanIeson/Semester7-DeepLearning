from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import numpy.typing as npt
import numpy as np
import retina.math
import math
import cv2 as cv


@dataclass
class FloatingPoint:
  x:float
  y:float
  @property
  def integer(self)->"Point":
    return Point(int(self.x), int(self.y))
  @property
  def tuple(self)->tuple[float, float]:
    return (self.x, self.y)
  @staticmethod
  def from_tuple(src: Tuple[int, int])->"Point":
    return Point(src[0], src[1])
  

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
  def nparray(self)->npt.NDArray:
    return np.array(self.tuple)
  @property
  def as_cell(self)->Tuple[int, int]:
    return (self.row, self.col)
  @staticmethod
  def cell(row: int, col:int):
    return Point(col, row)
  def forward(self, angle: float, shift: float)->"Point":
    # https://stackoverflow.com/questions/22252438/draw-a-line-using-an-angle-and-a-point-in-opencv
    angle = retina.math.deg2rad(angle)
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
  def normalized(self, dims: "Dimension")->"FloatingPoint":
    return FloatingPoint(self.x / dims.width, self.y / dims.height)
  
  @staticmethod
  def from_tuple(src: Union[Tuple[int, int], npt.NDArray]):
    return Point(src[0], src[1])
  

@dataclass
class Dimension:
  width: int
  height: int
  @property
  def tuple(self)->tuple[int, int]:
    return (self.width, self.height)
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
  def has_point(self, point: Point, *, cell = False)->bool:
    offset = -1 if cell else 0
    return 0 <= point.x <= self.width + offset and 0 <= point.y <= self.height + offset
  def at_edge(self, point: Point, *, cell = False)->bool:
    offset = -1 if cell else 0
    return point.x == 0 or point.x == self.width + offset or point.y == 0 or point.y == self.height + offset
  def scale(self, scale: float)->"Dimension":
    return Dimension(int(self.width * scale), int(self.height * scale))
  def sample(self, x: float, y: float)->Point:
    return Point(int(self.width * x), int(self.height * y))
  
  def partition(self, rows: int, cols: int)->list["Rectangle"]:
    row_delta = int(self.width / rows)
    col_delta = int(self.height / cols)
    rects: list["Rectangle"] = []
    for row in range(rows):
      for col in range(cols):
        row_start = row * row_delta
        col_start = col * col_delta
        rect = Rectangle(col_start, row_start, col_start + col_delta, row_start + row_delta)
        rects.append(rect)
    return rects
      
  
  @staticmethod
  def blerp(image: npt.NDArray, dimensions: "Dimension")->npt.NDArray:
    # https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
    height, width = dimensions.tuple
    img_height, img_width = image.shape[:2]

    resized = np.empty([height, width])

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    for i in range(height):
      for j in range(width):

        x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
        x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

        x_weight = (x_ratio * j) - x_l
        y_weight = (y_ratio * i) - y_l

        a = image[y_l, x_l]
        b = image[y_l, x_h]
        c = image[y_h, x_l]
        d = image[y_h, x_h]

        pixel = a * (1 - x_weight) * (1 - y_weight)\
                + b * x_weight * (1 - y_weight) + \
                c * y_weight * (1 - x_weight) + \
                d * x_weight * y_weight

        resized[i][j] = pixel
    return resized


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
  
  def scan_cell_indices(self):
    r = self.y0
    while r < self.y1:
      c = self.x0
      while c < self.x1:
        yield r, c
        c += 1
      r += 1
  def scan_cell_1d_indices(self):
    return (r * self.y1 + c for r, c in self.scan_cell_indices())

  def has_point(self, point: Point, *, cell = False)->bool:
    offset = -1 if cell else 0
    return self.x0 <= point.x <= self.x1 + offset and self.y0 <= point.y <= self.y1 + offset
  def at_edge(self, point: Point, *, cell = False)->bool:
    offset = -1 if cell else 0
    return point.x == self.x0 or point.x == self.x1 + offset or point.y == self.y0 or point.y == self.y1 + offset

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
      retina.math.clamp(self.x0, min_x, max_x),
      retina.math.clamp(self.y0, min_y, max_y),
      retina.math.clamp(self.x1, min_x, max_x),
      retina.math.clamp(self.y1, min_y, max_y),
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

STANDARD_DIMENSIONS = Dimension(240, 240)
FACE_DIMENSIONS = Dimension(120, 120)
PREVIEW_DIMENSIONS = Dimension(500, 500)

