from typing import Callable
import cv2 as cv
import sys

import numpy as np
from retina.size import *

def debugshow(img: cv.typing.MatLike, *, window_name:str = "Debug"):
  cv.imshow(window_name, img)
  cv.waitKey(0)

IS_PREVIEW = "--preview" in sys.argv
IS_SAVE = "--save" in sys.argv
PREVIEW_HEIGHT = 600

def wait_until_esc():
  while (cv.waitKey(5) != 27): pass

def finish_process(after: cv.typing.MatLike, before: Optional[cv.typing.MatLike] = None, path: Optional[str] = None, *, force_preview: bool = False, force_save: bool = False):
  if IS_PREVIEW or force_preview:
    after_dimensions = Dimension.from_shape(after.shape).resize(height=PREVIEW_HEIGHT).tuple
    cv.imshow("After", cv.resize(after, after_dimensions, interpolation=cv.INTER_CUBIC))
    if before is not None:
      before_dimensions = Dimension.from_shape(before.shape).resize(height=PREVIEW_HEIGHT).tuple
      cv.imshow("Before", cv.resize(before, before_dimensions, interpolation=cv.INTER_CUBIC))
    wait_until_esc()

  if (IS_SAVE or force_save) and path is not None:
    cv.imwrite(path, after)
    
def splice_matrix(dest: cv.typing.MatLike, src: cv.typing.MatLike, rect: Rectangle):
  dest_rect = rect.clamp(Dimension.from_shape(dest.shape))
  src_rect = Rectangle.with_dimensions(Dimension.from_shape(src.shape))\
    .clamp(dest_rect.dimensions)\
    .translate(dest_rect.x0 - rect.x0, dest_rect.y0 - rect.y0)
  dest[dest_rect.slice] = src[src_rect.slice]

def mask_matrix(dest: cv.typing.MatLike, src: cv.typing.MatLike, rect: Rectangle):
  dest_rect = rect.clamp(Dimension.from_shape(dest.shape))
  src_rect = Rectangle.with_dimensions(Dimension.from_shape(src.shape))\
    .clamp(dest_rect.dimensions)\
    .translate(dest_rect.x0 - rect.x0, dest_rect.y0 - rect.y0)
  src_mask = src[src_rect.slice].astype(bool)
  dest_mask = np.zeros(dest.shape, dtype=bool)
  dest_mask[dest_rect.slice] = src_mask
  return dest_mask

class MaskingCanvasState:
  canvas: cv.typing.MatLike
  mask: cv.typing.MatLike
  _drawing = False
  size = 10
  def __init__(self, canvas: cv.typing.MatLike, mask: cv.typing.MatLike) -> None:
    self.canvas = cv.resize(canvas, (640, 480), interpolation=cv.INTER_CUBIC).astype(np.uint8)
    self.mask = cv.resize(mask, (640, 480), interpolation=cv.INTER_NEAREST).astype(np.uint8)
  def callback(self, event:int, x:int, y:int, flags, param):
    # https://docs.opencv.org/3.4/db/d5b/tutorial_py_mouse_handling.html
    if event == cv.EVENT_LBUTTONDOWN:
      self._drawing = True
    elif event == cv.EVENT_MOUSEMOVE and self._drawing:
      cv.circle(self.mask, (x, y), 10, (0 if flags & cv.EVENT_FLAG_SHIFTKEY else 255,), -1)
    elif event == cv.EVENT_LBUTTONUP:
      self._drawing = False
  def draw(self, window_name: str):
    cv.imshow(window_name, self.canvas)
    cv.setMouseCallback(window_name, self.callback)
    while (cv.waitKey(5) != 27):
      cv.imshow(window_name, cv.addWeighted(self.canvas, 1, self.mask, 1, 0))
    return cv.resize(self.mask, (320, 240), interpolation=cv.INTER_CUBIC)

class SelectionCanvasState:
  canvas: cv.typing.MatLike
  original: cv.typing.MatLike
  _drawing = False
  _x = -1
  _y = -1
  _selection: Optional[Rectangle] = None
  def __init__(self, canvas: cv.typing.MatLike) -> None:
    self.canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR) if len(canvas.shape) == 2 else canvas
    self.original = self.canvas.copy()
  def callback(self, event:int, x:int, y:int, flags, param):
    # https://docs.opencv.org/3.4/db/d5b/tutorial_py_mouse_handling.html
    if event == cv.EVENT_LBUTTONDOWN:
      self._drawing = True
      self._x, self._y = x, y
    elif event == cv.EVENT_MOUSEMOVE and self._drawing:
      self.canvas = self.original.copy()
      cv.rectangle(self.canvas, (self._x, self._y), (x, y), (0, 255, 0), 2)
    elif event == cv.EVENT_LBUTTONUP:
      self._selection = Rectangle(self._x, self._y, x, x)
      self._drawing = False
      self._x = -1
      self._y = -1

  def get_selection(self, window_name: str):
    cv.imshow(window_name, self.canvas)
    cv.setMouseCallback(window_name, self.callback)
    while (cv.waitKey(5) != 27):
      cv.imshow(window_name, self.canvas)
    return self._selection
