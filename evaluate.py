import cv2 as cv
import keras
import numpy as np

import retina

expreg_model: keras.Sequential = keras.models.load_model(retina.filesys.EXPRESSION_RECOGNITION_MODEL_PATH) # type: ignore
class_names = sorted(retina.face.FacialExpressionLabel.__members__.values(), key=lambda x: x.value)
# preload all dependencies
retina.face.get_landmark_pca_model()
retina.face.get_face_haar_classifier()
retina.face.get_face_landmark_detector()

camera = cv.VideoCapture(0)
while cv.waitKey(5) != 27:
  ret, frame = camera.read()
  if not ret:
    break

  canvas = retina.cvutil.resize_image(frame, retina.size.STANDARD_DIMENSIONS)
  features = retina.face.face2vec(frame, canvas=canvas)

  canvas = retina.cvutil.resize_image(canvas, retina.size.PREVIEW_DIMENSIONS)
  cv.imshow("Camera Feed", canvas)
  if features is None:
    continue
  prediction = expreg_model.predict(features, verbose=0) # type: ignore
  labels = np.argmax(prediction, axis=1)

  for label in labels:
    print(class_names[label].name, end=' ')
  print(prediction)
