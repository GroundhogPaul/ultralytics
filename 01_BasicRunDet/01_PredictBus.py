# predict for the first time using source code
# see documentation B2.1.1

import cv2
import utilBasicRunDet
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Predict with the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image need load
results = model("bus.jpg")  # predict on an image

img = results[0].plot()
cv2.imwrite("predictedBus.jpg", img)