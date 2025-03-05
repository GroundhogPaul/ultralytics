# predict for the first time using source code
# see documentation B2.1.1

import cv2
import utilBasicRunPose
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

img = results[0].plot()
cv2.imshow("", img)
cv2.waitKey(0)