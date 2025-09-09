import cv2
import utilBasicRunSeg
from ultralytics import YOLO
import random
import numpy as np

# Load a model
model = YOLO("yolo11n-seg.pt", task='seg')  # load an official model

# Predict with the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image need load
results = model("bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)

img = result.plot()
cv2.imshow("", img)
cv2.waitKey(0)