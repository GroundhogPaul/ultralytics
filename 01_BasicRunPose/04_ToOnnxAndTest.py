import cv2
import onnx
import onnxruntime
import onnxslim
import utilBasicRunPose
from ultralytics import YOLO
import os

# ----- Load a model ----- #
sModelPath = "./runs/LapaTrain/train_20250613_128x128/weights/best.pt"
sModelPathOnnx = sModelPath.replace("pt", "onnx")
model = YOLO(sModelPath, task='pose', verbose=False) 

# ----- Export to ONNX ----- #
model.export(format="onnx", device="cpu", imgsz=(128))

# ----- Load ONNX model and run inference ----- #
onnx_model = YOLO(sModelPathOnnx, task='pose', verbose=False)
results = onnx_model("BabySmile_128x96.jpg", verbose=False, task='detect', device = "cpu", imgsz=128)  # predict on an image

img = results[0].plot(line_width=1, kpt_radius=1)
cv2.imshow("", img)
cv2.waitKey(0)