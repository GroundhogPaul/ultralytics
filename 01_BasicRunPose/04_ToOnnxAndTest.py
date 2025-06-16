import cv2
import onnx
import onnxruntime
import utilBasicRunPose
from ultralytics import YOLO

# Load a model
sModelPath = "./runs/LapaTrain/train_20250613_128x128/weights/best.pt"
model = YOLO(sModelPath)

model.export(format="onnx", device="cpu") # TODO not stable, sometimes crash

sModelPathOnnx = sModelPath.replace("pt", "onnx")
onnx_model = YOLO(sModelPathOnnx)
results = onnx_model("BabySmile.jpg", verbose=False, task='detect', device = "cpu", imgsz=128)  # predict on an image
# TODO the result cannot plot yet

# img = results[0].plot()
# cv2.imshow("", img)
# cv2.waitKey(0)