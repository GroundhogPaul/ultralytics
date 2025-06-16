import cv2
import utilBasicRunPose
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler

# Load a model
sModelPath = "./runs/LapaTrain/train_20250613_128x128/weights/best.pt"
model = YOLO(sModelPath)

# Predict with the model
results = model("BabySmile.jpg", verbose=False)  # predict on an image
img = results[0].plot()
cv2.imshow("", img)
cv2.waitKey(0)

GFlops = get_flops_with_torch_profiler(model, imgsz=[128, 128])
print(GFlops)