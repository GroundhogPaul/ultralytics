import cv2
import utilBasicRunPose
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler

# ----- load a model -----
sModelPath = "./runs/LapaTrain/train_20250613_128x128/weights/best.pt"
model = YOLO(sModelPath)

# ----- Predict with the model -----
results = model("BabySmile.jpg", verbose=False)  # predict on an image

# ----- Plot using Official Class Results(SimpleClass) -----
img = results[0].plot(line_width=1, kpt_radius=1)
cv2.imshow("", img)
cv2.waitKey(0)

# ----- Get GFlops -----
GFlops = get_flops_with_torch_profiler(model, imgsz=[128, 128])
print(GFlops)