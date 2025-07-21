import cv2
import utilBasicRunPose
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler

# ----- load a model -----
sModelPath = "./runs/LapaTrain/train_20250613_128x128/weights/best.pt"
model = YOLO(sModelPath, task="pose")
imgSize = 176

# ----- Predict with the model -----
# results = model("BabySmile.jpg", imgsz=imgSize, verbose=False)  # predict on an image
results = model('D:/PxyAI/DataSet/Lapa/valPxy/*jpg', imgsz=imgSize, verbose=False)  # predict on an image

# ----- Plot using Official Class Results(SimpleClass) -----
for res in results:
    img = res.plot(line_width=1, kpt_radius=1)
    cv2.imwrite(res.path.replace('.jpg', '_plot.jpg'), img)
    # cv2.imshow("", img)
    # cv2.waitKey(0)

# ----- Get GFlops ----
GFlops = get_flops_with_torch_profiler(model, imgsz=[128, 96])
print(GFlops)