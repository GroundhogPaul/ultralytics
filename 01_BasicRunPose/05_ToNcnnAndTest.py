import cv2
import utilBasicRunPose
from ultralytics import YOLO
import ncnn
import pnnx
import os

# ----- Load a model ----- #
sModelPath = "./runs/LapaTrain/train_20250613_128x128/weights/best.pt"
sModelFolder = os.path.dirname(sModelPath)
sModelOutputPath = os.path.join(sModelFolder, "best_ncnn_model")

# ----- Convert and export ----- #
model = YOLO(sModelPath)
model.export(format="ncnn", device="cpu")

# ----- load the exported model ----- #
ncnn_model = YOLO(
    # "runs/LapaTrain/train_20250613_128x128/weights/best_ncnn_model", 
    sModelOutputPath,
    task="pose",
    )

# ----- test the exported model ----- #
lstRes = ncnn_model("BabySmile.jpg", 
                    verbose=True, 
                    nms=True, conf=0.25, iou=0.7,
                    imgsz=128, device="cpu", half=False,
                    ) 
Res = lstRes[0]
img = Res.plot()
cv2.imshow("", img)
cv2.waitKey(0)