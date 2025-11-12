import cv2
import utilBasicRunDet
from ultralytics import YOLO
import pnnx
import ncnn
import os

# ----- Load a model ----- #
sModelPath = "./runs/Det/yolo11nOfficial/yolo11n.pt"
sModelFolder = os.path.dirname(sModelPath)
sModelOutputPath = os.path.join(sModelFolder, "yolo11n_ncnn_model")

# ----- Convert and export ----- #
Wnn, Hnn = 160, 160
model = YOLO(sModelPath)
model.export(format="ncnn", device="cpu", imgsz=(Hnn, Wnn))

# ----- load the exported model ----- #
ncnn_model = YOLO(
    sModelOutputPath,
    task = "detect",
    )

# ----- test the exported model ----- #
lstRes = ncnn_model("BabySmile_192x160.jpg", 
                    verbose=True, 
                    nms=True, conf=0.5, iou=0.7,
                    imgsz=(Hnn, Wnn), device="cpu", half=False,
                    ) 
Res = lstRes[0]
img = Res.plot(line_width=1, kpt_radius=1)
cv2.imshow("", img)
cv2.waitKey(0)