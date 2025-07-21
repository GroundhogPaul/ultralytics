import cv2
import utilBasicRunPose
from ultralytics import YOLO
import ncnn
import pnnx
import os

# ----- Load a model ----- #
# sModelPath = "./runs/LapaTrain/01A_2nd500epoch_rotate_mosaic_128_smallBatch/weights/best.pt"
sModelPath = "./runs/LapaTrain/176x3/weights/best.pt"
sModelFolder = os.path.dirname(sModelPath)
sModelOutputPath = os.path.join(sModelFolder, "best_ncnn_model")

# ----- Convert and export ----- #
Wnn, Hnn = 176, 144
model = YOLO(sModelPath)
model.export(format="ncnn", device="cpu", imgsz=(Hnn, Wnn))

# ----- load the exported model ----- #
ncnn_model = YOLO(
    sModelOutputPath,
    task = "pose",
    )

# ----- test the exported model ----- #
lstRes = ncnn_model("BabySmile_128x96.jpg", 
                    verbose=True, 
                    nms=True, conf=0.25, iou=0.7,
                    imgsz=(Hnn, Wnn), device="cpu", half=False,
                    single_cls=True
                    ) 
Res = lstRes[0]
img = Res.plot(line_width=1, kpt_radius=1)
cv2.imshow("", img)
cv2.waitKey(0)