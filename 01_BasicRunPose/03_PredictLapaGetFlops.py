'''
1. test a trained.pt with a batch of images
2. get GFlops
'''

import os
import cv2
import utilBasicRunPose
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler

# ----- load a model -----
sModelPath, imgSize = "./runs/LapaTrain/176xPruneStep2_1500epoch_L1shrinkBN/weights/best.pt", 176
model = YOLO(sModelPath, task="pose")

# ----- Predict with the model -----
sDataSetRootFolder = "../../../PxyAI/DataSet"
assert os.path.exists(sDataSetRootFolder)
sImgPosFolder = os.path.join(sDataSetRootFolder, "Lapa/valPxy")
assert os.path.exists(sImgPosFolder)
sImgNegFolder = os.path.join(sDataSetRootFolder, "Lapa_Plus-yolo11/FalsePositive250721/images/train")
assert os.path.exists(sImgNegFolder)
sImgFolder = sImgPosFolder
results = model(sImgFolder + '/*.jpg', imgsz=imgSize, verbose=False, conf=0.5)  # predict on an image

# ----- Plot using Official Class Results(SimpleClass) -----
sFolderOut = os.path.join(sImgFolder, "plot_test")
os.makedirs(sFolderOut, exist_ok=True)
for res in results:
    img = res.plot(line_width=1, kpt_radius=1)
    sImgOut = os.path.join(sFolderOut, os.path.basename(res.path))
    cv2.imwrite(sImgOut, img)
    # cv2.imshow("", img)
    # cv2.waitKey(0)

# ----- Get GFlops ----
GFlops = get_flops_with_torch_profiler(model, imgsz=[128, 96])
print(GFlops)