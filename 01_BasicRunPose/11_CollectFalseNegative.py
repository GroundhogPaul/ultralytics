import cv2
import utilBasicRunPose
from ultralytics import YOLO
import os
import shutil

# ----- load a model -----
sModelPath = "./runs/LapaTrain/01A_theSecond400epoch_rotate_mosaic/weights/best.pt"
model = YOLO(sModelPath)

# ----- collect images under a folder -----
sNameDataSet = "ffhq"
sImgsFolder = "D:/users/xiaoyaopan/PxyAI/DataSet2/ffhq/images1024x1024/16000"
sFolderOut = os.path.join(sImgsFolder, "LapaFailed")
os.makedirs(sFolderOut, exist_ok=True)
lstImg = []
for filename in os.listdir(sImgsFolder):
    full_path = os.path.join(sImgsFolder, filename)
    if os.path.isfile(full_path) and filename.lower().endswith('.png'):
        lstImg.append(os.path.normpath(full_path))
print(lstImg)

# ----- Predict with the model -----
for sImgPath in lstImg:
    sFileName = sNameDataSet + "_" + os.path.basename(sImgPath)
    results = model(sImgPath, verbose=False, imgsz = 32, device = 'cpu')  # predict on an image
    res = results[0]
    if 1 == len(res):
        # print("1 face found: pass img ", sFileName)
        continue
    sImgPathSave = os.path.join(sFolderOut, sFileName)
    if 0 == len(res):
        print("0 face found: save img ", sFileName)
        shutil.copy2(sImgPath, sImgPathSave)
    # if 1 < len(res):
    #     print("len(res) face found: save img ", sFileName)