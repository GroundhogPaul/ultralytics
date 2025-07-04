import cv2
import utilBasicRunPose
from ultralytics import YOLO
import os
import shutil

def CollectFalseNegative(sModelPath, sImgFolder, sNameDataSet, sFolderOut, conf):
    # ----- load a model -----
    model = YOLO(sModelPath)

    # ----- collect images under a folder -----
    sFolderOut = os.path.join(sImgFolder, sFolderOut)
    os.makedirs(sFolderOut, exist_ok=True)
    lstImg = []
    for filename in os.listdir(sImgFolder):
        full_path = os.path.join(sImgFolder, filename)
        if os.path.isfile(full_path) and filename.lower().endswith('.png'):
            lstImg.append(os.path.normpath(full_path))
    print(lstImg[0])
    print(lstImg[-1])

    # ----- Predict with the model -----
    for sImgPath in lstImg:
        sFileName = sNameDataSet + "_" + os.path.basename(sImgPath)
        results = model(sImgPath, verbose=False, imgsz = 32, device = '0', conf = conf)  # predict on an image
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

sModelPath = "./runs/LapaTrain/01A_theSecond400epoch_rotate_mosaic/weights/best.pt"
sNameDataSet = "ffhq"
conf = 0.51
# for i in range(0, 69001, 1000):
for i in range(0, 02001, 1000):
    formatted_i = f"{i:05d}"
    sImgFolder = "D:/users/xiaoyaopan/PxyAI/DataSet2/ffhq/images1024x1024/" + formatted_i
    sFolderOut = "LapaFailed"
    CollectFalseNegative(sModelPath, sImgFolder, sNameDataSet, sFolderOut, conf = conf)