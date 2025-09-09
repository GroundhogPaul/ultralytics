'''
collect false negative images from a folder of images
and save them to a specified output folder.
'''

import cv2
import utilBasicRunPose
from ultralytics import YOLO
import os
import shutil

def CollectFalsePositive(sImgFolder, sFolderOut = "BMP", sExt = "jpg"):
    assert os.path.exists(sImgFolder), "Image folder does not exist: " + sImgFolder 

    # ----- collect images under a folder -----
    sFolderOut = os.path.join(sImgFolder, sFolderOut)
    os.makedirs(sFolderOut, exist_ok=True)
    lstImg = []
    for filename in os.listdir(sImgFolder):
        full_path = os.path.join(sImgFolder, filename)
        if os.path.isfile(full_path) and filename.lower().endswith(sExt):
            lstImg.append(os.path.normpath(full_path))
    print(lstImg[0], lstImg[-1])

    # ----- Predict with the model -----
    for sImgPath in lstImg:
        print(sImgPath)
        imgOut = cv2.imread(sImgPath)
        sFileName = os.path.basename(sImgPath)
        sImgPathSave = os.path.join(sFolderOut, sFileName)
        sImgPathSave = sImgPathSave.replace("jpg", "bmp")
        cv2.imwrite(sImgPathSave, imgOut)

sModelPath, imgSize = "./runs/LapaTrain/176xPruneStep2_/weights/best.pt", 176
conf, sNameDataSet, sFolderOut = 0.5, "OldFalseNegative", "FNstillFailed"

# ----- Use this script to evalulate the input model for certain img set ----- #
sImgFolder = r"D:/image_database/mirflickr25k_BMP/mirflickr"
CollectFalsePositive(sImgFolder)
