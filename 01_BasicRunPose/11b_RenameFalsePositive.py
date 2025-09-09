'''
input: a folder of false positive images
output: images with Suffix and label.txt (empty, since false positive) with same name
'''

import cv2
import utilBasicRunPose
from ultralytics import YOLO
import os
import shutil

def RenameFalsePositive(sImgFolder, sFolderOut, sSuffix, sExt = 'jpg'):
    '''
    Collect images without face but detected by the model (false positive).
    input:
        sImgFolder: folder containing images to be processed
        sFolderOut: name of output folder
        sSuffix: suffix to be added to the image file name
        sExt: file extension of images to be processed
    '''
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

    for i, sImgPath in enumerate(lstImg):
        # sImgOutPath = sSuffix + os.path.splitext(os.path.basename(sImgPath))[0] + sSuffix + "." + sExt
        sImgOutPath = os.path.splitext(os.path.basename(sImgPath))[0] + "." + sExt
        sImgOutPath = os.path.join(sFolderOut, sImgOutPath)
        # shutil.copy2(sImgPath, sImgOutPath)
        sLabelOutPath = os.path.splitext(sImgOutPath)[0] + ".txt"
        print(i, sLabelOutPath)
        with open(sLabelOutPath, 'w') as f:
            f.write("") 

# sImgFolder = "../../DataSet/Lapa_Plus-yolo11/FalsePositive250805/images/train"
# sImgFolder = "../../DataSet2/coco-NoFace/images/LapaFP_Ancient/Coco2017conf0.5"
sImgFolder = "../../DataSet2/coco-NoFace/images/FP20250819"
sFolderOut = "FP250822"
sSuffix = ""
RenameFalsePositive(sImgFolder, sFolderOut, sSuffix, sExt = 'jpg')