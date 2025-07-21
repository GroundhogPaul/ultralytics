'''
collect false negative images from a folder of images
and save them to a specified output folder.
'''

import cv2
import utilBasicRunPose
from ultralytics import YOLO
import os
import shutil

def CollectFalseNegative(sModelPath, sImgFolder, sNameDataSet, sFolderOut, conf, 
                         sExt = 'jpg'):
    '''
    Collect images with face but cannot be detected by the model (false negative).
    input:
        sModelPath: path to the YOLO model
        sImgFolder: folder containing images to be processed
        sNameDataSet: name of the dataset for output file name PREFIX
        sFolderOut: name of the output folder to save images with no faces detected
        conf: confidence threshold for face detection
    '''
    assert os.path.exists(sModelPath), "Model path does not exist: " + sModelPath
    assert os.path.exists(sImgFolder), "Image folder does not exist: " + sImgFolder 

    # ----- load a model -----
    model = YOLO(sModelPath)

    # ----- collect images under a folder -----
    sFolderOut = os.path.join(sImgFolder, sFolderOut)
    os.makedirs(sFolderOut, exist_ok=True)
    lstImg = []
    for filename in os.listdir(sImgFolder):
        full_path = os.path.join(sImgFolder, filename)
        if os.path.isfile(full_path) and filename.lower().endswith(sExt):
            lstImg.append(os.path.normpath(full_path))
    print(lstImg[0])
    print(lstImg[-1])

    # ----- Predict with the model -----
    for sImgPath in lstImg:
        sFileName = sNameDataSet + "_" + os.path.basename(sImgPath)
        results = model(sImgPath, verbose=False, imgsz = 32, device = 'cpu', conf = conf)  # predict on an image
        res = results[0]
        if 1 == len(res):
            # print("1 face found: pass img ", sFileName)
            continue
        sImgPathSave = os.path.join(sFolderOut, sFileName)
        if 0 == len(res):
            print("0 face found: save img ", sFileName)
            shutil.copy2(sImgPath, sImgPathSave)
        if 1 < len(res):
            print("len(res) face found: pass img (1 face a time) ", sFileName)

sModelPath = "./runs/LapaTrain/01A_theSecond400epoch_rotate_mosaic/weights/best.pt"

# ----- Collect false negative images from multiple folders of 'ffhq' dataset ----- #
conf, sNameDataSet, sFolderOut = 0.51, "ffhq", "LapaFailed"
# for i in range(0, 69001, 1000):
#     formatted_i = f"{i:05d}"
#     sImgFolder = "D:/PxyAI/DataSet2/ffhq/images1024x1024/" + formatted_i
#     CollectFalseNegative(sModelPath, sImgFolder, sNameDataSet, sFolderOut, conf = conf)

# ----- Use this script to evalulate the input model for certain img ----- #
conf, sNameDataSet, sFolderOut = 0.5, "OldFalseNegative", "FNstillFailed"
# sImgFolder = r"D:\PxyAI\DataSet\Lapa_Plus-yolo11\FalseNegative\images\train"
# CollectFalseNegative(sModelPath, sImgFolder, sNameDataSet, sFolderOut, 
#                      conf = conf, sExt = '.png')
