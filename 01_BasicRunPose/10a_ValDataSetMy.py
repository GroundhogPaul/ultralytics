'''
collect false negative images from a folder of images
and save them to a specified output folder.
'''

import cv2
import utilBasicRunPose
from ultralytics import YOLO
import os
import shutil
from matplotlib import pyplot as plt

sModelPath, imgSize, conf = r"./runs/LapaTrain/176x_ds16_1000epoch_MileStone\weights/best.pt", 176, 0.5
sModelFolder = os.path.dirname(os.path.dirname(sModelPath))
sImgFolder, sExt = r"D:\PxyAI\DataSet\Lapa-yolo11\valTinyMultiResolution\images\val", 'jpg'
print("----- ", sModelPath, imgSize, conf, " -----")

# ----- Collect false negative images from val dataset ----- #
assert os.path.exists(sModelPath), "Model path does not exist: " + sModelPath
assert os.path.exists(sImgFolder), "Image folder does not exist: " + sImgFolder 

# ----- load a model -----
model = YOLO(sModelPath)

mapFaceRatioToScore = {}
mapFaceRatioToRecall = {}
mapFaceRatioToMiss = {}

lstImg = []
for filename in os.listdir(sImgFolder):
    full_path = os.path.join(sImgFolder, filename)
    if os.path.isfile(full_path) and filename.lower().endswith(sExt):
            lstImg.append(os.path.normpath(full_path))

# ----- Predict with the model -----
for sImgPath in lstImg:
    print(sImgPath)
    sFileName = sImgFolder + "_" + os.path.basename(sImgPath)
    faceRatio = float(sFileName[-8:-4])
    results = model(sImgPath, verbose=False, imgsz = imgSize, device = 'cpu', conf = conf, rect = False)
    res = results[0]
    if 1 == len(res):
        print("len(res) face found: pass img (1 face a time) ", sFileName)
        mapFaceRatioToRecall[faceRatio] = mapFaceRatioToRecall.get(faceRatio, 0) + 1
        mapFaceRatioToScore[faceRatio] = mapFaceRatioToScore.get(faceRatio, 0) + res.boxes.conf[0].item()
        continue
    if 0 == len(res):
        print("0 face found: save img ", sFileName)
        mapFaceRatioToMiss[faceRatio] = mapFaceRatioToMiss.get(faceRatio, 0) + 1

lstFaceRatio = list(set(mapFaceRatioToRecall.keys()) | set(mapFaceRatioToMiss.keys()))
lstFaceRatio = sorted(lstFaceRatio)
lstFaceScore = []
lstFaceRecall = []
for faceRatio in lstFaceRatio:
    recall = mapFaceRatioToRecall.get(faceRatio, 0)
    miss = mapFaceRatioToMiss.get(faceRatio, 0)
    score = mapFaceRatioToScore.get(faceRatio, 0)
    lstFaceScore.append(score / recall if recall > 0 else 0)
    lstFaceRecall.append(recall / (recall + miss) if (recall + miss) > 0 else 0)

# ----- plot the result -----
plt.plot(lstFaceRatio, lstFaceScore, marker='o', label='Face Score')
plt.plot(lstFaceRatio, lstFaceRecall, marker='o', label='Face Recall')
plt.legend()
plt.xlabel('Face Ratio')
plt.ylabel('Face Score')
plt.grid(True)          
plt.savefig(os.path.join(sModelFolder, 'AAA_FaceScoreAndRecall.jpg'),
            dpi=300,     
            bbox_inches='tight', 
            format='jpeg') 

# ----- save res to csv -----
with open(os.path.join(sModelFolder, 'AAA_FaceScoreAndRecall.csv'), 'w') as f:
    f.write('FaceRatio,FaceScore,FaceRecall\n')
    for i in range(len(lstFaceRatio)):
        f.write(f'{lstFaceRatio[i]:.2f},{lstFaceScore[i]:.2f},{lstFaceRecall[i]:.2f}\n')