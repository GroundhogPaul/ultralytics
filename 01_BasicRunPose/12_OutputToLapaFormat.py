'''
create Lapa format dataset from trained .pt 
'''

import cv2
import utilBasicRunPose
from ultralytics import YOLO
import os
import shutil

# ----- load a model -----
sModelPath = "./runs/LapaTrain/01A_2nd300epoch_rotate_mosaic_640/weights/best.pt"
model = YOLO(sModelPath)

# ----- collect images under a folder -----
sNameDataSet = "LapaFN01"
sImgsFolder = r"D:/PxyAI/DataSet2/ffhq/images1024x1024/LapaFailed"
sFolderOut = os.path.join(sImgsFolder, sNameDataSet + "_withLapaLabel")

# the 3 possible output type
sFolderOut_0face = os.path.join(sFolderOut, "0face")
sFolderOut_1face = os.path.join(sFolderOut, "1face") 
sFolderOut_2face = os.path.join(sFolderOut, "2face") # means 2 or more faces

os.makedirs(sFolderOut, exist_ok=True)
os.makedirs(sFolderOut_0face, exist_ok=True)
os.makedirs(sFolderOut_1face, exist_ok=True)
os.makedirs(sFolderOut_2face, exist_ok=True)

# collect images under a folder
lstImg = []
for filename in os.listdir(sImgsFolder):
    full_path = os.path.join(sImgsFolder, filename)
    if os.path.isfile(full_path) and filename.lower().endswith('.png'):
        lstImg.append(os.path.normpath(full_path))
print(lstImg)

# ----- Predict with the model -----
for sImgPath in lstImg:
    sFileName = os.path.basename(sImgPath)
    results = model(sImgPath, verbose=False, imgsz = 640, device = 'cpu')  # predict on an image
    res = results[0]

    # ----- save img ----- #
    if 0 == len(res):
        print("0 face found: pass img ", sFileName)
        shutil.copy2(sImgPath, sFolderOut_0face)
        continue

    img = res.plot(line_width=2, kpt_radius=7)
    if 1 < len(res):
        print(len(res), " face found: pass img ", sFileName)
        sImgPathSave = os.path.join(sFolderOut_2face, sFileName)
        cv2.imwrite(sImgPathSave, img)
        continue

    if 1 == len(res): # this is the case we want, both img and label
        print("1 face found: save img and create label: ", sFileName)
        sFileNamePlot = os.path.splitext(sFileName)[0] + "_plot.png" 
        sImgPlotSavePath = os.path.join(sFolderOut_1face, sFileNamePlot)
        cv2.imwrite(sImgPlotSavePath, img)

        pngIn = cv2.imread(sImgPath)
        sImgPathSave = os.path.join(sFolderOut_1face, sFileName).replace(".png", ".jpg")
        cv2.imwrite(sImgPathSave, pngIn)
    
    # ----- save label file in Lapa format ----- #
    sLabelFile = os.path.splitext(sFileName)[0] + ".txt"
    sLabelPath = os.path.join(sFolderOut_1face, sLabelFile)

    with open(sLabelPath, 'w') as f:
        f.write(f"106\n")
        lstXY = res[0].keypoints.xy[0].cpu().numpy().tolist()
        for x,y in lstXY:
            # print(x, y)
            f.write(f"{x:.6f} {y:.6f}\n")