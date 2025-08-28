import cv2
import utilBasicRunPose
from ultralytics import YOLO

# Load a model
sModelPath, imgSize = "./runs/LapaTrain/160x1000epoch_MileStone/weights/best.pt", 160
model = YOLO(sModelPath, task='pose')

sData = "./01_BasicRunPoseSetting/DataSet-lapa-pose-plus.yaml"

metrics = model.val(
    data=sData, imgsz=imgSize, conf=0.5, iou=0.25, batch=16, device='cpu', 
    # project = TODO
    split="train",
    # split="val",
    # split="test",
    # save_json = True,
    # save_txt = True,
    single_cls = True,
    # visualize = True,
    plots=True)