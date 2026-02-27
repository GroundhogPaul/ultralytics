# predict for the first time using source code
# see documentation B2.1.1

import cv2
import utilBasicRunDet
from ultralytics import YOLO
from thop import profile
import torch

# Load a model
model = YOLO("./runs/LapaTrain/192x_72LM4/weights/best.pt")  # load an official model
model = model.model

img = torch.randn(1, 3, 192, 176)
# model(img)
flops, params = profile(model, inputs = (img,))
gflops = flops/1e9
imgSizeM = img.shape[2]*img.shape[3]*3/1e6
print(f"FLOPs: {gflops:.2f}G, Params: {params/1e6:.2f}M, gFlops per 1M pixel = {gflops/imgSizeM:.2f}G")