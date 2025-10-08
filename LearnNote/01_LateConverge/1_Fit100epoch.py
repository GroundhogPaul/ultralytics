import numpy as np
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] = script_dir + os.pathsep + os.environ['PATH']
sys.path.append(str(script_dir))

# sResCsv = os.path.join(script_dir, r"176x1000epoch_MileStone\results.csv")
# sResCsv = os.path.join(script_dir, r"176xPruneStep2_1500epoch_L1shrinkBN\results.csv")
sResCsv = os.path.join(script_dir, r"176x500epochTest_LR0.001\results.csv")
sResCsv = os.path.join(script_dir, r"160x1000epoch_MileStone\results.csv")

data = np.genfromtxt(sResCsv, delimiter=',', names=True, encoding=None)

epochs = data['epoch']
box_loss = data['trainbox_loss']
pose_loss = data['trainpose_loss']
lrs = data['lrpg0']

# 打印验证
# print("Epochs:", epochs)
# print("Box Loss:", box_loss)
# print("Pose Values:", pose)
epochBeg = 500
epochEnd = 1000
epochStep = 100
for epoch in np.arange(epochBeg, epochEnd, epochStep):
    if epoch + epochStep > len(epochs):
        break
    slopeBox, _ = np.polyfit(epochs[epoch: epoch + epochStep], box_loss[epoch: epoch + epochStep], 1)
    slopePose, _ = np.polyfit(epochs[epoch: epoch + epochStep], pose_loss[epoch: epoch + epochStep], 1)
    print(f"epoch: {epoch}, lr = {lrs[epoch + epochStep // 2]:.6f}, slopeBox = {slopeBox:.6f}, slopePose = {slopePose:.6f}")