import numpy as np
import utilBasicRunDet
import ncnn
import torch
import cv2
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.plotting import Annotator
import os

# ----- Load an image and prepare it for inference ----- #
sImgIn = "BabySmile_192x160.jpg"
inBGR_HWCh = cv2.imread(sImgIn)
inRGB_BChHW= utilBasicRunDet.read_image_BChHW(sImgIn, bNormalize=True, bSwapRB=True, bChHW=True)
inRGB_BChHW = torch.tensor(inRGB_BChHW)

# ----- Load the NCNN model and run inference ----- #
sFolderModel = "runs/Det/yolo11nOfficial/yolo11n_ncnn_model/"
out = []
with ncnn.Net() as net:
    net.load_param(os.path.join(sFolderModel, "model.ncnn.param"))
    net.load_model(os.path.join(sFolderModel, "model.ncnn.bin"))

    with net.create_extractor() as ex:
        in0 = inRGB_BChHW.squeeze(0).numpy()
        ex.input("in0", ncnn.Mat(in0).clone())

        _, out0 = ex.extract("out0")
        out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

# ----- Post-process the output ----- #
outNMS = non_max_suppression(
            out,
            0.25,
            0.7,
            agnostic=False,
            max_det=300,
            classes=None,
            nc=1,
        )

print(outNMS[0].shape)
outNMS = outNMS[0]

# ----- Annotate the image with the results ----- #
myAnnotator = Annotator(inBGR_HWCh, line_width=2)
for i in range(outNMS.shape[0]):
    myAnnotator.box_label(
        box=outNMS[i, :4].cpu().numpy(),
        label=f"{int(outNMS[i, 5])} {outNMS[i, 4]:.2f}",
    )
    # kpts = outNMS[i, 6:6+106*3].cpu().numpy().reshape(-1, 3)
    # myAnnotator.kpts(kpts=kpts)

cv2.imwrite("05a_TestNcnnMyCode.jpg", myAnnotator.result())