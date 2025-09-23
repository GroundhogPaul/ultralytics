# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import deepcopy
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

from ultralytics.data.augment import *

def v8_transforms_PXY(dataset, imgsz, hyp, stretch=False):
    print("!!!!! using v8_transforms_PXY !!!!!")
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste > 0.01:
        if hyp.copy_paste_mode == "flip":
            pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
        else:
            pre_transform.append(
                CopyPaste(
                    dataset,
                    pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                    p=hyp.copy_paste,
                    mode=hyp.copy_paste_mode,
                )
            )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")
    
    lstCompose = []
    lstCompose.append(pre_transform)
    if hyp.mixup > 0.005:
        lstCompose.append(MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup))
    lstCompose.append(Albumentations(p=1.0))
    lstCompose.append(RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v))
    if hyp.flipud > 0.005:
        lstCompose.append(RandomFlip(direction="vertical", p=hyp.flipud))
    if hyp.fliplr > 0.005:
        lstCompose.append(RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx))

    return Compose(lstCompose)