# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None

from ultralytics.nn.tasks import *

class ModifyYoloPoseN():

    def __init__(self, model):
        assert False, "Do not use this class, it is only for reference."
        # assert isinstance(model, )
        super().__init__()
        self.model = model

    def shrink_with_bn(self):
        return

    def prune_model(self):
        return self.model


class RawDnsLoss():
    """Criterion class for computing raw denoise losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the RawDns class, taking a de-paralleled model as argument."""
        super().__init__(model)

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(1, device=self.device)  # 
        # dns = preds # the only output is the denoised image
        batch_size, _, mask_h, mask_w = preds.shape  # batch size, number of masks, mask height, mask width
        # pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        #     (self.reg_max * 4, self.nc), 1
        # )

        # B, grids, ..
        # pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        # pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        # dtype = pred_scores.dtype
        # imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            # targets = torch.cat((batch_idx, batch["dns"].view(-1, 1)), 1)
            # mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ raw denoise dataset incorrectly formatted or not a raw denoise dataset.\n"
                "This error can occur when incorrectly training a 'raw denoise' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\n"
            ) from e

        # Masks loss
        masks = batch["masks"].to(self.device).float()
        assert tuple(masks.shape[-2:]) == (mask_h, mask_w)
        loss[0] = abs(masks - preds)
        return loss.sum() * batch_size, loss.detach()  # loss(denoise)

class RawDns(DetectionModel):
    """RawD Denoise model."""

    def __init__(self, cfg="unet.yaml", ch=4, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=None, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return RawDnsLoss(self)