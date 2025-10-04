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
from ultralytics.nn.tasks import PoseModel

class ModifyYoloPoseN():

    def __init__(self):
        # assert False, "Do not use this class, it is only for reference."
        # assert isinstance(model, )
        super().__init__()

    def shrink_bn(model, l1_lambda):

        def _shrink_bn(bn, l1_lambda):
            assert isinstance(bn, nn.BatchNorm2d)
            bn.weight.grad.data.add_(l1_lambda * torch.sign(bn.weight.data))
            bn.bias.grad.data.add_(1e-2 * torch.sign(bn.bias.data))

        assert isinstance(model, PoseModel)
        seq = model.model
        assert isinstance(seq, nn.modules.container.Sequential)

        # ----- shrink the BN of hidden layer of bottleneck ----- #
        for name, m in seq.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                _shrink_bn(m, l1_lambda)

        # ----- shrink the BN of Conv and C3k2 layer ----- #
        for i, layer in enumerate(seq):
            if i == 3 or i == 5 or i == 8 or i == 19 or i == 22:
                assert isinstance(layer, Conv)
                _shrink_bn(layer.bn, l1_lambda)
            
            if i == 2 or i == 4 or i == 6 or i == 9 or i == 15 or i == 18 or i == 21 or i == 24:
                assert isinstance(layer, C3k2)
                _shrink_bn(layer.cv2.bn, l1_lambda)
        
        # ----- shrink the Pose ?? TODO ----- #
        myPose = seq[-1]
        assert isinstance(layer, Pose)
        for idxLayer in range(myPose.nl):
            PoseSubSeq = myPose.cv4[idxLayer]
            assert isinstance(PoseSubSeq, nn.modules.container.Sequential)

            conv0 = PoseSubSeq[0]
            assert isinstance(conv0, Conv)
            _shrink_bn(conv0.bn, l1_lambda)

            conv1 = PoseSubSeq[1]
            assert isinstance(conv1, Conv)
            _shrink_bn(conv1.bn, l1_lambda)
        
        return 0
    
    def prune_model(model):
        assert isinstance(model, PoseModel)
        seq = model.model
        assert isinstance(seq, nn.modules.container.Sequential)



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