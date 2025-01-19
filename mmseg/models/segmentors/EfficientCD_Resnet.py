import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .encoder_decoder import EncoderDecoder
import torch
from einops import rearrange


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule
from mmseg.models.backbones.resnet import BasicBlock


class CDNet(nn.Module):
    def __init__(self, neck, model_name='resnet18'):
        super(CDNet, self).__init__()
        self.model = timm.create_model(model_name, features_only=True)
        self.interaction_layers = ['blocks']
        FPN_DICT = neck
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        self.fpnA = MODELS.build(FPN_DICT)
        self.fpnB = MODELS.build(FPN_DICT)
        self.sigmoid = nn.Sigmoid()
        self.decode_layers1 = self._make_layers(FPN_DICT['out_channels'], norm_cfg, BasicBlock, FPN_DICT['num_outs'])
        self.decode_layers2 = self._make_layers(FPN_DICT['out_channels'], norm_cfg, BasicBlock, FPN_DICT['num_outs'])

    def _make_layers(self, out_channels, norm_cfg, block, num_outs):
        layers = []
        for iii in range(num_outs):
            layers.append(block(inplanes=out_channels*2, planes=out_channels*2, norm_cfg=norm_cfg))
        return nn.Sequential(*layers)

    def change_feature(self, x, y):
        i = 2
        for index in range(0, len(x), i):
            x[index], y[index] = y[index], x[index]
        return x, y
    
    def euclidean_distance(self, img1, img2):
        diff_squared = (img1 - img2) ** 2
        distances = torch.sqrt(torch.sum(diff_squared, dim=1)).unsqueeze(1)
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)
        return normalized_distances

    def forward(self, xA, xB):
        xA_list = self.model(xA)
        xB_list = self.model(xB)
        xA_list, xB_list = self.change_feature(xA_list, xB_list)
        xA_list = self.fpnA(xA_list)
        xB_list = self.fpnB(xB_list)
        xA_list, xB_list = self.change_feature(list(xA_list), list(xB_list))

        change_map = []

        curAB4 = torch.cat([xA_list[4], xB_list[4]], dim=1)
        curAB4 = self.decode_layers1[4](curAB4)
        curAB4 = F.interpolate(curAB4, scale_factor=2, mode='bilinear', align_corners=False)
        dist4 = self.euclidean_distance(xA_list[4], xB_list[4])
        dist4 = F.interpolate(dist4, scale_factor=2, mode='bilinear', align_corners=False)
        curAB4 = dist4*self.decode_layers2[4](curAB4)
        change_map.append(curAB4)

        curAB3 = torch.cat([xA_list[3], xB_list[3]], dim=1)
        curAB3 = curAB4+self.decode_layers1[3](curAB3)
        curAB3 = F.interpolate(curAB3, scale_factor=2, mode='bilinear', align_corners=False)
        dist3 = self.euclidean_distance(xA_list[3], xB_list[3])
        dist3 = F.interpolate(dist3, scale_factor=2, mode='bilinear', align_corners=False)
        curAB3 = dist3*self.decode_layers2[3](curAB3)
        change_map.append(curAB3)

        curAB2 = torch.cat([xA_list[2], xB_list[2]], dim=1)
        curAB2 = curAB3+self.decode_layers1[2](curAB2)
        curAB2 = F.interpolate(curAB2, scale_factor=2, mode='bilinear', align_corners=False)
        dist2 = self.euclidean_distance(xA_list[2], xB_list[2])
        dist2 = F.interpolate(dist2, scale_factor=2, mode='bilinear', align_corners=False)
        curAB2 = dist2*self.decode_layers2[2](curAB2)
        change_map.append(curAB2)

        curAB1 = torch.cat([xA_list[1], xB_list[1]], dim=1)
        curAB1 = curAB2+self.decode_layers1[1](curAB1)
        curAB1 = F.interpolate(curAB1, scale_factor=2, mode='bilinear', align_corners=False)
        dist1 = self.euclidean_distance(xA_list[1], xB_list[1])
        dist1 = F.interpolate(dist1, scale_factor=2, mode='bilinear', align_corners=False)
        curAB1 = dist1*self.decode_layers2[1](curAB1)
        change_map.append(curAB1)

        return change_map


@MODELS.register_module()
class EfficientCD_Resnet(EncoderDecoder):
    def __init__(
        self,
        backbone: ConfigType = None,
        decode_head: ConfigType = None,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        model_name: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(backbone=backbone, decode_head=decode_head, data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.model = CDNet(neck, model_name)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_map = self.model(imgs1, imgs2)
        # self.G_loss =  self._pxl_loss(self.G_pred1, gt) + self._pxl_loss(self.G_pred2, gt) + 0.5*(self._pxl_loss(self.G_middle1, gt)+self._pxl_loss(self.G_middle2, gt))
        loss_decode = self._decode_head_forward_train(change_map, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_map, data_samples)
            losses.update(loss_aux)
        return losses
