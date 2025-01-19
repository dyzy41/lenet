import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from .lenet4 import LENet4



@MODELS.register_module()
class MM_LENet4(EncoderDecoder):
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
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(backbone=backbone, decode_head=decode_head, data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.model = LENet4(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_maps = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict(change_maps, batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_maps = self.model(imgs1, imgs2)
        loss_decode = self._decode_head_forward_train(change_maps, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_maps, data_samples)
            losses.update(loss_aux)
        return losses