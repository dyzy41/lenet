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
from .bit import BIT
from .cdnet import CDNet
from .dsifn import DSIFN
from .lunet import LUNet
from .p2v import P2VNet
from .stanet import STANet
from .siamunet_conc import SiamUNet_conc
from .siamunet_diff import SiamUNet_diff
from .mfpnet import MFPNET
from .sunet import SUNnet
from .CGNet import CGNet, HCGMNet
from .AFCF3D_Net import AFCF3D
from .elgcnet import ELGCNet
from .acabfnet import ACABFNet
from .dminet import DMINet
from .CDNeXt.cdnext import CDNeXt
from .GASNet import GASNet
from .hanet import HANet
from .isdanet import ISDANet
from .STRobustNet.STRobustNet import STRobustNet
from .ScratchFormer.scratch_former import ScratchFormer
from .DARNet.DARNet import DARNet
from .BASNet import BASNet
from .rctnet import BaseNet



@MODELS.register_module()
class MM_RCTNet(EncoderDecoder):
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

        self.model = BaseNet()    
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


@MODELS.register_module()
class MM_BASNet(EncoderDecoder):
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

        self.model = BASNet()    
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


@MODELS.register_module()
class MM_DARNet(EncoderDecoder):
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

        self.model = DARNet()    
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


@MODELS.register_module()
class MM_ScratchFormer(EncoderDecoder):
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

        self.model = ScratchFormer()    
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



@MODELS.register_module()
class MM_STRobustNet(EncoderDecoder):
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

        self.model = STRobustNet()    
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


@MODELS.register_module()
class MM_ISDANet(EncoderDecoder):
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

        self.model = ISDANet()    
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


@MODELS.register_module()
class MM_HANet(EncoderDecoder):
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

        self.model = HANet()    
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


@MODELS.register_module()
class MM_GASNet(EncoderDecoder):
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

        
        self.model = GASNet()    
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



@MODELS.register_module()
class MM_CDNeXt(EncoderDecoder):
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

        
        self.model = CDNeXt(out_channels=2)    
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



@MODELS.register_module()
class MM_DMINet(EncoderDecoder):
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

        self.model = DMINet()

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


@MODELS.register_module()
class MM_HATNet(EncoderDecoder):
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

        self.model = HATNet()

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



@MODELS.register_module()
class MM_ELGCNet(EncoderDecoder):
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

        self.model = ELGCNet()

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


@MODELS.register_module()
class MM_ACABFNet(EncoderDecoder):
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

        self.model = ACABFNet()

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


@MODELS.register_module()
class MM_AFCF3D(EncoderDecoder):
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

        self.model = AFCF3D()

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


@MODELS.register_module()
class MM_HCGMNet(EncoderDecoder):
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

        self.model = HCGMNet()

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
        loss_decode = self._decode_head_forward_train(change_map, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_map, data_samples)
            losses.update(loss_aux)
        return losses


@MODELS.register_module()
class MM_CGNet(EncoderDecoder):
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

        self.model = CGNet()

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
        loss_decode = self._decode_head_forward_train(change_map, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_map, data_samples)
            losses.update(loss_aux)
        return losses


@MODELS.register_module()
class MM_MSCANet(EncoderDecoder):
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

        self.model = MSCANet()

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
        loss_decode = self._decode_head_forward_train(change_map, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_map, data_samples)
            losses.update(loss_aux)
        return losses


@MODELS.register_module()
class MM_SUNet(EncoderDecoder):
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

        self.model = SUNnet()

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict([x], batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_map = self.model(imgs1, imgs2)
        loss_decode = self._decode_head_forward_train([change_map], data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train([change_map], data_samples)
            losses.update(loss_aux)
        return losses


@MODELS.register_module()
class MM_MFPNet(EncoderDecoder):
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

        self.model = MFPNET(2)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict([x], batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_map = self.model(imgs1, imgs2)
        loss_decode = self._decode_head_forward_train([change_map], data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train([change_map], data_samples)
            losses.update(loss_aux)
        return losses


@MODELS.register_module()
class MM_SiamUNet_diff(EncoderDecoder):
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

        self.model = SiamUNet_diff(3, 2)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict([x], batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_map = self.model(imgs1, imgs2)
        loss_decode = self._decode_head_forward_train([change_map], data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train([change_map], data_samples)
            losses.update(loss_aux)
        return losses


@MODELS.register_module()
class MM_SiamUNet_conc(EncoderDecoder):
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

        self.model = SiamUNet_conc(3, 2)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict([x], batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_map = self.model(imgs1, imgs2)
        loss_decode = self._decode_head_forward_train([change_map], data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train([change_map], data_samples)
            losses.update(loss_aux)
        return losses


@MODELS.register_module()
class MM_STANet(EncoderDecoder):
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

        self.model = STANet()

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



@MODELS.register_module()
class MM_BIT(EncoderDecoder):
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

        self.model = BIT(3, 2)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict([x], batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_map = self.model(imgs1, imgs2)
        loss_decode = self._decode_head_forward_train([change_map], data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_map, data_samples)
            losses.update(loss_aux)
        return losses


@MODELS.register_module()
class MM_CDNet(EncoderDecoder):
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

        self.model = CDNet(6, 2)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict([x], batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_map = self.model(imgs1, imgs2)
        loss_decode = self._decode_head_forward_train([change_map], data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train([change_map], data_samples)
            losses.update(loss_aux)
        return losses


@MODELS.register_module()
class MM_DSIFN(EncoderDecoder):
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

        self.model = DSIFN()

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
        loss_decode = self._decode_head_forward_train(change_map, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_map, data_samples)
            losses.update(loss_aux)
        return losses



@MODELS.register_module()
class MM_LUNet(EncoderDecoder):
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

        self.model = LUNet(3, 2)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict([x], batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_map = self.model(imgs1, imgs2)
        loss_decode = self._decode_head_forward_train([change_map], data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_map, data_samples)
            losses.update(loss_aux)
        return losses



@MODELS.register_module()
class MM_P2V(EncoderDecoder):
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

        self.model = P2VNet(3)

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
        loss_decode = self._decode_head_forward_train(change_map, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_map, data_samples)
            losses.update(loss_aux)
        return losses