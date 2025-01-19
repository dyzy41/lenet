import timm
import torch




from einops import rearrange
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
import torch

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule

from mmseg.models.backbones.resnet import BasicBlock
from mmseg.models.utils.se_layer import SELayerIn2Out
from timm.models.swin_transformer_v2 import PatchMerging, SwinTransformerV2Block
import torch.utils.checkpoint as checkpoint
from typing import Tuple, Union
from timm.layers import to_2tuple

_int_or_tuple_2_t = Union[int, Tuple[int, int]]

class SwinTV2Block(nn.Module):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: _int_or_tuple_2_t,
            depth: int=2,
            num_heads: int=8,
            window_size: _int_or_tuple_2_t=8,
            downsample: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0., 
            attn_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            pretrained_window_size: _int_or_tuple_2_t = 0,
            output_nchw: bool = False,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x.permute(0, 3, 1, 2)

    def _init_respostnorm(self) -> None:
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


# class CDWeights3D(nn.Module):
#     def __init__(self, channels=256, norm_cfg=dict(type='SyncBN', requires_grad=True)):
#         super(CDWeights3D, self).__init__()
#         self.convA = ConvModule(in_channels=channels*2, out_channels=channels, kernel_size=3, padding=1, norm_cfg=norm_cfg)
#         self.convB = ConvModule(in_channels=channels*2, out_channels=channels, kernel_size=3, padding=1, norm_cfg=norm_cfg)
#         self.sigmoid = nn.Sigmoid()
#         self.pool = nn.AdaptiveAvgPool2d((8, 8))

#     def forward(self, xA, xB):
#         N, C, H, W = xA.shape
#         # spatial difference
#         # xA_flat = xA.reshape(-1, C)
#         # xB_flat = xB.reshape(-1, C)

#         xA_flat = rearrange(xA, 'N C H W -> (N H W) C')
#         xB_flat = rearrange(xB, 'N C H W -> (N H W) C')
#         cosine_sim_hw = F.cosine_similarity(xA_flat, xB_flat, dim=1)
        
#         cosine_sim_hw = cosine_sim_hw.view(N, H, W)
#         cosine_sim_hw = cosine_sim_hw.unsqueeze(1)
#         cosine_sim_hw = 1-self.sigmoid(cosine_sim_hw)

#         xA_hw = cosine_sim_hw*xA
#         xB_hw = cosine_sim_hw*xB

#         # channel difference
#         # xA_flat2 = xA.reshape(-1, H*W)
#         # xB_flat2 = xB.reshape(-1, H*W)
#         xA_flat2 = rearrange(xA, 'N C H W -> (N C) (H W)')
#         xB_flat2 = rearrange(xB, 'N C H W -> (N C) (H W)')
#         cosine_sim_c = F.cosine_similarity(xA_flat2, xB_flat2, dim=1)
#         cosine_sim_c = cosine_sim_c.view(N, C).unsqueeze(-1).unsqueeze(-1)
#         cosine_sim_c = 1-self.sigmoid(cosine_sim_c)

#         xA_c = cosine_sim_c*xA
#         xB_c = cosine_sim_c*xB

#         xA = self.convA(torch.cat([xA_hw, xA_c], dim=1))
#         xB = self.convB(torch.cat([xB_hw, xB_c], dim=1))
#         return xA, xB  


class CDWeights(nn.Module):
    def __init__(self, channels=128, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(CDWeights, self).__init__()
        self.convA = BasicBlock(channels, planes=channels, norm_cfg=norm_cfg)
        self.convB = BasicBlock(channels, planes=channels, norm_cfg=norm_cfg)
        self.sigmoid = nn.Sigmoid()

    def spatial_difference(self, xA, xB):
        xA_flat = xA.permute(0, 2, 3, 1).reshape(-1, xA.size(1))
        xB_flat = xB.permute(0, 2, 3, 1).reshape(-1, xB.size(1))
        cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)
        cosine_sim = cosine_sim.view(xA.size(0), xA.size(2), xA.size(3))
        cosine_sim = cosine_sim.unsqueeze(1)
        c_weights = 1-self.sigmoid(cosine_sim)
        return c_weights

    def channel_difference(self, xA, xB):
        N, C, H, W = xA.shape
        xA_flat = xA.view(N, C, -1)
        xB_flat = xB.view(N, C, -1)
        cosine_sim = 1-self.sigmoid(F.cosine_similarity(xA_flat, xB_flat, dim=2))
        hw_weights = cosine_sim.unsqueeze(-1).unsqueeze(-1)
        return hw_weights

    def forward(self, xA, xB):
        # 计算空间权重和通道权重
        c_weights = self.spatial_difference(xA, xB)  # (2, 1, 32, 32)
        hw_weights = self.channel_difference(xA, xB)  # (2, 128, 1, 1)

        # 将 c_weights 扩展到与 hw_weights 相同的形状
        c_weights_expanded = c_weights.expand(-1, hw_weights.size(1), -1, -1)  # (2, 128, 32, 32)

        # 合并权重 (比如可以选择相乘，也可以进行加权平均)
        combined_weights = c_weights_expanded * hw_weights  # (2, 128, 32, 32)

        # 对 xA 和 xB 进行加权处理
        xA_weighted = xA * combined_weights
        xB_weighted = xB * combined_weights

        # 通过卷积层处理
        xA_d = self.convA(xA_weighted)
        xB_d = self.convB(xB_weighted)

        # 得到最终输出
        outA = xA_d + xA
        outB = xB_d + xB

        return outA, outB


class LENet4(nn.Module):
    def __init__(self, neck, model_name='efficientnet_b5'):
        super(LENet4, self).__init__()
       # self.model = timm.create_model('swinv2_base_window8_256', pretrained=True, pretrained_cfg_overlay=dict(file='/home/dongsijun/project/pretrained/swinv2_base_window8_256'), features_only=True)
        self.model = timm.create_model('swinv2_base_window8_256', pretrained=True, pretrained_cfg_overlay=dict(file='pretrained/swinv2_base_window8_256.ms_in1k/pytorch_model.bin'), features_only=True)
        self.interaction_layers = ['patch_embed']
        '''
        swinv2_base_window8_256
        torch.Size([16, 64, 64, 128])
        torch.Size([16, 32, 32, 256])
        torch.Size([16, 16, 16, 512])
        torch.Size([16, 8, 8, 1024])
        '''
        self.interaction_layers = ['blocks']
        FPN_DICT = neck
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
        self.fpnA = MODELS.build(FPN_DICT)
        self.fpnB = MODELS.build(FPN_DICT)

        self.decode_layersA = nn.Sequential(
            nn.Identity(),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=64, num_heads=4),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=32, num_heads=8),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=16, num_heads=16),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=8, num_heads=32)
        )
        self.decode_layersB = nn.Sequential(
            nn.Identity(),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=64, num_heads=4),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=32, num_heads=8),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=16, num_heads=16),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=8, num_heads=32)
        )

        self.channelA = nn.Sequential(
            nn.Identity(),
            SELayerIn2Out(in_channels=FPN_DICT['out_channels']*2, out_channels=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=FPN_DICT['out_channels']*2, out_channels=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=FPN_DICT['out_channels']*2, out_channels=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=FPN_DICT['out_channels']*2, out_channels=FPN_DICT['out_channels'], norm_cfg=norm_cfg)
        )
        self.channelB = nn.Sequential(
            nn.Identity(),
            SELayerIn2Out(in_channels=FPN_DICT['out_channels']*2, out_channels=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=FPN_DICT['out_channels']*2, out_channels=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=FPN_DICT['out_channels']*2, out_channels=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=FPN_DICT['out_channels']*2, out_channels=FPN_DICT['out_channels'], norm_cfg=norm_cfg)
        )

        self.sigmoid = nn.Sigmoid()
        self.re_weight3 = CDWeights(256, norm_cfg=norm_cfg)
        self.re_weight2 = CDWeights(256, norm_cfg=norm_cfg)
        self.re_weight1 = CDWeights(256, norm_cfg=norm_cfg)
        
        self.rwe = nn.Sequential(
            CDWeights(128, norm_cfg=norm_cfg),
            CDWeights(128, norm_cfg=norm_cfg),
            CDWeights(256, norm_cfg=norm_cfg),
            CDWeights(512, norm_cfg=norm_cfg),
            CDWeights(1024, norm_cfg=norm_cfg)
        )


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

        xA_list = []
        xB_list = []
        ii = 0
        for name, module in self.model.named_children():
            # print(f"Module Name: {name}")
            if name in self.interaction_layers:
                xA = module(xA)
                xB = module(xB)
            else:
                xA = module(xA)
                xB = module(xB)
                xA = xA.permute(0, 3, 1, 2)
                xB = xB.permute(0, 3, 1, 2)
                xA, xB = self.rwe[ii](xA, xB)
                xA = xA.permute(0, 2, 3, 1)
                xB = xB.permute(0, 2, 3, 1)
                xA_list.append(xA)
                xB_list.append(xB)
                ii+=1

        xA_list = xA_list[1:]
        xA_list = [x.permute(0, 3, 1, 2) for x in xA_list]
        xB_list = xB_list[1:]
        xB_list = [x.permute(0, 3, 1, 2) for x in xB_list]
        
        xA_list, xB_list = self.change_feature(xA_list, xB_list)
        xA_list = self.fpnA(xA_list)
        xB_list = self.fpnB(xB_list)
        xA_list, xB_list = self.change_feature(list(xA_list), list(xB_list))

        # xA_list[0], xB_list[0] = self.rwe1(xA_list[0], xB_list[0])
        # xA_list[1], xB_list[1] = self.rwe2(xA_list[1], xB_list[1])
        # xA_list[2], xB_list[2] = self.rwe3(xA_list[2], xB_list[2])
        # xA_list[3], xB_list[3] = self.rwe4(xA_list[3], xB_list[3])

        xA_list = [x.permute(0, 2, 3, 1) for x in xA_list]
        xB_list = [x.permute(0, 2, 3, 1) for x in xB_list]
        xA1, xA2, xA3, xA4 = xA_list
        xB1, xB2, xB3, xB4 = xB_list

        change_maps = []
        xA4_ = self.decode_layersA[4](xA4)
        xB4_ = self.decode_layersB[4](xB4)
        xA4_ = torch.cat([xA4_, xB4.permute(0, 3, 1, 2)], dim=1)
        xB4_ = torch.cat([xB4_, xA4.permute(0, 3, 1, 2)], dim=1)
        xA4 = self.channelA[4](xA4_)
        xB4 = self.channelB[4](xB4_)
        xA4 = F.interpolate(xA4, scale_factor=2, mode='bilinear', align_corners=False)
        xB4 = F.interpolate(xB4, scale_factor=2, mode='bilinear', align_corners=False)
        # change_maps.append(torch.cat([xA4, xB4], dim=1))

        # xA3 = torch.cat([xA3, xA4.permute(0, 2, 3, 1)], dim=-1)
        # xB3 = torch.cat([xB3, xB4.permute(0, 2, 3, 1)], dim=-1)
        xA3 = xA3+xB4.permute(0, 2, 3, 1)
        xB3 = xB3+xA4.permute(0, 2, 3, 1)
        xA3_ = self.decode_layersA[3](xA3)
        xB3_ = self.decode_layersB[3](xB3)
        xA3_ = torch.cat([xA3_, xB3.permute(0, 3, 1, 2)], dim=1)
        xB3_ = torch.cat([xB3_, xA3.permute(0, 3, 1, 2)], dim=1) 
        xA3 = self.channelA[3](xA3_)
        xB3 = self.channelB[3](xB3_)
        xA3 = F.interpolate(xA3, scale_factor=2, mode='bilinear', align_corners=False)
        xB3 = F.interpolate(xB3, scale_factor=2, mode='bilinear', align_corners=False)
        xA3, xB3 = self.re_weight3(xA3, xB3)
        change_maps.append(torch.cat([xA3, xB3], dim=1))

        # xA2 = torch.cat([xA2, xA3.permute(0, 2, 3, 1)], dim=-1)
        # xB2 = torch.cat([xB2, xB3.permute(0, 2, 3, 1)], dim=-1)
        xA2 = xA2+xB3.permute(0, 2, 3, 1)
        xB2 = xB2+xA3.permute(0, 2, 3, 1)
        xA2_ = self.decode_layersA[2](xA2)
        xB2_ = self.decode_layersB[2](xB2)
        xA2_ = torch.cat([xA2_, xB2.permute(0, 3, 1, 2)], dim=1)
        xB2_ = torch.cat([xB2_, xA2.permute(0, 3, 1, 2)], dim=1)
        xA2 = self.channelA[2](xA2_)
        xB2 = self.channelB[2](xB2_)
        xA2 = F.interpolate(xA2, scale_factor=2, mode='bilinear', align_corners=False)
        xB2 = F.interpolate(xB2, scale_factor=2, mode='bilinear', align_corners=False)
        xA2, xB2 = self.re_weight2(xA2, xB2)
        change_maps.append(torch.cat([xA2, xB2], dim=1))

        # xA1 = torch.cat([xA1, xA2.permute(0, 2, 3, 1)], dim=-1)
        # xB1 = torch.cat([xA1, xB2.permute(0, 2, 3, 1)], dim=-1)
        xA1 = xA1+xB2.permute(0, 2, 3, 1)
        xB1 = xB1+xA2.permute(0, 2, 3, 1)
        xA1_ = self.decode_layersA[1](xA1)
        xB1_ = self.decode_layersB[1](xB1)
        xA1_ = torch.cat([xA1_, xB1.permute(0, 3, 1, 2)], dim=1)
        xB1_ = torch.cat([xB1_, xA1.permute(0, 3, 1, 2)], dim=1)
        xA1 = self.channelA[1](xA1_)
        xB1 = self.channelB[1](xB1_)
        xA1, xB1 = self.re_weight1(xA1, xB1)
        change_maps.append(torch.cat([xA1, xB1], dim=1))
        return change_maps
