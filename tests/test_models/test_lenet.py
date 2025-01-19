import timm
import torch





from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
import torch
from einops import rearrange


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

from mmseg.models.backbones.resnet import BasicBlock
from mmseg.models.utils.se_layer import SELayerIn2Out
from timm.models.swin_transformer_v2 import PatchMerging, SwinTransformerV2Block
import torch.utils.checkpoint as checkpoint
from typing import Tuple, Union
from timm.layers import to_2tuple

_int_or_tuple_2_t = Union[int, Tuple[int, int]]

class SwinTBlock(nn.Module):
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
        return x

    def _init_respostnorm(self) -> None:
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


x = torch.rand(2, 64, 64, 128)
model = SwinTBlock(dim=128, out_dim=128, input_resolution=(64, 64))
y = model(x)
print(y.shape)
