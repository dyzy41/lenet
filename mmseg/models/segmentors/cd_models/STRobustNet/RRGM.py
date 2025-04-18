# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv2d

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer


class RRGM(nn.Module): #
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        dec_layers: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            dec_layers: number of Transformer decoder layers
        """
        super().__init__()

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_decoder_layers=dec_layers,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.token_embed = nn.Embedding(num_queries, hidden_dim) # 随机初始化
        
    def forward(self, encoder_feature_list, encoder_feature_B_list):
        # [B, 128, H/4, W/4]
        left_pos_list = []
        for i in range(len(encoder_feature_list)):
            left_pos_list.append(self.pe_layer(encoder_feature_list[i]))
        mask = None

        token_feat = self.transformer(encoder_feature_list, encoder_feature_B_list, mask, self.token_embed.weight, left_pos_list)#, pos_em) # [B, N, 128]
        return token_feat# , self.token_embed.weight



