# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            None,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, left_feat_list, right_feat_list, mask, token_embed, left_pos_embed_list):#, right_pos_embed_list):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = left_feat_list[0].shape
        
        feat_list = []
        pos_embed_list = []
        for i in range(len(left_feat_list)):
            left_feat_list[i] = left_feat_list[i].flatten(2).permute(2, 0, 1)# [HW, B, C]
            right_feat_list[i] = right_feat_list[i].flatten(2).permute(2, 0, 1)
            feat_list.append(torch.cat((left_feat_list[i], right_feat_list[i]), dim=0))

            left_pos_embed_list[i] = left_pos_embed_list[i].flatten(2).permute(2, 0, 1) # [HW, B, C]位置编码
            # right_pos_embed_list[i] = right_pos_embed_list[i].flatten(2).permute(2, 0, 1)
            # print("feat shape:", left_feat_list[i].shape)
            # print(left_feat_list[i].shape)
            pos_embed_list.append(torch.cat((left_pos_embed_list[i], left_pos_embed_list[i]), dim=0))

        token_embed = token_embed.unsqueeze(1).repeat(1, bs, 1) # [qN, B, C]
        if mask is not None: # 初始为none
            mask = mask.flatten(1) 

        tgt = torch.zeros_like(token_embed) # 初始为0

        hs = self.decoder(
            tgt, feat_list, memory_key_padding_mask=mask, left_pos=pos_embed_list, token_pos=token_embed
        ) # [N, B, C]
        return hs.transpose(0, 1) # [B, N, C]


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.num_feature_levels = None
    def forward(
        self,
        tgt,
        left_memory_list,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        left_pos: Optional[Tensor] = None,
        token_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []
        self.num_feature_levels = len(left_memory_list)
        for i in range(self.num_layers): # output 初始是0
            level_index = i % self.num_feature_levels
            output = self.layers[i](
                output,
                left_memory_list[level_index],
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                left_pos=left_pos[level_index],
                token_pos=token_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # print("tensor shape:", tensor.shape)
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        left_memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        left_pos: Optional[Tensor] = None,
        token_pos: Optional[Tensor] = None,
    ):
        #######一次cross attention#######
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, token_pos),
            key=self.with_pos_embed(left_memory, left_pos),
            value=left_memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        # print("tgt shape:", tgt.shape)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        #######一次cross attention#######
        #      
        #######FFN#######
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        #######FFN#######
        return tgt



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
