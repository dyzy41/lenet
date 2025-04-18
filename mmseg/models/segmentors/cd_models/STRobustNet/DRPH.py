import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature import BasicBlock



def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


class DRPH(nn.Module):
    def __init__(self, num_queries):
        super(DRPH, self).__init__()

        
        self.att_fusion = nn.Sequential(
            nn.Conv2d(num_queries*2, num_queries*2, stride=1, kernel_size=1),
            nn.BatchNorm2d(num_queries*2),
            nn.ReLU(),
            nn.Conv2d(num_queries*2, 2, stride=1, kernel_size=1))
        
        self.conv = conv2d(8, 32)
        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, att_maps, img_A, img_B):
        """
        Args:
            att_maps: [B, 2*N, H, W]
            left_A: [B, 3, H, W]
            right_B: [B, 3, H, W]
        """

        feat = self.att_fusion(att_maps)
        concat = torch.cat((feat, img_A, img_B), dim=1)
        out = self.conv(concat)
        out = self.dilated_blocks(out)
        final_out = self.final_conv(out)
        # final_out = torch.sigmoid(final_out)
        return final_out
    