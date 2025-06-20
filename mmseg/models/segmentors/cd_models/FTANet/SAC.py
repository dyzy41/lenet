import torch
import torch.nn as nn
import math


def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i+0.5)/L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)
def get_dct_weights(width,height,channel,fidx_u,fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i*c_part: (i+1)*c_part, t_x, t_y] = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    return dct_weights
class FCABlock(nn.Module):
    """
        FcaNet: Frequency Channel Attention Networks
        https://arxiv.org/pdf/2012.11879.pdf
    """
    def __init__(self, channel,width,height,fidx_u, fidx_v, m = 8):
        super(FCABlock, self).__init__()
        mid_channel = channel // m
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(width,height,channel,fidx_u,fidx_v))
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()

        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)
class SFCA(nn.Module):
    def __init__(self, in_channel,width,height):
        super(SFCA, self).__init__()
        fidx_u = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2]
        fidx_v = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2]
        fidx_u = [temp_u * (width // 8) for temp_u in fidx_u]
        fidx_v = [temp_v * (width // 8) for temp_v in fidx_v]
        self.FCA = FCABlock(in_channel, width, height, fidx_u, fidx_v)

    def forward(self, x):
        # FCA
        F_fca = self.FCA(x)

        return F_fca

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
    def forward(self, x):
        x1,x2=x,x

        B,C,H,W = x1.shape
        x1 = x1.transpose(-1, -3).reshape(B, -1, C)

        x1 = x1.view(B, H, W, C)
        x1 = x1.to(torch.float32)

        x1 = torch.fft.rfft2(x1, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)

        x1 = x1 * weight
        x1 = torch.fft.irfft2(x1, s=(H, W), dim=(1, 2), norm='ortho')
        x1 = x1.reshape(B,C,H,W )

        con = self.conv1(x2)
        con = self.norm(con)
        F_con = x2 * con

        return x1 + F_con
