import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops.einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
class Swish(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeDoubleConvOld(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SqueezeDoubleConvOld, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU())
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            )
        self.acfun = nn.GELU()

    def forward(self, x):
        x = self.squeeze(x)
        block_x = self.double_conv(x)
        x = self.acfun(x + block_x)
        return  x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, CAon=True, SAon=True, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.CAon = CAon
        self.SAon = SAon
        if self.CAon: 
            self.ca = ChannelAttention(in_planes, ratio)
        if self.SAon: 
            self.sa = SpatialAttention(kernel_size)

    def forward(self, x): 
        if self.CAon:
            x = self.ca(x) * x
        if self.SAon: 
            x = self.sa(x) * x
        return x

class NonLocal2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """
        super(NonLocal2D, self).__init__()

        assert dimension in [2,]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        # self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                      kernel_size=1, stride=1, padding=0)
        # self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)

        self.theta = nn.Sequential(
            # nn.BatchNorm2d(self.inter_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels)
        )
        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels)
            )


    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)#.permute(0, 2, 1)#compare with source github, this change more regular
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z
    
class SpatiotemporalAttentionFull(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """
        super(SpatiotemporalAttentionFull, self).__init__()
        assert dimension in [2,]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
                         )
                         
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0),
            )
        self.energy_time_1_sf = nn.Softmax(dim=-1)
        self.energy_time_2_sf = nn.Softmax(dim=-1)
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)
        
    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x12 = g_x11.permute(0, 2, 1)
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)
        g_x22 = g_x21.permute(0, 2, 1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        
        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        energy_time_1s = self.energy_time_1_sf(energy_time_1) 
        energy_time_2s = self.energy_time_2_sf(energy_time_2) 
        energy_space_2s = self.energy_space_2s_sf(energy_space_1) 
        energy_space_1s = self.energy_space_1s_sf(energy_space_2) 

        # energy_time_2s*g_x11*energy_space_2s = C2*S(C1) × C1*H1W1 × S(H1W1)*H2W2 = (C2*H2W2)' is rebuild C1*H1W1
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()#C2*H2W2
        # energy_time_1s*g_x12*energy_space_1s = C1*S(C2) × C2*H2W2 × S(H2W2)*H1W1 = (C1*H1W1)' is rebuild C2*H2W2
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W(y1), x2 + self.W(y2)

class SpatiotemporalAttentionBase(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """
        super(SpatiotemporalAttentionBase, self).__init__()
        assert dimension in [2,]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
                         )
                         
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0),
            )
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)
        
    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        
        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)

        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)
        energy_space_2s = self.energy_space_2s_sf(energy_space_1) # S(H1W1)*H2W2
        energy_space_1s = self.energy_space_1s_sf(energy_space_2) # S(H2W2)*H1W1

        # g_x11*energy_space_2s = C1*H1W1 × S(H1W1)*H2W2 = (C1*H2W2)' is rebuild C1*H1W1
        y1 = torch.matmul(g_x11, energy_space_2s).contiguous()#C2*H2W2
        # g_x21*energy_space_1s = C2*H2W2 × S(H2W2)*H1W1 = (C2*H1W1)' is rebuild C2*H2W2
        y2 = torch.matmul(g_x21, energy_space_1s).contiguous()
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W(y1), x2 + self.W(y2)

class SpatiotemporalAttentionFullNotWeightShared(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """
        super(SpatiotemporalAttentionFullNotWeightShared, self).__init__()
        assert dimension in [2,]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
                         )
        self.g2 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),
                         )
                         
        self.W1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.W2 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0),
            )
        
    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g1(x1).reshape(batch_size, self.inter_channels, -1)
        g_x12 = g_x11.permute(0, 2, 1)
        g_x21 = self.g2(x2).reshape(batch_size, self.inter_channels, -1)
        g_x22 = g_x21.permute(0, 2, 1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        
        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        energy_time_1s = F.softmax(energy_time_1, dim=-1) 
        energy_time_2s = F.softmax(energy_time_2, dim=-1) 
        energy_space_2s = F.softmax(energy_space_1, dim=-2) 
        energy_space_1s = F.softmax(energy_space_2, dim=-2) 
        #  C1*S(C2) energy_time_1s * C1*H1W1 g_x12 * energy_space_1s S(H2W2)*H1W1 -> C1*H1W1
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()#C2*H2W2
        #  C2*S(C1) energy_time_2s * C2*H2W2 g_x21 * energy_space_2s S(H1W1)*H2W2 -> C2*H2W2
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()#C1*H1W1
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W1(y1), x2 + self.W2(y2)

class DANetModule(nn.Module):
    def __init__(self, in_dim):
        super(DANetModule, self).__init__()
        self.chanel_in = in_dim
        self.sa = PAM_Module(in_dim)
        self.sc = CAM_Module(in_dim)

    def forward(self, x):
        sa_feat = self.sa(x)
        sc_feat = self.sc(x)
        return sa_feat + sc_feat


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1),
            nn.BatchNorm2d(in_dim//8),
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1),
            nn.BatchNorm2d(in_dim//8),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1),
            nn.BatchNorm2d(in_dim//8),
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1),
            nn.BatchNorm2d(in_dim//8),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, C, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = self.value_conv(x).view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out