import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import MobileNetV2
from TSINN import Trans, INN
from SAC import kernel_size, SFCA, SpectralGatingNetwork
from LOG import  log_feature



class FeatureFusionModule(nn.Module):
    def __init__(self, id_d, out_d,
                 dim=64,
                 heads=[16, 16, 16],
                 ):
        super(FeatureFusionModule, self).__init__()

        self.id_d = id_d
        self.out_d = out_d

        self.relu = nn.ReLU(inplace=True)

        self.baseFeature = Trans(dim=dim, num_heads=heads[2])
        self.detailFeature = INN()

    def forward(self, c_fuse):
        base_feature = self.baseFeature(c_fuse)
        detail_feature = self.detailFeature(c_fuse)

        return  base_feature,detail_feature


class FTFM(nn.Module):


    def __init__(self, in_channel,width,height):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)
        self.fca = SFCA(64,width,height)
        self.ffn = SpectralGatingNetwork(64,height,width//2+1)

    def forward(self, t1, t2, log=None, module_name=None,
                img_name=None):
        # channel part

        f1 = self.fca(t1)
        f2 = self.fca(t2)
        t1_channel_avg_pool = self.avg_pool(f1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(f1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(f2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(f2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        m1 = self.ffn(t1)
        m2 = self.ffn(t2)

        # spatial part
        t1_spatial_avg_pool = torch.mean(m1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(m1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(m2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(m2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add

        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w

        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        return fuse

class DGConv(nn.Module):
    def __init__(self, in_channel, out_channel, padding, dilation,kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size, padding=padding, dilation=dilation),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        return self.conv(conv_data)


class NeighborFeatureAggregation(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(NeighborFeatureAggregation, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = 64
        self.out_d = out_d
        # scale 2

        self.conv_scalec21 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s2 = FeatureFusionModule(self.in_d[0], self.out_d)
        # scale 3

        self.conv_scalec32 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

        self.conv_aggregation_s3 = FeatureFusionModule(self.in_d[1], self.out_d)
        # scale 4

        self.conv_scalec43 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

        self.conv_aggregation_s4 = FeatureFusionModule(self.in_d[2], self.out_d)
        # scale 5

        self.conv_scalec54 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s5 = FeatureFusionModule(self.in_d[3], self.out_d)

    def forward(self, c2, c3, c4, c5,log=True, module_name=None, img_name=None):
        s11, s12 = self.conv_aggregation_s2(self.conv_scalec21(c2))

        s21, s22 = self.conv_aggregation_s3(self.conv_scalec32(c3))

        s31, s32 = self.conv_aggregation_s4(self.conv_scalec43(c4))

        s41, s42 = self.conv_aggregation_s5(self.conv_scalec54(c5))

        log_list = [self.conv_scalec21(c2)]
        feature_name_list = ['1']
        log_feature(log_list=log_list, module_name=module_name,
                    feature_name_list=feature_name_list,
                    img_name=img_name, module_output=True)
        return s11, s12, s21, s22, s31, s32, s41, s42

class TemporalFeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d, width, height

                 ):
        super(TemporalFeatureFusionModule, self).__init__()

        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)

        self.dpfa = FTFM(64, width, height)

        self.rfa1 = DGConv(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.rfa3 = DGConv(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.rfa5 = DGConv(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=5, dilation=5)
        # self.fuse = Fuse(32)

    def forward(self, x1,x2, module_name=None, img_name=None):
        x = self.dpfa(x1,x2)
        x_out = x + self.rfa1(self.relu(x + self.rfa3(self.relu(x + self.rfa5(x)))))

        # log_list1 = [x_out]
        # feature_name_list1 = ['1111']
        # log_feature(log_list=log_list1, module_name=module_name,
        #             feature_name_list=feature_name_list1,
        #             img_name=img_name, module_output=True)
        return x_out, x

class TemporalFusionModule(nn.Module):
    def __init__(self, in_d=32, out_d=32, ):
        super(TemporalFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.tffm_x2 = TemporalFeatureFusionModule(64, 64,64,64)
        self.tffm_x3 = TemporalFeatureFusionModule(64, 64,32,32)
        self.tffm_x4 = TemporalFeatureFusionModule(64, 64,16,16)
        self.tffm_x5 = TemporalFeatureFusionModule(64, 64,8,8)

        self.sa = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.cls = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, s1,x1,s2,x2,s3,x3,s4,x4, module_name=None, img_name=None):
        # temporal fusion
        c2,m1 = self.tffm_x2(s1,x1)
        c3,m2 = self.tffm_x3(s2,x2)
        c4,m3 = self.tffm_x4(s3,x3)
        c5,m4 = self.tffm_x5(s4,x4)

        log_list = [m1]
        feature_name_list = ['3']
        log_feature(log_list=log_list, module_name=module_name,
                    feature_name_list=feature_name_list,
                    img_name=img_name, module_output=True)

        log_list1 = [c2]
        feature_name_list1 = ['4']
        log_feature(log_list=log_list1, module_name=module_name,
                    feature_name_list=feature_name_list1,
                    img_name=img_name, module_output=True)

        return c2, c3, c4, c5

class SSM(nn.Module):
    def __init__(self, mid_d):
        super(SSM, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f

        x_out = self.conv2(x.mul(self.conv_context(torch.cat([mask_f, mask_b], dim=1))))

        return x_out, mask

class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.sam_p5 = SSM(self.mid_d)
        self.sam_p4 = SSM(self.mid_d)
        self.sam_p3 = SSM(self.mid_d)
        self.conv_p4 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5, log=True, module_name=None, img_name=None):
        # high-level
        p5, mask_p5 = self.sam_p5(d5)
        p4 = self.conv_p4(d4 + F.interpolate(p5, scale_factor=(2, 2), mode='bilinear'))

        p4, mask_p4 = self.sam_p4(p4)
        p3 = self.conv_p3(d3 + F.interpolate(p4, scale_factor=(2, 2), mode='bilinear'))

        p3, mask_p3 = self.sam_p3(p3)
        p2 = self.conv_p2(d2 + F.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
        mask_p2 = self.cls(p2)

        log_list = [mask_p2]
        feature_name_list = ['5']
        # log_feature(log_list=log_list, module_name=module_name,
        #             feature_name_list=feature_name_list,
        #             img_name=img_name, module_output=True)

        return  mask_p2, mask_p3, mask_p4, mask_p5

class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(BaseNet, self).__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.swa = NeighborFeatureAggregation(channles, self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.en_d * 2)
        self.decoder = Decoder(self.en_d * 2)

    def forward(self, x1, x2, log=True, img_name=None, labels=None):

        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)

        s11, s12, s21, s22, s31, s32, s41, s42 = self.swa(x1_2, x1_3, x1_4, x1_5,module_name='11', img_name=img_name)
        x11, x12, x21, x22, x31, x32, x41, x42 = self.swa(x2_2, x2_3, x2_4, x2_5,module_name='22', img_name=img_name)

        c2, c3, c4, c5 = self.tfm(torch.abs(s11 - x11), torch.abs(s12 - x12), torch.abs(s21 - x21),
                                  torch.abs(s22 - x22), torch.abs(s31 - x31), torch.abs(s32 - x32),
                                  torch.abs(s41 - x41), torch.abs(s42 - x42), module_name='two', img_name=img_name)

        mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5,log=log, module_name='last', img_name=img_name)

        X1 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')

        mask_p2 = torch.sigmoid(X1)
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = F.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear')
        mask_p5 = torch.sigmoid(mask_p5)

        # log_feature(log_list=[X1], module_name='model',
        #             feature_name_list=['change_out'],
        #             img_name=img_name, module_output=False)

        return mask_p2, mask_p3, mask_p4, mask_p5
