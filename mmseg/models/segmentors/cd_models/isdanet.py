
'''
@ARTICLE{10879780,
  author={Ren, Hongjin and Xia, Min and Weng, Liguo and Lin, Haifeng and Huang, Junqing and Hu, Kai},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Interactive and Supervised Dual-Mode Attention Network for Remote Sensing Image Change Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-18},
  keywords={Feature extraction;Remote sensing;Accuracy;Semantics;Training;Decoding;Convolutional neural networks;Computational modeling;Spatiotemporal phenomena;Noise;Bitemporal feature interaction;change detection;deep supervision;multiscale fusion},
  doi={10.1109/TGRS.2025.3540864}}

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import MobileNetV2

class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Channel_Attention(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mip = max(8, in_ch // reduction)
        self.fc1 = nn.Conv2d(in_ch, mip, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(2*mip, in_ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        k = x
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        avg_ = self.relu1(self.fc1(avg))
        max_ = self.relu1(self.fc1(max))

        x_ch = torch.cat([avg_, max_], dim=1)

        out = self.fc2(x_ch)
        out = self.sigmoid(out)
        out = k * out


        return out



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordChannelAtt(nn.Module):
    def __init__(self, inp,  reduction=4):
        super(CoordChannelAtt, self).__init__()

        self.cha = Channel_Attention(inp, reduction=reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)

        self.conv1_ = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        cha = self.cha(x)

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv1_(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(x_h)  # CxHx1
        a_w = self.sigmoid(x_w)  # Cx1xW

        # a_h = self.conv_h(x_h).sigmoid()
        # a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        out = out + cha


        return out





class NFAM(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(NFAM, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d
        # scale 1
        self.conv_scale1_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[0], 32, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv_scale1_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv_s1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # scale 2
        self.conv_scale2_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], 64, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_s2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # scale 3
        self.conv_scale3_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], 128, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv_s3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1,stride=1,),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # scale 4
        self.conv_scale4_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[3], 256, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_s4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(nn.Conv2d(24,32,kernel_size=1))
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=1))
        self.conv3 = nn.Sequential(nn.Conv2d(96,128, kernel_size=1))
        self.conv4 = nn.Sequential(nn.Conv2d(320,256, kernel_size=1))


    def forward(self,c1, c2, c3, c4, c5):
        # scale 1
        c1_s1 = self.conv_scale1_c1(c1) # [4,32,64,64]
        c2_s1 = self.conv_scale1_c2(c2) # [4,32,64,64]

        c1_c2 = torch.cat([c1_s1,c2_s1],dim=1)
        s1 = self.conv_s1(c1_c2) #[4,32,64,64]
        c2_conv = self.conv1(c2)  # [4,32,64,64]
        s1 = self.relu(s1+c1_s1+c2_conv) # [4,32,64,64]

        # scale 2
        c2_s2 = self.conv_scale2_c2(c2) # [4,64,32,32]
        c3_s2 = self.conv_scale2_c3(c3) # [4,64,32,32]

        c2_c3 =  torch.cat([c2_s2,c3_s2],dim=1)
        s2 = self.conv_s2(c2_c3) #[4,64,32,32]
        c3_conv = self.conv2(c3) # [4,64,32,32]
        s2 = self.relu(s2+c2_s2+c3_conv) # [4,64,32,32]

        # scale 3
        c3_s3 = self.conv_scale3_c3(c3) #[4,128,16,16]
        c4_s3 = self.conv_scale3_c4(c4) #[4,128,16,16]

        c3_c4 = torch.cat([c3_s3,c4_s3],dim=1)
        s3 = self.conv_s3(c3_c4) #[4,128,16,16]
        c4_conv = self.conv3(c4) #[4，128，16，16]
        s3 = self.relu(s3+c3_s3+c4_conv)  #[4,128,16,16]



        # scale 4
        c4_s4 = self.conv_scale4_c4(c4) #[4,256,8,8]
        c5_s4 = self.conv_scale4_c5(c5) # [4,256,8,8]

        c4_c5 = torch.cat([c4_s4,c5_s4],dim=1)
        s4 = self.conv_s4(c4_c5) #[4,256,8,8]
        c5_conv =self.conv4(c5) #[4,256,8,8]
        s4 = self.relu(s4+c4_s4+c5_conv) #[4,256,8,8]

        return s1, s2, s3, s4  # [4,32,64,64]  # [4,64,32,32]  # [4,128,16,16]  # [4,256,8,8]




class CrossselfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, stride=1, padding=1, groups=in_channels // 8),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.query1_ = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.key1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, stride=1, padding=1, groups=in_channels // 8),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key1_ = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        # self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.value1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels , kernel_size=1),
            nn.BatchNorm2d(in_channels ),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels , in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels , in_channels , kernel_size=1),
            nn.BatchNorm2d(in_channels ),
            nn.ReLU(inplace=True)
        )

        self.query2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, stride=1, padding=1, groups=in_channels // 8),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.query2_ = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.key2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, stride=1, padding=1, groups=in_channels // 8),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key2_ = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        # self.value2 = nn.Conv2d(in_channels, in_channels , kernel_size = 1, stride = 1)
        self.value2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels , kernel_size=1),
            nn.BatchNorm2d(in_channels ),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels , in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels , in_channels , kernel_size=1),
            nn.BatchNorm2d(in_channels ),
            nn.ReLU(inplace=True)
        )

        self.softmax = nn.Softmax(dim = -1)


        self.conv1x1 = nn.Conv2d(2*in_channels, in_channels , kernel_size = 1, stride = 1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels , kernel_size=1),
            nn.BatchNorm2d(in_channels ),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels , in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels , in_channels , kernel_size=1),
            nn.BatchNorm2d(in_channels ),
            nn.ReLU(inplace=True)
        )

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1_ = self.key1_(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2)
        k2_ = self.key2_(input2).view(batch_size, -1, height * width)

        q = torch.cat([q1, q2], 1).view(batch_size, -1, height * width).permute(0, 2, 1)

        attn_matrix1_k = torch.bmm(q, k1_)
        attn_matrix1_k = self.softmax(attn_matrix1_k)

        attn_matrix2_k = torch.bmm(q, k2_)
        attn_matrix2_k = self.softmax(attn_matrix2_k)

        k1 = self.key1(input1)
        q1_= self.query1_(input1).view(batch_size, -1, height * width)

        k2 = self.key2(input2)
        q2_= self.query2_(input2).view(batch_size, -1, height * width)

        k = torch.cat([k1, k2], 1).view(batch_size, -1, height * width).permute(0, 2, 1)

        attn_matrix1_q = torch.bmm(k, q1_)
        attn_matrix1_q = self.softmax(attn_matrix1_q)

        attn_matrix2_q = torch.bmm(k, q2_)
        attn_matrix2_q = self.softmax(attn_matrix2_q)

        # attn_matrix1 = attn_matrix1_k + attn_matrix1_q
        # attn_matrix2 = attn_matrix2_k + attn_matrix2_q

        v1 = self.value1(input1).view(batch_size, -1, height * width)
        v2 = self.value1(input2).view(batch_size, -1, height * width)


        out1_k = torch.bmm(v1 ,attn_matrix1_k.permute(0, 2, 1))
        out1_q = torch.bmm(v1, attn_matrix1_q.permute(0, 2, 1))

        out1_k = out1_k.view(*input1.shape)
        out1_q = out1_q.view(*input1.shape)
        out1 = torch.cat([out1_k,out1_q],1)
        # out1 = out1.view(*input1.shape)
        out1 = self.conv1x1(out1)
        out1 = out1 + input1


        out2_k = torch.bmm(v2 ,attn_matrix2_k.permute(0, 2, 1))
        out2_q = torch.bmm(v2, attn_matrix2_q.permute(0, 2, 1))
        out2_k = out2_k.view(*input2.shape)
        out2_q = out2_q.view(*input2.shape)
        out2 = torch.cat([out2_k,out2_q],1)
        # out2 = out2.view(*input2.shape)
        out2 = self.conv1x1(out2)
        out2 = out2 + input2



        return out1, out2



class SupervisionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SupervisionAttentionModule, self).__init__()
        self.SA_Block = CoordChannelAtt(in_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_ = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.dwconv = nn.Sequential(

            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        # self.conv1_ = nn.Sequential(
        #     nn.Conv2d(2*in_channels, in_channels, kernel_size=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )


    def forward(self, x):
        mask_a = self.conv1(x)

        mask_a1 = self.dwconv(mask_a)

        mask_a_ = self.SA_Block(mask_a)

        mask_b_ = self.SA_Block(mask_a1)

        # mask_b = mask_a_ + mask_b_
        # mask_b = torch.cat([mask_a_,mask_b_],1)

        # mask_b1 =  self.conv1_(mask_b)

        mask = torch.cat([mask_a_,mask_b_],1)
        mask_ = self.conv1_(mask)

        mask_b1 = self.dwconv(mask_)

        mask_b2 = mask_a + mask_b1

        deep_flow = mask_b2

        out = self.conv1(mask_b2)


        return out, deep_flow




class ISDANet(nn.Module):
    def __init__(self,num_classes=2):
        super(ISDANet, self).__init__()

        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)

        # self.net = Backbone_resnet(backbone='resnet34')

        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2      #64
        self.swa = NFAM(channles, self.mid_d)

        """上采样"""
        self.up4 = nn.Sequential(
            Double_conv(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up3 = nn.Sequential(
            Double_conv(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up2 = nn.Sequential(
            Double_conv(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up1 = nn.Sequential(
            Double_conv(224, 112),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        # self.up4_ = nn.Sequential(
        #     Double_conv(128, 128),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )
        #
        # self.up3_ = nn.Sequential(
        #     Double_conv(64, 128),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )
        #
        # self.up1_ = nn.Sequential(
        #     Double_conv(32, 128),
        #     )


        self.sp1 = SupervisionAttentionModule(64)
        self.sp2 = SupervisionAttentionModule(128)
        self.sp3 = SupervisionAttentionModule(256)

        self.cross1 = CrossselfAttention(32)
        self.cross2 = CrossselfAttention(64)
        self.cross3 = CrossselfAttention(128)
        self.cross4 = CrossselfAttention(256)

        self.output_aux_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

        self.output_aux_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        self.output_aux_1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

        #
        # self.output =nn.Sequential(
        #
        #     nn.Conv2d(16,num_classes,kernel_size=3,stride=1,padding=1,bias=False),
        # )

        self.output = nn.Sequential(
            nn.Conv2d(112, 56, kernel_size=1,bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(56, num_classes, kernel_size=1, bias=False),
            # nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self,x1,x2):
        h, w = x1.shape[2:]
        ## x_layer0:[4,16,128,128] x_layer1:[4,24,64,64] x_layer2:[4,32,32,32] x_layer3:[4,96,16,16] x_layer4:[4,320,8,8]
        x1_layer0,x1_layer1,x1_layer2,x1_layer3,x1_layer4 = self.backbone(x1)
        x2_layer0,x2_layer1,x2_layer2,x2_layer3,x2_layer4 = self.backbone(x2)

        # [4,32,64,64]  # [4,64,32,32]  # [4,128,16,16]  # [4,256,8,8]
        x1_layer1, x1_layer2, x1_layer3, x1_layer4 = self.swa(x1_layer0,x1_layer1,x1_layer2,x1_layer3,x1_layer4)
        x2_layer1, x2_layer2, x2_layer3, x2_layer4 = self.swa(x2_layer0,x2_layer1,x2_layer2,x2_layer3,x2_layer4)

        inter1_a, inter1_b = self.cross1(x1_layer1 , x2_layer1)  #

        inter2_a, inter2_b = self.cross2(x1_layer2 , x2_layer2)  # [4,64,32,32]

        inter3_a, inter3_b = self.cross3(x1_layer3 , x2_layer3)  # [4,128,16,16]

        inter4_a, inter4_b = self.cross4(x1_layer4 , x2_layer4)  # [4,256,8,8]

        sub_layer1_ = inter1_a - inter1_b
        sub_layer2_ = inter2_a - inter2_b
        sub_layer3_ = inter3_a - inter3_b
        sub_layer4_ = inter4_a - inter4_b

        sp3, aux3 = self.sp3(sub_layer4_)  # [4,256,8,8]
        up4 = self.up4(sp3)   # [4,128,16,16]


        aux_3 = self.output_aux_3(aux3) #[4,2,8,8]

        add3 = sub_layer3_ + up4  # [4,128,16,16]
        sp2, aux2 = self.sp2(add3)      # [4,128,16,16]
        up3 = self.up3(sp2)       # [4,64,32,32]

        aux_2 = self.output_aux_2(aux2) #[4,2,16,16]

        add2 = sub_layer2_ + up3  # [4,64,32,32]
        sp1, aux1 = self.sp1(add2)      # [4,64,32,32]
        up2 = self.up2(sp1)       # [4,32,64,64]

        aux_1 =self.output_aux_1(aux1) #[4,2,32,32]

        add1 = sub_layer1_ + up2  # [4,32,64,64]

        # add1_ = self.up1_(add1) #[4,64,64,64]
        out = torch.cat([F.upsample(add3, add1.shape[2:], mode='bilinear', align_corners=True),
                         F.upsample(add2, add1.shape[2:], mode='bilinear', align_corners=True), add1], dim=1)  #[4,224,64,64]


        out= self.up1(out)       # [4,112,128,128]

        output = self.output(out)  # [4,2,128,128]

        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
        output1 = F.upsample(aux_1,size=(h, w), mode='bilinear', align_corners=True)
        output2 = F.upsample(aux_2,size=(h, w), mode='bilinear', align_corners=True)
        output3 = F.upsample(aux_3, size=(h, w), mode='bilinear', align_corners=True)

        return [output, output1, output2, output3]



if __name__ == '__main__':
    x1 = torch.rand(4,3,256,256)
    x2 = torch.rand(4,3,256,256)
    model = ISDANet()
    out, out1, out2, out3 = model(x1,x2)
    print('out:',out.shape,'out1:',out1.shape,'out2:',out2.shape,'out3:',out3.shape)