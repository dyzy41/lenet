import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

'''
@ARTICLE{10530342,
  author={Wang, Zhongchen and Gu, Guowei and Xia, Min and Weng, Liguo and Hu, Kai},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Bitemporal Attention Sharing Network for Remote Sensing Image Change Detection}, 
  year={2024},
  volume={17},
  number={},
  pages={10368-10379},
  keywords={Feature extraction;Transformers;Remote sensing;Computational efficiency;Semantics;Earth;Decoding;Attention mechanism;convolutional neural network (CNN);change detection (CD);remote sensing (RS) images;transformer},
  doi={10.1109/JSTARS.2024.3400925}}

'''


class AttShare(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.key1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.key2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels// 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels// 4),
            nn.Sigmoid()
        )


    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape

        g = self.global_att(torch.cat([input1,input2],1)).view(batch_size, -1, 1 * 1)
        g = g.repeat(1, 1, height * width)

        q1 = self.query1(input1).view(batch_size, -1, height * width).permute(0, 2, 1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2).view(batch_size, -1, height * width).permute(0, 2, 1)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        global_att1 = torch.bmm(q1, g)

        attn_matrix1 = torch.bmm(q1, k1) + global_att1
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        global_att2 = torch.bmm(q2, g)
        attn_matrix2 = torch.bmm(q2, k2) + global_att2
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2
        return out1, out2


class CAG(nn.Module):
    def __init__(self, out_channels, channels):
        super(CAG, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1,
                padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1, stride=1,
                padding=0, bias=False),
        )

        self.left2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.right = nn.Sequential(
            nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=1,
                padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1, stride=1,
                padding=0, bias=False),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left(x_d)
        left2 = self.left2(x_d)
        right1 = self.right(x_s)
        right2 = self.right(x_s)
        right1 = F.interpolate(right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(right, size=dsize, mode='bilinear', align_corners=True)
        out = self.conv(torch.cat((right, left), dim=1))
        return out


class Decoder(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = nn.Sequential(
                   nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False),
                   nn.BatchNorm2d(c2),
                   nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.cv1(x)
        x = self.up(x)

        return x


class WF(nn.Module):

    def __init__(self, channels=64, channelr=128, r=4):
        super(WF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.de = Decoder(channelr, channels)

    def forward(self, x, residual):
        residual = self.de(residual)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)

        return xo

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class BASNet(nn.Module):
    def __init__(self, num_classes=2, normal_init=True):
        super(BASNet, self).__init__()

        #==================Backbone=========================#

        try:
            self.resnet = timm.create_model('resnet18', pretrained=True, pretrained_cfg_overlay=dict(file='pretrained/resnet50.a1_in1k/pytorch_model.bin'), features_only=True)
        except:
            self.resnet = timm.create_model('resnet18', pretrained=True, features_only=True)

        self.resnet.layer4 = nn.Identity()
        bc = 256  # the bottom_ch of the last layer

        #==================Attention layer==================#
        self.bas1 = AttShare(bc // 4)
        self.bas2 = AttShare(bc // 2)
        self.bas3 = AttShare(bc)

        #==================Fusion layer=====================#
        self.caga1 = CAG(bc // 4, bc // 2)
        self.caga2 = CAG(bc // 2, bc)
        self.cagb1 = CAG(bc // 4, bc // 2)
        self.cagb2 = CAG(bc // 2, bc)

        #==================Decoder==========================#
        self.wfa3 = WF(channels=bc // 2, channelr=bc)
        self.wfa2 = WF(channels=bc // 4, channelr=bc // 2)

        self.wfb3 = WF(channels=bc // 2, channelr=bc)
        self.wfb2 = WF(channels=bc // 4, channelr=bc // 2)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        )

        if normal_init:
            self.init_weights()

    def forward(self, a, b):
        #==================Backbone=========================#
        a0 = self.resnet.conv1(a)
        a0 = self.resnet.bn1(a0)
        a0 = self.resnet.act1(a0)
        a0 = self.resnet.maxpool(a0)
        a1 = self.resnet.layer1(a0)
        a2 = self.resnet.layer2(a1)
        a3 = self.resnet.layer3(a2)

        b0 = self.resnet.conv1(b)
        b0 = self.resnet.bn1(b0)
        b0 = self.resnet.act1(b0)
        b0 = self.resnet.maxpool(b0)
        b1 = self.resnet.layer1(b0)
        b2 = self.resnet.layer2(b1)
        b3 = self.resnet.layer3(b2)

        #==================Attention layer==================#
        a1c1, b1c1 = self.bas1(a1, b1)
        a2c2, b2c2 = self.bas2(a2, b2)
        a3c3, b3c3 = self.bas3(a3, b3)

        #==================Fusion layer=====================#
        af1 = self.caga1(a1c1, a2c2)
        af2 = self.caga2(a2c2, a3c3)
        bf1 = self.cagb1(b1c1, b2c2)
        bf2 = self.cagb2(b2c2, b3c3)

        # ==================Decoder=========================#
        """sub_branch"""
        sub1 = torch.abs(af1 - bf1)
        sub2 = torch.abs(af2 - bf2)
        sub3 = torch.abs(a3c3 - b3c3)

        """add_branch"""
        add1 = af1 + bf1
        add2 = af2 + bf2
        add3 = a3c3 + b3c3

        """WF_branch"""
        ad3 = self.wfa3(sub2, sub3)
        ad2 = self.wfa2(sub1, ad3)

        bd3 = self.wfb3(add2, add3)
        bd2 = self.wfb2(add1, bd3)

        cat = torch.cat([ad2, bd2], 1)
        out = self.upsamplex4(self.classifier(cat))

        return [out]

    def init_weights(self):
        self.bas1.apply(init_weights)
        self.bas2.apply(init_weights)
        self.bas3.apply(init_weights)

        self.caga1.apply(init_weights)
        self.caga2.apply(init_weights)
        self.cagb1.apply(init_weights)
        self.cagb2.apply(init_weights)

        self.wfa3.apply(init_weights)
        self.wfa2.apply(init_weights)
        self.wfb3.apply(init_weights)
        self.wfb2.apply(init_weights)

        self.classifier.apply(init_weights)


if __name__ == '__main__':
    x1 = torch.rand(1, 3, 256, 256)
    x2 = torch.rand(1, 3, 256, 256)

    model = BASNet(pretrained=False, normal_init=True)

    from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(x1, x2)
    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))
