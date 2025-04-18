import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F

import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, pretrain, encoder_dim):
        super(ResNet18, self).__init__()

        resnet = models.resnet18(pretrained=pretrain)
        self.guided_feature_first_conv = nn.Sequential(*list(resnet.children())[:4]) # 1/4
        self.guided_feature_layers = nn.ModuleList()
        for i in range(3):
            self.guided_feature_layers.append(nn.Sequential(*list(resnet.children())[i+4]))

        self.in_channels = [64, 128, 256]
        self.encoder_dim = encoder_dim



    def forward(self, img):
        x = self.guided_feature_first_conv(img) # 1/4 conv bn relu pool;
        encoder_feat_list = []
        for i in range(3):
            x = self.guided_feature_layers[i](x) # 1/4 1/8 1/16
            encoder_feat_list.append(x)
        return encoder_feat_list[::-1] # 倒序

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))

class feature_decoder(nn.Module):
    def __init__(self, output_dim):
        super(feature_decoder, self).__init__()
        print(output_dim)
        self.toplayer = nn.Conv2d(256, output_dim, 1, 1, 0)  # 1*1卷积
 
        self.latlayer1 = nn.Conv2d(128, output_dim, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(64, output_dim, 1, 1, 0)

        self.smooth1 = nn.Sequential(
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1),
                                        nn.BatchNorm2d(output_dim))
        self.smooth2 = nn.Sequential(
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1),
                                        nn.BatchNorm2d(output_dim))
    def _upsample_add(self, x, y):
        _, _, H, W = y.shape  # b c h w
        return F.interpolate(x, size=(H, W), mode='bilinear',align_corners=True) + y
    def forward(self, x, is_right=False):

        p4 = self.toplayer(x[0])
        p3 = self._upsample_add(p4, self.latlayer1(x[1]))
        p2 = self._upsample_add(p3, self.latlayer2(x[2]))

        p4 = p4
        p3 = self.smooth1(p3)
        p2 = self.smooth2(p2)

        return [p4, p3, p2]

class multi_scale_encoder_decoder(nn.Module):
    def __init__(self, encoder_dim):
        super(multi_scale_encoder_decoder, self).__init__()
        self.encoder = ResNet18(pretrain=True, encoder_dim=encoder_dim)
        self.decoder = feature_decoder(output_dim=encoder_dim)
    
    def forward(self, x):
        enc_feat_list = self.encoder(x)
        return self.decoder(enc_feat_list)