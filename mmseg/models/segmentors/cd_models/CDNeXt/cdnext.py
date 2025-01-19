from collections import OrderedDict
from .layers import *
import copy
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge, get_convnext
from .resnet import resnet18
import torch.nn.functional as F
from einops.einops import rearrange
__all__ = ['CDNeXt', 'get_cdnext',]

class CDNeXt(nn.Module):
    def __init__(self, backbone_scale="large", out_channels=2, isTemporalAttention=[1,2,3,4], isCBAM=[0,0,0,0], isNonlocal=[0,0,0,0]):
        super().__init__()
        #'tiny','small','base','large'
        self.encoder = get_convnext(pretrained=True, backbone_scale=backbone_scale, classes=2, in_22k=True, resolution=384)
        self.isFeatureFusion = True

        self.CAon = False
        self.SAon = False
        self.isCBAMconcat = False

        self.isNonlocalConcat = False
        self.AttentionModule = DANetModule#NonLocal2D    DANetModule

        self.isTemporalAttention = isTemporalAttention
        self.SpatiotemporalAttentionModule = SpatiotemporalAttentionFull #SpatiotemporalAttentionBase  SpatiotemporalAttentionFull  SpatiotemporalAttentionFullNotWeightShared
        self.isCBAM = isCBAM
        self.isNonlocal = isNonlocal
        self.encoderName = backbone_scale
        if "resnet" in self.encoderName:
            self.encoderNameScale = 2
        else:
            self.encoderNameScale = 4
        self.AdvSupResult = []
        self.downScale = [16, 8, 4, 0]
        self.stageNumber = 4
        self.backbone_depth = {
                                'tiny': [3, 3, 9, 3], 
                                'small': [3, 3, 27, 3],
                                'base': [3, 3, 27, 3], 
                                'large': [3, 3, 27, 3],
                                'xlarge': [3, 3, 27, 3],
                                "resnet18": [2, 2, 2, 2]
                            }
        self.backbone_dims = {
                                'tiny': [96, 192, 384, 768], 
                                'small': [96, 192, 384, 768],
                                'base': [128, 256, 512, 1024], 
                                'large': [192, 384, 768, 1536],
                                'xlarge': [256, 512, 1024, 2048],
                                "resnet18": [64, 128, 256, 512]
                            }
        self.size_dict = {
                            'tiny': [24, 96, 192, 384, 768], 
                            'small': [24, 96, 192, 384, 768],
                            'base': [32, 128, 256, 512, 1024], 
                            'large': [48, 192, 384, 768, 1536],
                            'xlarge': [64, 256, 512, 1024, 2048],
                            "resnet18": [32, 64, 128, 256, 512]
                        }
        # source constructure
        #init attention module
        self.CBAMs = []
        self.TemporalAttentions = []
        self.Nonlocals = []
        self.ChangeSqueezeConv = []
        # module sequence,  F.interpolate、TemporalAttention、AdversialSupervised、concat feature、conv、

        for index in range(self.stageNumber):
            if index == 0:

                if self.isCBAM[index] > 0:
                    if self.isCBAMconcat:
                        self.CBAMs.append(CBAM(self.n_channels*2, CAon=self.CAon, SAon=self.SAon))
                    else:
                        self.CBAMs.append(CBAM(self.n_channels, CAon=self.CAon, SAon=self.SAon))
                if self.isTemporalAttention[index] > 0:
                    self.TemporalAttentions.append(self.SpatiotemporalAttentionModule(self.size_change[index],))
                if self.isNonlocal[index] > 0:
                    if self.isNonlocalConcat:
                        self.Nonlocals.append(self.AttentionModule(self.n_channels*2))
                    else:
                        self.Nonlocals.append(self.AttentionModule(self.n_channels))
                self.ChangeSqueezeConv.append(SqueezeDoubleConvOld(self.n_channels*2, self.n_channels))
            else:
                if self.isCBAM[index] > 0:
                    if self.isCBAMconcat:
                        self.CBAMs.append(CBAM(self.size_change[index]*2, CAon=self.CAon, SAon=self.SAon))
                    else:
                        self.CBAMs.append(CBAM(self.size_change[index], CAon=self.CAon, SAon=self.SAon))
                if self.isTemporalAttention[index] > 0:
                    self.TemporalAttentions.append(self.SpatiotemporalAttentionModule(self.size_change[index],))
                if self.isNonlocal[index] > 0:
                    if self.isNonlocalConcat:
                        self.Nonlocals.append(self.AttentionModule(self.size_change[index]*2))
                    else:
                        self.Nonlocals.append(self.AttentionModule(self.size_change[index]))
                self.ChangeSqueezeConv.append(SqueezeDoubleConvOld(self.size_change[index]*4, self.size_change[index]))

        self.CBAMs = nn.ModuleList(self.CBAMs)
        self.TemporalAttentions = nn.ModuleList(self.TemporalAttentions)
        self.Nonlocals = nn.ModuleList(self.Nonlocals)
        self.ChangeSqueezeConv = nn.ModuleList(self.ChangeSqueezeConv)
        if self.isFeatureFusion == True:
            self.ChangeFinalSqueezeConv = SqueezeDoubleConvOld(sum(self.size_change[:-1]), self.size_change[-1]*self.encoderNameScale)
            self.ChangeFinalConv = nn.Sequential(SqueezeDoubleConvOld(self.size_change[-1]*self.encoderNameScale, self.size_change[-1]), 
                                                nn.Conv2d(self.size_change[-1], out_channels, kernel_size=1))
        else:
            self.ChangeFinalSqueezeConv = SqueezeDoubleConvOld(self.size_change[-2], self.size_change[-1]*self.encoderNameScale)
            # self.ChangeFinalSqueezeConv = SqueezeDoubleConvOld(self.size_change[-2], self.size_change[-1])
            self.ChangeFinalConv = nn.Sequential(SqueezeDoubleConvOld(self.size_change[-1]*self.encoderNameScale, self.size_change[-1]), 
                                                nn.Conv2d(self.size_change[-1], out_channels, kernel_size=1))
        # self.softmax = nn.Softmax(dim=1)
        self.register_hook(self.encoder)
        self.backboneFeatures = []

    def register_hook(self, backbone):
        if "resnet" in self.encoderName:
            def hook(module, input, output):
                self.backboneFeatures.append(output)
            depth = self.backbone_depth[self.encoderName]
            for index, depth_num in enumerate(depth):
                getattr(backbone, "layer"+str(index+1)).register_forward_hook(hook)
        else:
            def hook(module, input, output):
                self.backboneFeatures.append(output)
            depth = self.backbone_depth[self.encoderName]
            for index, depth_num in enumerate(depth):
                backbone.stages[index][depth_num-1].register_forward_hook(hook)

    @property
    def n_channels(self):
        return self.backbone_dims[self.encoderName][-1]

    @property
    def size_change(self):
        size_dict =  copy.deepcopy(self.size_dict)
        size_dict = size_dict[self.encoderName][::-1]
        return size_dict

    def forward(self, x1, x2):
        input_1 = x1
        input_2 = x2
        _ = self.encoder(x1)
        _ = self.encoder(x2)
        blocks1 = self.backboneFeatures[0:self.stageNumber]
        blocks2 = self.backboneFeatures[self.stageNumber:]
        self.backboneFeatures = []
        FusionFeatures = []
        change = None
        for stage in range(self.stageNumber):
            moduleIdx = stage
            eff_last_1 = blocks1.pop()#.permute(0, 3, 1, 2) 
            eff_last_2 = blocks2.pop()#.permute(0, 3, 1, 2)

            if self.isTemporalAttention[stage] > 0:
                moduleRealIdx = self.isTemporalAttention[stage] - 1
                eff_last_1, eff_last_2 = self.TemporalAttentions[moduleRealIdx](eff_last_1, eff_last_2)

            if self.isNonlocal[stage] > 0:
                moduleIdx = self.isNonlocal[stage] - 1
                if self.isNonlocalConcat:
                    eff_last = self.Nonlocals[moduleIdx](torch.cat([eff_last_1, eff_last_2], dim=1))
                    sliceNum = eff_last.shape[1]//2
                    eff_last_1, eff_last_2 = eff_last[:,0:sliceNum], eff_last[:,sliceNum:]
                else:
                    eff_last_1, eff_last_2 = self.Nonlocals[moduleIdx](eff_last_1), self.Nonlocals[moduleIdx](eff_last_2)

            if self.isCBAM[stage] > 0:
                moduleIdx = self.isCBAM[stage] - 1
                if self.isCBAMconcat:
                    eff_last = self.CBAMs[moduleIdx](torch.cat([eff_last_1, eff_last_2], dim=1))
                    sliceNum = eff_last.shape[1]//2
                    eff_last_1, eff_last_2 = eff_last[:,0:sliceNum], eff_last[:,sliceNum:]
                else:
                    eff_last_1, eff_last_2 = self.CBAMs[moduleIdx](eff_last_1), self.CBAMs[moduleIdx](eff_last_2)
            
            if stage == 0:
                change = torch.cat([eff_last_1, eff_last_2], dim=1)
            else:
                change = torch.cat([eff_last_1, eff_last_2, change], dim=1)

            if stage == self.stageNumber-1:
                change = self.ChangeSqueezeConv[stage](change)    
                FusionFeatures.append(change)
            else:
                change = self.ChangeSqueezeConv[stage](change)    
                FusionFeatures.append(change)
                change = F.interpolate(change, scale_factor=2., mode='bilinear', align_corners=True)
            
        if self.isFeatureFusion == True:
            for index, down in enumerate(self.downScale):
                FusionFeatures[index] = F.interpolate(FusionFeatures[index], scale_factor=2**(self.stageNumber-index-1), 
                                                        mode='bilinear', align_corners=True)
            fusion = torch.cat(FusionFeatures, dim=1)
        else:
            fusion = change

        change = self.ChangeFinalSqueezeConv(fusion)
        change = F.interpolate(change, scale_factor=self.encoderNameScale, mode='bilinear', align_corners=True)
        change = self.ChangeFinalConv(change)

        return [change]

def get_cdnext(out_channels=2, backbone_scale="tiny", pretrained=True, in_22k=True, resolution=384, backbone_trainable=False, **kwargs):

    encoder = get_convnext(pretrained=pretrained, backbone_scale=backbone_scale, classes=out_channels, in_22k=in_22k, resolution=resolution, **kwargs)
    model = CDNeXt(encoder, backbone_scale=backbone_scale, out_channels=out_channels, **kwargs)    

    if "resnet" in backbone_scale:
        pass
    else: 
        if backbone_trainable == False:
            for name, value in model.named_parameters():
                if "encoder" in name:
                    value.requires_grad = False

    return model