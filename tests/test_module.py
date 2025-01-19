from efficientnet_pytorch import EfficientNet
from timm.models.efficientformer import LayerScale2d
import timm
import torch


model = timm.create_model('efficientnet_b6', pretrained=False)
# model_effi = EfficientNet.from_pretrained('efficientnet-b7')

xA = torch.rand(2, 3, 224, 224)
xB = torch.rand(2, 3, 224, 224)
interaction_layers = ['blocks']

for name, module in model.named_children():
    # print(f"Module Name: {name}")
    if name not in interaction_layers:
        xA = module(xA)
        xB = module(xB)
    else:
        xA_list = []
        xB_list = []
        for sub_name, sub_module in module.named_children():
            # print(f"Module Name: {name}, Submodule Name: {sub_name}")
            xA = sub_module(xA)
            xB = sub_module(xB)
            xA_list.append(xA)
            xB_list.append(xB)
        break

for item in xA_list:
    print(item.shape)