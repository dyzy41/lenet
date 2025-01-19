import torch
import numpy as np
import torch.nn.functional as F


def euclidean_distance(img1, img2):
    diff_squared = (img1 - img2) ** 2
    distances = torch.sqrt(torch.sum(diff_squared, dim=1)).unsqueeze(1)
    max_distance = torch.max(distances)
    map1 = F.sigmoid(distances / max_distance)
    map2 = F.sigmoid(1-torch.cosine_similarity(img1, img2, dim=1)).unsqueeze(1)
    return map1, map2

img1 = torch.rand((2, 256, 64, 64))  # 随机值张量
img2 = torch.zeros((2, 256, 64, 64))  # 全零张量
x = euclidean_distance(img1, img2)
print(x)