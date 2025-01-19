import torch
from torch import nn
from mmseg.registry import MODELS


@MODELS.register_module()
class BCLoss(nn.Module):
    """
    batch-balanced contrastive loss
    no-change, 1
    change, -1
    """
    def __init__(self, 
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='loss_bcl',
                 margin=2.0):
        super(BCLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, distance, label, weight=None, ignore_index=255):
        label = torch.argmax(label.unsqueeze(1), 1).unsqueeze(1).float()
        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return self.loss_weight*loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name