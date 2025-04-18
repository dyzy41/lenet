'''
@ARTICLE{10879578,
  author={Zhang, Hong and Teng, Yuhang and Li, Haojie and Wang, Zhihui},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={STRobustNet: Efficient Change Detection via Spatial–Temporal Robust Representations in Remote Sensing}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Feature extraction;Transformers;Robustness;Semantics;Aggregates;Training;Diffusion models;Computational modeling;Computational complexity;Buildings;Change detection (CD);feature confusion;spatial–temporal robust representation (STRobustNet);transformer},
  doi={10.1109/TGRS.2025.3540794}}

'''

import torch.nn as nn
from torch.nn import init
import torch
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

from .RRGM import RRGM

from .multi_scale_encoder_decoder import multi_scale_encoder_decoder
from einops import rearrange
import numpy as np
from .DRPH import *


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x) # 最后一层不relu
        return x
    

class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, *, warm_up=1, T_max=10, start_ratio=0.0):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0    # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:            
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 *\
                            (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)
            
        self.cur += 1
        return [lr for base_lr in self.base_lrs]


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'cos':
        scheduler = WarmupCosineLR(optimizer, lr_min=1e-5, lr_max= args.lr, warm_up=args.warm_up, T_max=args.max_epochs, start_ratio=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    print(args.net_G)
    if args.net_G == 'STR':
        net = STRobustNet(args)
    elif args.net_G == 'STR-fast':
        net = STRobustNet_fast(args)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)



class STRobustNet(nn.Module):
    def __init__(self,):
        super(STRobustNet, self).__init__()
        
        self.cfg = {}
        self.cfg['hidden_dim'] = 256
        self.cfg['num_queries'] = 100
        self.cfg['nheads'] = 8
        self.cfg['dropout'] = 0
        self.cfg['dim_feedforward'] = 2048
        self.cfg['dec_layers'] = 6
        print(self.cfg)
        self.vis_token = False
        self.encoder_decoder = multi_scale_encoder_decoder(encoder_dim=256)
        self.RRGM = RRGM(**self.cfg)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.DRPH = DRPH(self.cfg['num_queries'])

    def forward(self, A, B):
        ms_featA_list = self.encoder_decoder(A)
        ms_featB_list = self.encoder_decoder(B)
        img_featA = ms_featA_list[2].clone()
        img_featB = ms_featB_list[2].clone()

        token = self.RRGM(ms_featA_list, ms_featB_list)

        pred_A = torch.einsum("bqc,bchw->bqhw",token, img_featA)
        pred_B = torch.einsum("bqc,bchw->bqhw", token, img_featB)
        pred_A = torch.softmax(pred_A, dim=1)
        pred_B = torch.softmax(pred_B, dim=1)
        pred = self.DRPH(self.upsamplex4(torch.cat((pred_A, pred_B), dim=1)), A, B)
        return [pred]


class STRobustNet_fast(nn.Module):
    def __init__(self, cfg):
        super(STRobustNet_fast, self).__init__()
        # self.cfg = cfg
        # print(self.cfg)
        self.cfg = {}
        self.cfg['hidden_dim'] = cfg.hidden_dim
        self.cfg['num_queries'] = cfg.num_queries
        self.cfg['nheads'] = cfg.nhead
        self.cfg['dropout'] = cfg.drop_out
        self.cfg['dim_feedforward'] = cfg.dim_forward
        self.cfg['dec_layers'] = cfg.dec_layers
        print(self.cfg)
        self.vis_token = cfg.vis_token
        self.encoder_decoder = multi_scale_encoder_decoder(encoder_dim=cfg.hidden_dim)
        self.RRGM = RRGM(**self.cfg)

        self.to_pred = nn.Sequential(
            nn.Conv2d(cfg.num_queries*2, 32, stride=1, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, stride=1, kernel_size=1))
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

    
    def forward(self, A, B):
        ms_featA_list = self.encoder_decoder(A)
        ms_featB_list = self.encoder_decoder(B)
        img_featA = ms_featA_list[2].clone()
        img_featB = ms_featB_list[2].clone()

        token = self.RRGM(ms_featA_list, ms_featB_list)
        # bs, _, _, _ = img_featA.shape

        pred_A = torch.einsum("bqc,bchw->bqhw",token, img_featA)
        pred_B = torch.einsum("bqc,bchw->bqhw", token, img_featB)
        pred_A = torch.softmax(pred_A, dim=1)
        pred_B = torch.softmax(pred_B, dim=1)

        pred = self.to_pred(self.upsamplex4(torch.cat((pred_A, pred_B), dim=1)))

        if self.vis_token:
            return pred, pred_A, pred_B
        else:
            return pred