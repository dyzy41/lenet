import torch
from utils.path_hyperparameter import ph
import cv2
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T


def normalize(tensor):
    """Normalize a tensor to range [0, 1]"""
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    return (tensor - tensor_min) / (tensor_max - tensor_min + 1e-5)  # Adding small epsilon to avoid division by zero

def log_feature(log_list, module_name, feature_name_list, img_name, module_output=True, labels=None):
    """ Log output feature of module and model

    Log some output features in a module. Feature in :obj:`log_list` should have corresponding name
    in :obj=`feature_name_list`.

    For module output feature, interpolate it to :math=`ph.patch_size`×:math=`ph.patch_size`,
    log it in :obj=`cv2.COLORMAP_JET` format without other change,
    and log it in :obj=`cv2.COLORMAP_JET` format with equalization.
    For model output feature, log it without any change.

    Notice that feature is log in :obj=`ph.log_path`/:obj=`module_name`/
    name in :obj=`feature_name_list`/:obj=`img_name`.jpg.

    Parameter:
        log_list(list): list of output need to log.
        module_name(str): name of module which output the feature we log,
            if :obj=`module_output` is False, :obj=`module_name` equals to `model`.
        module_output(bool): determine whether output is from module or model.
        feature_name_list(list): name of output feature.
        img_name(str): name of corresponding image to output.


    """
    for k, log in enumerate(log_list):
        log = log.clone().detach()
        b, c, h, w = log.size()
        print(f"Original log shape: {log.shape}, min: {log.min().item()}, max: {log.max().item()}")  # Debug info

        if module_output:
            log = torch.mean(log, dim=1, keepdim=True)
            print(f"After mean reduction: {log.shape}, min: {log.min().item()}, max: {log.max().item()}")  # Debug info

            log = F.interpolate(log, size=(ph.patch_size, ph.patch_size), mode='nearest')
            log = normalize(log)  # Normalize to [0, 1]
            log = log * 255  # Scale to [0, 255]
            log = log.reshape(b, ph.patch_size, ph.patch_size, 1).cpu().numpy().astype(np.uint8)
            print(f"After reshaping: {log.shape}, min: {log.min()}, max: {log.max()}")  # Debug info

            log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_equalize_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '_equalize/'
            Path(log_equalize_dir).mkdir(parents=True, exist_ok=True)

            for i in range(b):
                log_i = cv2.applyColorMap(log[i], cv2.COLORMAP_JET)
                cv2.imwrite(log_dir + img_name[i] + '.jpg', log_i)

                log_i_equalize = cv2.equalizeHist(log[i])
                log_i_equalize = cv2.applyColorMap(log_i_equalize, cv2.COLORMAP_JET)
                cv2.imwrite(log_equalize_dir + img_name[i] + '.jpg', log_i_equalize)
        else:
                log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                log = torch.round(torch.sigmoid(log))
                log = F.interpolate(log, scale_factor=ph.patch_size // h,
                                    mode='nearest').cpu()
                to_pil_img = T.ToPILImage(mode=None)
                for i in range(b):
                    log_i = to_pil_img(log[i])
                    log_i.save(log_dir + img_name[i] + '.jpg')