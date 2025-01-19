import os
import argparse
import cv2
import numpy as np


def vis(path):
    files = os.listdir(path)
    save = path+'_vis'
    os.makedirs(save, exist_ok=True)
    for file in files:
        img = cv2.imread(os.path.join(path, file), 0)
        img = img*255
        cv2.imwrite(os.path.join(save, file), img.astype(np.uint8))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='参数')
    parser.add_argument('--pppred', type=str, default='/home/dsj/0code_hub/cd_code/SegNeXt-main/work_dirs/sem_fpn_p2t_b_levir_80k_cdsa/test_result_big')
    args = parser.parse_args()

    x = vis(args.pppred)