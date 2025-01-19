import yimage
import numpy as np
import os
import tqdm
import os.path as osp


def read_data(p_dict):
    p_imgA = p_dict[0]
    p_imgB = p_dict[1]
    p_lab = p_dict[2]

    imgA = yimage.io.read_image(p_imgA)
    imgB = yimage.io.read_image(p_imgB)
    lab = yimage.io.read_image(p_lab)
    # print(set(img.flatten()))
    return imgA, imgB, lab


def save_data(img_s, lab_s, p_img, p_lab):
    yimage.io.write_image(p_img, img_s)
    yimage.io.write_image(p_lab, lab_s)
    # cv2.imwrite(p_lab, lab_s)


def gen_dict():
    imgsA = os.listdir(osp.join(root_path, nameA))
    imgsB = os.listdir(osp.join(root_path, nameB))
    labs = os.listdir(osp.join(root_path, label))

    imgsA = sorted([osp.join(root_path, nameA, i) for i in imgsA])
    imgsB = sorted([osp.join(root_path, nameB, i) for i in imgsB])
    labs = sorted([osp.join(root_path, label, i) for i in labs])

    path_list = []
    for i in range(len(imgsA)):
        path_list.append([imgsA[i], imgsB[i], labs[i]])
    return path_list


def cut_data(cut_size, over_lap, save_dir):
    path_list = gen_dict()

    os.makedirs(osp.join(save_dir, nameA))
    os.makedirs(osp.join(save_dir, nameB))
    os.makedirs(osp.join(save_dir, label))

    for i in tqdm.tqdm(range(len(path_list))):
        img_name = os.path.basename(path_list[i][2]).split('.')[0]
        imgA, imgB, lab = read_data(path_list[i])
        h, w = lab.shape
        down, left = cut_size, cut_size
        h_new = ((h - cut_size) // (cut_size - over_lap) + 1) * (cut_size - over_lap) + cut_size
        h_pad = h_new - h
        w_new = ((w - cut_size) // (cut_size - over_lap) + 1) * (cut_size - over_lap) + cut_size
        w_pad = w_new - w

        pad_u = h_pad//2
        pad_d = h_pad-pad_u
        pad_l = w_pad//2
        pad_r = w_pad-pad_l

        lab = np.pad(lab, ((pad_u, pad_d), (pad_l, pad_r)), 'reflect')
        imgA = np.pad(imgA, ((pad_u, pad_d), (pad_l, pad_r), (0, 0)), 'reflect')
        imgB = np.pad(imgB, ((pad_u, pad_d), (pad_l, pad_r), (0, 0)), 'reflect')

        ni = 0
        while left <= w_new:
            slice_imgA = imgA[:, left - cut_size:left, :]
            slice_imgB = imgB[:, left - cut_size:left, :]
            slice_lab = lab[:, left - cut_size:left]
            ni += 1
            nj = 0
            while down <= h_new:
                img_sA = slice_imgA[down - cut_size:down, :, :]
                img_sB = slice_imgB[down - cut_size:down, :, :]
                lab_s = slice_lab[down - cut_size:down, :]

                nj += 1
                yimage.io.write_image(osp.join(save_dir, nameA, '{}_{}_{}.{}'.format(img_name, ni, nj, lab_suffix)), img_sA)
                yimage.io.write_image(osp.join(save_dir, nameB, '{}_{}_{}.{}'.format(img_name, ni, nj, lab_suffix)), img_sB)
                yimage.io.write_image(osp.join(save_dir, label, '{}_{}_{}.{}'.format(img_name, ni, nj, lab_suffix)), lab_s)

                down = down + cut_size - over_lap
            down = cut_size
            left = left + cut_size - over_lap

    print('finished data cutting')


if __name__ == '__main__':
    root_path = '/home/user/dsj_files/CDdata/S2Looking'
    nameA = 'T1'
    nameB = 'T2'
    label = 'labels_change'

    cut_size = 256
    over_lap = 0
    save_dir = os.path.join(root_path, 'cut_data')

    img_suffix = 'tif'
    lab_suffix = 'png'
    cut_data(cut_size, over_lap, save_dir)
