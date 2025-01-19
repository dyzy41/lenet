import os
import imageio
import tqdm
import numpy as np

dataset_path = "/home/jicredt_data/dsj/CDdata/BANDON"  # 将此处替换为您的数据集路径
output_path = "/home/jicredt_data/dsj/CDdata/BANDON"  # 将此处替换为您想要保存txt文件的输出路径

# 创建保存txt文件的文件夹
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 分别处理train、val和test文件夹
for folder_name in ["train", "val", "test", "test_ood"]:
    txt_path = os.path.join(output_path, f"{folder_name}.txt")

    label_folder = os.path.join(dataset_path, folder_name, "labels_unch0ch1ig255")


    label_files = sorted(os.listdir(label_folder))

    for label_file in tqdm.tqdm(label_files):
        if label_file.endswith('.png'):
            label_path = os.path.join(label_folder, label_file)
            lab = imageio.imread(label_path)
            lab = np.where(lab > 0, 255, 0)
            imageio.imwrite(label_path, lab.astype(np.uint8))

print("处理完成")

