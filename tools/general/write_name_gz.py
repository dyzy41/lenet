import os
import random

dataset_path = "/home/ps/HDD/zhaoyq_data/CDdata/CD_Data_GZ/cut_data"  # 将此处替换为您的数据集路径
output_path = "/home/ps/HDD/zhaoyq_data/CDdata/CD_Data_GZ/cut_data"  # 将此处替换为您想要保存txt文件的输出路径

# 创建保存txt文件的文件夹
if not os.path.exists(output_path):
    os.makedirs(output_path)

files = os.listdir(os.path.join(dataset_path, 'T1'))
random.shuffle(files)
train_files = files[:int(0.8 * len(files))]
val_files = files[int(0.8 * len(files)):int(0.9 * len(files))]
test_files = files[int(0.9 * len(files)):]

# 分别处理train、val和test文件夹

with open(os.path.join(dataset_path, 'train.txt'), 'w') as f:
    for item in train_files:
        image_pathA = os.path.join(dataset_path, 'T1', item)
        image_pathB = os.path.join(dataset_path, 'T2', item)
        label_path = os.path.join(dataset_path, 'labels_change', item)
        assert os.path.exists(image_pathA) and os.path.exists(image_pathB) and os.path.exists(label_path)
        f.write(f"{image_pathA}  {image_pathB}  {label_path}\n")

with open(os.path.join(dataset_path, 'val.txt'), 'w') as f:
    for item in val_files:
        image_pathA = os.path.join(dataset_path, 'T1', item)
        image_pathB = os.path.join(dataset_path, 'T2', item)
        label_path = os.path.join(dataset_path, 'labels_change', item)
        assert os.path.exists(image_pathA) and os.path.exists(image_pathB) and os.path.exists(label_path)
        f.write(f"{image_pathA}  {image_pathB}  {label_path}\n")

with open(os.path.join(dataset_path, 'test.txt'), 'w') as f:
    for item in test_files:
        image_pathA = os.path.join(dataset_path, 'T1', item)
        image_pathB = os.path.join(dataset_path, 'T2', item)
        label_path = os.path.join(dataset_path, 'labels_change', item)
        assert os.path.exists(image_pathA) and os.path.exists(image_pathB) and os.path.exists(label_path)
        f.write(f"{image_pathA}  {image_pathB}  {label_path}\n")

print("所有txt文件生成完成！")

