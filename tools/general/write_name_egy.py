import os

dataset_path = "/home/ps/HDD/zhaoyq_data/CDdata/CD_Data_GZ/cut_data"  # 将此处替换为您的数据集路径
output_path = "/home/ps/HDD/zhaoyq_data/CDdata/CD_Data_GZ/cut_data"  # 将此处替换为您想要保存txt文件的输出路径

# 创建保存txt文件的文件夹
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 分别处理train、val和test文件夹
for folder_name in ["train", "val", "test"]:
    txt_path = os.path.join(output_path, f"{folder_name}.txt")
    names = open(os.path.join(dataset_path, 'EGY_list', f"{folder_name}.txt"), 'r').readlines()
    with open(txt_path, "w") as f:
        # 写入图片和标签路径到txt文件
        for name in names:
            image_pathA = os.path.join(dataset_path, 'A', name.strip())
            image_pathB = os.path.join(dataset_path, 'B', name.strip())
            label_path = os.path.join(dataset_path, 'label', name.strip())
            f.write(f"{image_pathA}  {image_pathB}  {label_path}\n")

print("所有txt文件生成完成！")

