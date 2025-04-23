import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import random
import tqdm

def ensure_three_channels(img, tag=None):
    """确保图像有三个通道"""
    if len(img.shape) == 2:
        if tag == 'gt':
            img = np.stack((img, img, img), axis=-1)
        else:
            img = img * 255
            img = np.stack((img, img, img), axis=-1)
            
    elif len(img.shape) == 3 and img.shape[-1] == 4:
        # 如果图像是4通道的，裁剪掉最后一个通道
        img = img[:, :, :3]

    return img

def generate_result_image(ground_truth, prediction):
    ground_truth = ground_truth[:, :, 0].astype(bool)
    prediction = prediction[:, :, 0].astype(bool)
    
    result_image = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)

    # True Positives - White
    tp = (ground_truth == 1) & (prediction == 1)
    result_image[tp] = [255, 255, 255]

    # True Negatives - Black
    tn = (ground_truth == 0) & (prediction == 0)
    result_image[tn] = [0, 0, 0]

    # False Positives - Green
    fp = (ground_truth == 0) & (prediction == 1)
    result_image[fp] = [0, 255, 0]

    # False Negatives - Red
    fn = (ground_truth == 1) & (prediction == 0)
    result_image[fn] = [255, 0, 0]

    return result_image

def random_crop(img, top, left, crop_size):
    return img[top:top + crop_size, left:left + crop_size]

def plot_change_detection_results(img_paths, output_path, crop_size=256, num=2000):
    imgA_files = sorted(os.listdir(img_paths['imgA_path']))
    imgB_files = sorted(os.listdir(img_paths['imgB_path']))
    label_files = sorted(os.listdir(img_paths['label_path']))

    model_names = list(img_paths.keys())
    model_names.remove('imgA_path')
    model_names.remove('imgB_path')
    model_names.remove('label_path')

    model_files = {name: sorted(os.listdir(img_paths[name])) for name in model_names}

    num_images = len(imgA_files)
    num_models = len(model_names)
    spacer_width = 6  # 间隔的宽度

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in tqdm.tqdm(range(num)):
        random_index = random.randint(0, num_images - 1)
        
        imgA = ensure_three_channels(imread(os.path.join(img_paths['imgA_path'], imgA_files[random_index])))
        imgB = ensure_three_channels(imread(os.path.join(img_paths['imgB_path'], imgB_files[random_index])))
        label = ensure_three_channels(imread(os.path.join(img_paths['label_path'], label_files[random_index])), tag='gt')

        predictions = [ensure_three_channels(imread(os.path.join(img_paths[name], model_files[name][random_index]))) for name in model_names]

        h, w = imgA.shape[:2]
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)

        imgA_crop = random_crop(imgA, top, left, crop_size)
        imgB_crop = random_crop(imgB, top, left, crop_size)
        label_crop = random_crop(label, top, left, crop_size)
        prediction_crops = [random_crop(prediction, top, left, crop_size) for prediction in predictions]

        result_images = [generate_result_image(label_crop, prediction_crop) for prediction_crop in prediction_crops]

        images = [imgA_crop, imgB_crop, label_crop] + result_images

        # 获取图像的高度和宽度
        img_height, img_width = imgA_crop.shape[:2]

        # 计算拼接图像的总宽度
        total_width = img_width * (num_models + 3) + spacer_width * (num_models + 2)

        # 创建一个白色背景的空白图像
        combined_image = np.ones((img_height, total_width, 3), dtype=np.uint8) * 255

        # 将各图像拼接到白色背景图像上
        x_offset = 0
        for img in images:
            combined_image[:, x_offset:x_offset + img_width] = img
            x_offset += img_width + spacer_width

        # 保存拼接后的图像
        save_path = os.path.join(output_path, "{}_{}_{}.png".format(label_files[random_index].split('.')[0], top, left))
        plt.imsave(save_path, combined_image)

# 示例文件夹路径
# img_paths = {
#     'imgA_path': '/data1/smfu_data/LEVIR-CD/test/A',
#     'imgB_path': '/data1/smfu_data/LEVIR-CD/test/B',
#     'label_path': '/data1/smfu_data/LEVIR-CD/test/label',
#     'model1': 'dedd/levir/test_result',
#     'model2': 'levir/cdnext_LEVIR/test_result',
#     'model3': 'levir/gasnet_LEVIR/test_result',
#     'model4': 'levir/hatnet_LEVIR/test_result',
#     'model5': 'levir/stanet_LEVIR/test_result'
# }

# output_path = 'compare_levir_lenet_zfc'

# 示例文件夹路径
# img_paths = {
#     'imgA_path': '/home/user/dsj_files/CDdata/S2Looking/test/Image1',
#     'imgB_path': '/home/user/dsj_files/CDdata/S2Looking/test/Image2',
#     'label_path': '/home/user/dsj_files/CDdata/S2Looking/test/label',
#     'model1': 's2looking/lenet_S2Looking/test_result',
#     'model2': 's2looking/cdnext_S2Looking/test_result',
#     'model3': 's2looking/changer_S2Looking/tmp_infer_s2looking_20240627/vis_data/vis_image',
#     'model4': 's2looking/dminet_S2Looking/test_result',
#     'model5': 's2looking/bit_S2Looking/test_result'
# }

# output_path = 'compare_s2looking_lenet_zfc'

# 示例文件夹路径_fsm
# img_paths = {
#     'imgA_path': '/data1/smfu_data/PX-CLCD/test/A',
#     'imgB_path': '/data1/smfu_data/PX-CLCD/test/B',
#     'label_path': '/data1/smfu_data/PX-CLCD/test/label',
#     'model1': '/data2/dsj_files/lenet_experiment/dedd/pxclcd/test_result',
#     'model2': '/data2/dsj_files/wscdnet_experiment_fsm/pxclcd_work_dirs/dminet/test_result',
#     'model3': '/data2/dsj_files/wscdnet_experiment_fsm/pxclcd_work_dirs/gasnet/test_result',
#     'model4': '/data2/dsj_files/wscdnet_experiment_fsm/pxclcd_work_dirs/bit/test_result',
#     'model5': '/data2/dsj_files/wscdnet_experiment_fsm/pxclcd_work_dirs/hatnet/test_result',
# }
# img_paths = {
#     'imgA_path': '/data1/smfu_data/WHUCD/test/A/',
#     'imgB_path': '/data1/smfu_data/WHUCD/test/B/',
#     'label_path': '/data1/smfu_data/WHUCD/test/label/',
#     'model1': '/home/dongsj/fusiming/A_New_Start/work_dirs_3090/style_self_mul_diff_whucd_90.73/test_result',
#     'model2': 'whucd/cdnext/test_result',
#     'model3': 'whucd/hcgmnet/test_result',
#     'model4': 'whucd/bit/test_result',
#     'model5': 'whucd/cdnext/test_result',
# }

# img_paths = {
#     'imgA_path': '/data1/smfu_data/SYSU-CD/test/time1',
#     'imgB_path': '/data1/smfu_data/SYSU-CD/test/time2',
#     'label_path': '/data1/smfu_data/SYSU-CD/test/label',
#     'model1': '/data2/dsj_files/lenet_experiment/dedd/sysu/test_result',
#     'model2': 'sysu/afcf3d_SYSU/test_result',
#     'model3': 'sysu/hatnet_SYSU/test_result',
#     'model4': 'sysu/gasnet_SYSU/test_result',
#     'model5': 'sysu/stanet_SYSU/test_result',
# }

img_paths = {
    'imgA_path': '/data8T/DSJJ/CDdata/CDD/test/A',
    'imgB_path': '/data8T/DSJJ/CDdata/CDD/test/B',
    'label_path': '/data8T/DSJJ/CDdata/CDD/test/label/',
    'model1': '/home/sj/change_detection_compare_methods/CDD/DEED/test_result',
    'model2': '/home/sj/change_detection_compare_methods/CDD/ScratchFormer/test_result/',
    'model3': '/home/sj/change_detection_compare_methods/CDD/GASNet/test_result/',
    'model4': '/home/sj/change_detection_compare_methods/CDD/ELGCNet/test_result/',
    'model5': '/home/sj/change_detection_compare_methods/CDD/BASNet/test_result/',
}

output_path = '/home/sj/change_detection_compare_methods/CDD/compare_seed'

os.makedirs(output_path, exist_ok=True)

plot_change_detection_results(img_paths, output_path)