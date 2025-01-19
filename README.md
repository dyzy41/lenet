# MMRSCD Configuration Tutorial

## Data Prepared

In order to facilitate the use of relative paths, CDPATH is set in the ~/.bashrc file. Add the follow line in ~/.bashrc

```
export CDPATH="/data8T/DSJJ/CDdata"
```

After adding CDPATH as mentioned above, you can quickly navigate to the respective data path in the following way:

```bash
import os  
data_root = os.path.join(os.environ.get("CDPATH"), 'SYSU-CD')
```

***

### Take SYSU-CD dataset as an example, here introduce the usage of the code.

Use tools/general/write_path.py to generate a txt file for the dataset path. The format is as follows (for details, please refer to the code). The dataset function in this code reads the txt file to get the data list.

```bash
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/03414.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/03414.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/03414.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/00708.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/00708.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/00708.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/03907.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/03907.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/03907.png
```

***

# Environment

### Create a conda environment with python3.8 or above installed.

```bash
conda create --name mmrscd python=3.8
conda activate mmrscd
```

### Make sure you have mmcv>=2.1.0 installed, and make sure your torch version matches mmcv. You can find version matching information from the following linked documents.

### <https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html>

### For quick start, you can install them by the following command

```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install torch==2.1.0+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

***

### Run the follow lines to install the code.

```bash
git clone https://github.com/dyzy41/lenet
cd lenet
pip install -v -e .
```

***

### Please install the following dependencies in turn

#### gdal

```bash
conda install GDAL
```

#### ftfy, regex, einops, timm, kornia

```bash
pip install ftfy
pip install regex
pip install einops
pip install timm
pip install kornia
```

****

### We have provided training configuration files for some classic change detection algorithms. As follows:

```
configs/rscd/afcf3d.py
configs/rscd/bit.py
configs/rscd/cdnext.py
configs/rscd/cgnet.py
configs/rscd/dminet.py
configs/rscd/elgcnet.py
configs/rscd/gasnet.py
configs/rscd/hatnet.py
configs/rscd/mscanet.py
```

Train command

```
python tools/train.py configs/rscd/bit.py
```

Other command please refer the [mmsegmentation]([GitHub - open-mmlab/mmsegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark.](https://github.com/open-mmlab/mmsegmentation))
