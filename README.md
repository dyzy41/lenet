# [A Remote Sensing Image Change Detection Method Integrating Layer Exchange and Channel-Spatial Differences](https://arxiv.org/abs/2501.10905)
# OPEN-RSCD Configuration Tutorial

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
### First, you can read the [environment.txt](environment.txt) and [environment.yml](environment.yml). If you install this env by yourself, please check the follow steps.

### Create a conda environment with python3.8 or above installed.

```bash
conda create --name mmrscd python=3.9
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


[configs/rscd/afcf3d.py](https://ieeexplore.ieee.org/document/10221754)  
[configs/rscd/bit.py](https://ieeexplore.ieee.org/document/9491802)  
[configs/rscd/cdnext.py](https://www.sciencedirect.com/science/article/pii/S1569843224001213)  
[configs/rscd/cgnet.py](https://ieeexplore.ieee.org/document/10234560?denied=)  
[configs/rscd/darnet.py](https://ieeexplore.ieee.org/document/9734050)  
[configs/rscd/dminet.py](https://ieeexplore.ieee.org/document/10034787)  
[configs/rscd/elgcnet.py](https://ieeexplore.ieee.org/abstract/document/10423067)  
[configs/rscd/gasnet.py](https://www.sciencedirect.com/science/article/pii/S0924271623000849)  
[configs/rscd/hanet.py](https://ieeexplore.ieee.org/abstract/document/10093022)  
[configs/rscd/hatnet.py](https://ieeexplore.ieee.org/document/10462583)  
[configs/rscd/hcgmnet.py](https://ieeexplore.ieee.org/document/10283341)  
[configs/rscd/isdanet.py](https://ieeexplore.ieee.org/document/10879780)  
[configs/rscd/lunet.py](https://ieeexplore.ieee.org/document/9301184)  
[configs/rscd/mscanet.py](https://ieeexplore.ieee.org/document/9780164)  
[configs/rscd/p2v.py](https://ieeexplore.ieee.org/document/9975266)  
[configs/rscd/rctnet.py](https://ieeexplore.ieee.org/document/10687791)  
[configs/rscd/scratch_former.py](https://ieeexplore.ieee.org/document/10489990)  
[configs/rscd/stanet.py](https://www.mdpi.com/2072-4292/12/10/1662)  
[configs/rscd/strobstnet.py](https://ieeexplore.ieee.org/document/10879578)  


### Train command

```
python tools/train.py configs/rscd/bit.py
```

### The train command of our [LENet](https://arxiv.org/abs/2501.10905) (Contains the complete training, validation and testing process).
```
bash tools/train.sh
```


Other command please refer the [mmsegmentation]([GitHub - open-mmlab/mmsegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark.](https://github.com/open-mmlab/mmsegmentation))

# Other Change Detection Projects, please refer [EfficientCD](https://github.com/dyzy41/mmrscd), [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP)

## Citation 

 If you use this code for your research, please cite our papers.  

```
@misc{dong2025remotesensingimagechange,
      title={A Remote Sensing Image Change Detection Method Integrating Layer Exchange and Channel-Spatial Differences}, 
      author={Sijun Dong and Fangcheng Zuo and Geng Chen and Siming Fu and Xiaoliang Meng},
      year={2025},
      eprint={2501.10905},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.10905}, 
}
```