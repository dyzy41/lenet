# [A Remote Sensing Image Change Detection Method Integrating Layer Exchange and Channel-Spatial Differences](https://ieeexplore.ieee.org/document/11024553)
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


# Remote Sensing Change Detection Algorithms

### We have provided training configuration files for some classic change detection algorithms. As follows:


| 配置文件                       | 模型名称                                                 | 期刊          | 时间 |
|--------------------------------|----------------------------------------------------------|---------------|------|
| `configs/rscd/afcf3d.py`       | [AFCF3D](https://ieeexplore.ieee.org/document/10221754)   | TGRS          | 2023 |
| `configs/rscd/bit.py`          | [BIT](https://ieeexplore.ieee.org/document/9491802)       | TGRS          | 2021 |
| `configs/rscd/cdnext.py`       | [CDNeXt](https://www.sciencedirect.com/science/article/pii/S1569843224001213) | JAG         | 2024 |
| `configs/rscd/cgnet.py`        | [CGNet](https://ieeexplore.ieee.org/document/10234560)    | JSTAR         | 2023 |
| `configs/rscd/darnet.py`       | [DARNet](https://ieeexplore.ieee.org/document/9734050)    | TGRS          | 2022 |
| `configs/rscd/dminet.py`       | [DMINet](https://ieeexplore.ieee.org/document/10034787)   | TGRS          | 2023 |
| `configs/rscd/elgcnet.py`      | [ELGCNet](https://ieeexplore.ieee.org/abstract/document/10423067) | TGRS    | 2024 |
| `configs/rscd/gasnet.py`       | [GASNet](https://www.sciencedirect.com/science/article/pii/S0924271623000849) | ISPRS       | 2023 |
| `configs/rscd/hanet.py`        | [HANet](https://ieeexplore.ieee.org/abstract/document/10093022) | JSTAR    | 2023 |
| `configs/rscd/hatnet.py`       | [HATNet](https://ieeexplore.ieee.org/document/10462583)   | TIM           | 2024 |
| `configs/rscd/hcgmnet.py`      | [HCGMNet](https://ieeexplore.ieee.org/document/10283341)  | IGARSS        | 2023 |
| `configs/rscd/isdanet.py`      | [ISDANet](https://ieeexplore.ieee.org/document/10879780)  | TGRS          | 2025 |
| `configs/rscd/lunet.py`        | [LU-Net](https://ieeexplore.ieee.org/document/9301184)    | GRSL          | 2020 |
| `configs/rscd/mscanet.py`      | [MSCANet](https://ieeexplore.ieee.org/document/9780164)   | JSTAR         | 2022 |
| `configs/rscd/p2v.py`          | [P2V](https://ieeexplore.ieee.org/document/9975266)       | TIP           | 2022 |
| `configs/rscd/rctnet.py`       | [RCTNet](https://ieeexplore.ieee.org/document/10687791)   | ICME          | 2024 |
| `configs/rscd/scratch_former.py` | [ScratchFormer](https://ieeexplore.ieee.org/document/10489990) | TGRS    | 2024 |
| `configs/rscd/stanet.py`       | [STANet](https://www.mdpi.com/2072-4292/12/10/1662)       | Remote Sensing| 2020 |
| `configs/rscd/strobstnet.py`   | [STRobustNet](https://ieeexplore.ieee.org/document/10879578) | TGRS      | 2025 |
| `configs/rscd/c2fnet.py`       | [C2FNet](https://ieeexplore.ieee.org/document/10445496)   | TGRS          | 2024 |
| `configs/rscd/ftanet.py`       | [FTANet](https://ieeexplore.ieee.org/abstract/document/10824909)   | JSTAR          | 2025 |


# Remote Sensing Change Detection Datasets

[SYSU-CD](https://pan.baidu.com/s/1g9t_PPPlYOjJN5J9F6tt2Q?pwd=rscd) | 
[LEVIR-CD](https://pan.baidu.com/s/1HcAsf5YgcxRjK-DbwLUK1A?pwd=rscd) | 
[PX-CLCD](https://pan.baidu.com/s/1IGYmsGfWGlOTsPR3P-WwHw?pwd=rscd) | 
[WaterCD](https://pan.baidu.com/s/1HcdXgC0A2Zpn8kHIUby0pQ?pwd=rscd) | 
[CDD](https://pan.baidu.com/s/1vh1Ztk8zLqCrtERh7xJt3Q?pwd=rscd) | 
[CLCD](https://pan.baidu.com/s/1_op60cPouU1cr_KkDk4SIg?pwd=rscd)


# Train command

```
python tools/train.py configs/rscd/bit.py
```

### The train command of our [LENet](https://ieeexplore.ieee.org/document/11024553) (Contains the complete training, validation and testing process).
```
bash tools/train.sh
```


Other command please refer the [mmsegmentation]([GitHub - open-mmlab/mmsegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark.](https://github.com/open-mmlab/mmsegmentation))

# Other Change Detection Projects, please refer [EfficientCD](https://github.com/dyzy41/mmrscd), [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP)

## Citation 

###  If you use this code for your research, please cite our papers.  

```
@Article{Dong_IeeeJSelTopApplEarthObsRemoteSens_2025_p1,
    author =   {Sijun Dong and Fangcheng Zuo and Geng Chen and Siming Fu and Xiaoliang Meng},
    title =    {{A Remote Sensing Image Change Detection Method Integrating Layer-Exchange and Channel-Spatial Differences}},
    journal =  {Ieee J. Sel, Top, Appl, Earth Obs. Remote. Sens.},
    year =     2025,
    pages =    {1--17},
    doi =      {10.1109/JSTARS.2025.3576831}  ,
}
```