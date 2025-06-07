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

## Experiments
COMPARISON RESULTS ON THE CLCD DATASET
| Model            | IoU   | F1    | Rec   | Prec  |
| ---------------- | ----- | ----- | ----- | ----- |
| ACABFNet [41]    | 51.45 | 67.94 | 63.63 | 72.88 |
| STANet [42]      | 51.49 | 67.97 | 64.16 | 72.26 |
| P2V [43]         | 54.10 | 70.22 | 65.93 | 75.11 |
| MSCANet [44]     | 55.83 | 71.65 | 67.07 | 76.91 |
| HATNet [45]      | 56.90 | 72.53 | 69.42 | 75.94 |
| BIT [46]         | 58.36 | 73.71 | 66.63 | 82.47 |
| DSIFN [24]       | 59.42 | 74.54 | 67.86 | 82.69 |
| MIN-Net [47]     | 62.08 | 76.60 | 75.70 | 77.53 |
| AMTNet [48]      | 62.35 | 76.81 | 75.06 | 78.64 |
| SAM-CD2 [49]     | 62.54 | 76.95 | 71.60 | 83.17 |
| CGNet [50]       | 62.67 | 77.05 | 71.71 | 83.25 |
| CACG-Net [51]    | 64.76 | 78.61 | 76.71 | 80.61 |
| EfficientCD [33] | 65.14 | 78.89 | 75.83 | 82.21 |
| LENet            | 66.83 | 80.12 | 77.09 | 83.39 |

<hr>
COMPARISON RESULTS ON THE LEVIR-CD DATASET

| Model               | IoU   |   F1   |  Rec   | Prec   |
|---------------------|-------|--------|--------|--------|
| STANet [35]         | 81.85 |  90.02 |  87.13 |  93.10 |
| ChangeFormer [52]   | 82.66 |  90.50 |  90.18 |  90.83 |
| Changer [25]        | --    |  92.06 |  90.56 |  93.61 |
| SSCD [37]           | 82.78 |  90.58 |  89.08 |  92.12 |
| CDMamba [53]        | 83.07 |  90.75 |  90.08 |  91.43 |
| DMATNet [27]        | 84.13 |  90.75 |  89.98 |  91.56 |
| GASNet [54]         | --    |  91.21 |  90.62 |  91.82 |
| Hybrid-MambaCD [55] | 84.31 |  91.48 |  90.78 |  92.20 |
| ACAHNet [56]        | 84.35 |  91.51 |  90.68 |  92.36 |
| HATNet [45]         | 84.41 |  91.55 |  90.23 |  92.90 |
| ConMamba [57]       | --    |  91.70 |  90.06 |  93.14 |
| FEMCD [58]          | --    |  92.02 |  90.88 |  93.18 |
| IMDCD [59]          | 84.66 |  91.34 |  91.12 |  91.56 |
| DED-SAM [60]        | 85.11 |  92.00 |  90.47 |  93.51 |
| PCAANet [61]        | 85.22 |  92.02 |  90.67 |  93.41 |
| MSA [62]            | 85.34 |  92.09 |  90.55 |  93.68 |
| HFIFNet [63]        | 85.46 |  92.16 |  90.09 |  93.37 |
| EfficientCD [33]    | 85.55 |  92.21 |  91.22 |  93.23 |
| SAM-CD2 [49]        | 85.59 |  92.24 |  90.93 |  93.58 |
| CACG-Net [51]       | 85.68 |  92.29 |  92.41 |  92.16 |
| CDNeXt [35]         | 85.86 |  92.39 |  90.92 |  93.91 |
| RSBuilding [64]     | 86.19 |  92.59 |  91.80 |  93.39 |
| LENet               | 86.30 |  92.64 |  91.22 |  94.12 |

<hr>
COMPARISON RESULTS ON THE S2LOOKING DATASET

| Model         |  IoU  |   F1   |  Rec   |  Prec  |
|---------------|-------|--------|--------|--------|
| BIT [46]      | 47.94 |  64.81 |  58.15 |  73.20 |
| HATNet [45]   | 47.08 |  64.02 |  60.90 |  67.48 |
| FHD [65]      | 47.33 |  64.25 |  56.71 |  74.09 |
| CGNet [50]    | 47.41 |  64.33 |  59.38 |  70.18 |
| SAM-CD [66]   | 48.29 |  65.13 |  58.92 |  72.80 |
| DMINet [26]   | 48.33 |  65.16 |  62.13 |  68.51 |
| PCAANet [61]  | 48.54 |  65.36 |  61.54 |  69.68 |
| HFIFNet [63]  | 48.54 |  65.35 |  61.04 |  70.33 |
| CDNeXt [35]   | 50.05 |  66.71 |  63.08 |  70.78 |
| Changer [25]  | 50.47 |  67.08 |  62.04 |  73.01 |
| LENet         | 51.19 |  67.71 |  61.90 |  74.72 |

<hr>
QUANTITATIVE RESULTS ON THE PX-CLCD DATASET

| Model           |  IoU  |   F1   |  Rec   |  Prec  |
|-----------------|-------|--------|--------|--------|
| HATNet [45]     | 88.99 |  94.18 |  93.83 |  94.53 |
| MSCANet [44]    | 89.00 |  94.18 |  93.95 |  94.41 |
| BIT [46]        | 90.78 |  95.17 |  94.80 |  95.54 |
| GASNet [54]     | 92.51 |  96.11 |  96.42 |  95.80 |
| DMINet [26]     | 92.83 |  96.28 |  96.31 |  96.25 |
| SNUNet3+ [67]   | 93.61 |  96.64 |  96.79 |  96.60 |
| CGNet [50]      | 93.82 |  96.81 |  97.33 |  96.30 |
| LENet           | 94.86 |  97.36 |  97.08 |  97.65 |

****

# Remote Sensing Change Detection Algorithms

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
[configs/rscd/c2fnet.py](https://ieeexplore.ieee.org/document/10445496)  


### Train command

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