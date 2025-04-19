#!/usr/bin/env bash


CLCD=$CDPATH/CLCD
LEVIR=$CDPATH/LEVIR-CD
SYSU=$CDPATH/SYSU-CD
CDD=$CDPATH/ChangeDetectionDataset/Real/subset
S2Looking=$CDPATH/S2Looking
WHUCD=$CDPATH/WHUCD/cut_data
LEVIRPLUS=$CDPATH/LEVIR_CD_PLUS
PXCLCD=$CDPATH/PX-CLCD


bash tools/dist_train.sh configs/lenet/lenet_levir.py 2 --work-dir work_dirs/lenet_levir
bash tools/test.sh LEVIR configs/lenet/lenet_levir.py 1 work_dirs/lenet_levir


bash tools/dist_train.sh configs/lenet/lenet_clcd.py 2 --work-dir work_dirs/lenet_clcd
bash tools/test.sh CLCD configs/lenet/lenet_clcd.py 1 work_dirs/lenet_clcd


bash tools/dist_train.sh configs/lenet/lenet_pxclcd.py 2 --work-dir work_dirs/lenet_pxclcd
bash tools/test.sh PXCLCD configs/lenet/lenet_pxclcd.py 1 work_dirs/lenet_pxclcd


bash tools/dist_train.sh configs/lenet/lenet_s2looking.py 2 --work-dir work_dirs/lenet_s2looking
bash tools/test.sh S2Looking configs/lenet/lenet_s2looking.py 1 work_dirs/lenet_s2looking






