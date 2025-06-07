#!/usr/bin/env bash


CLCD=$CDPATH/CLCD
LEVIR=$CDPATH/LEVIR-CD
SYSU=$CDPATH/SYSU-CD
CDD=$CDPATH/ChangeDetectionDataset/Real/subset
S2Looking=$CDPATH/S2Looking
WHUCD=$CDPATH/WHUCD/cut_data
LEVIRPLUS=$CDPATH/LEVIR_CD_PLUS
PXCLCD=$CDPATH/PX-CLCD


bash tools/dist_train.sh configs/rscd/c2fnet.py 2 --work-dir work_dirs/c2fnet
bash tools/test.sh LEVIR configs/rscd/c2fnet.py 1 work_dirs/c2fnet








