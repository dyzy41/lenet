#!/usr/bin/env bash

# 函数：执行指定的命令，如果执行时间小于30分钟，则重试
run_command() {
    local cmd=$1
    local min_time=$2 # 最小时间，以秒为单位

    while true; do
        local start_time=$(date +%s)

        eval $cmd

	local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        if [ $duration -ge $min_time ]; then
            echo "命令执行成功，耗时 $(($duration / 60)) 分钟."
            break
        else
            echo "命令执行时间少于指定的 $(($min_time / 60)) 分钟，重试..."
            sleep 5  # 短暂休眠后重试
        fi
    done
}

# 使用 run_command 函数执行每个实验
# 参数：完整的命令字符串和最小执行时间（秒）
train_time=$((1 * 100))  # 10分钟
test_time=$((10))  # 20秒

# 使用 run_command 函数执行每个实验
# 参数：完整的命令字符串和重试间隔时间（秒）


CLCD=$CDPATH/CLCD
LEVIR=$CDPATH/LEVIR-CD
SYSU=$CDPATH/SYSU-CD
CDD=$CDPATH/ChangeDetectionDataset/Real/subset
S2Looking=$CDPATH/S2Looking
WHUCD=$CDPATH/WHUCD/cut_data
LEVIRPLUS=$CDPATH/LEVIR_CD_PLUS
PXCLCD=$CDPATH/PX-CLCD


# bash tools/dist_train.sh configs/lenet/lenet_levir.py 2 work_dirs/lenet_levir
# bash tools/dist_train.sh configs/lenet/lenet_clcd.py 2 work_dirs/lenet_clcd
# bash tools/dist_train.sh configs/lenet/lenet_pxclcd.py 2 work_dirs/lenet_pxclcd
# bash tools/dist_train.sh configs/lenet/lenet_s2looking.py 2 work_dirs/lenet_s2looking


# bash tools/test.sh LEVIR configs/lenet/lenet_levir.py 1 work_dirs/lenet_levir
# bash tools/test.sh CLCD configs/lenet/lenet_clcd.py 1 work_dirs/lenet_clcd
# bash tools/test.sh PXCLCD configs/lenet/lenet_pxclcd.py 1 work_dirs/lenet_pxclcd
# bash tools/test.sh S2Looking configs/lenet/lenet_s2looking.py 1 work_dirs/lenet_s2looking

