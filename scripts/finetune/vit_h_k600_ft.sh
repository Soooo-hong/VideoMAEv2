#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/home/shk00315/intelligent_robot/VideoMAEv2/outputs' #모델 저장할 경로 
DATA_PATH='/mnt/datasets/soohong/k600/skeleton/train' # 데이터셋 있는 장소 + 여기에 val.csv, train.csv 둘 다 넣으면 될 듯 
MODEL_PATH='/home/shk00315/intelligent_robot/VideoMAEv2/vit_g_hybrid_pt_1200e.pth' #finetuning model이나  학습 후 모델 경로로 지정하면될 듯 

# JOB_NAME=$1
# PARTITION=${PARTITION:-"video"}
# # 8 for 1 node, 16 for 2 node, etc.
# GPUS=${GPUS:-1}
# GPUS_PER_NODE=${GPUS_PER_NODE:-1}
# CPUS_PER_TASK=${CPUS_PER_TASK:-10}
# SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
# srun -p $PARTITION \
#         --job-name=${JOB_NAME} \
#         --gres=gpu:${GPUS_PER_NODE} \
#         --ntasks=${GPUS} \
#         --ntasks-per-node=${GPUS_PER_NODE} \
#         --cpus-per-task=${CPUS_PER_TASK} \
#         --kill-on-bad-exit=1 \
#         --quotatype=auto \
#         ${SRUN_ARGS} \
python infer_for_test.py \
        --model vit_huge_patch16_224 \
        --data_set Kinetics-10 \
        --nb_classes 10 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.2 \
        --head_drop_rate 0.5 \
        --layer_decay 0.8 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --warmup_epochs 5 \
        --epochs 40 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}
