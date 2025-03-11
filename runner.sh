#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y
#$ -l rocky

# Set environment
module load miniforge
mamba activate Fairness

CUDA_VISIBLE_DEVICES='0' python main.py \
    --workers 8 \
    --pre-train-epochs 15 \
    --pre-train-lr 0.01 \
    --fine-tune-epochs 10 \
    --fine-tune-lr 0.06 \
    --batch-size 128 \
    --weight-decay 1e-4 \
    --print-freq 50 \
    --milestones 10 \
    --seeds 100 200 300 400 500 600 700 800 900 1000 \
    --job-id $JOB_ID \
    --target-attribute "Smiling" \
    --sensitive-attribute "Male" \
    --bias-degree "1/9" \
    --image-size 224 \
    --real-data-path "/data/EECS-IoannisLab/datasets/img_align_celeba/" \
    --synthetic-data-root "/data/scratch/acw717/synthetic_data/" \
    --number-train-data-real 20000 \
    --number-balanced-synthetic-0-0 5000 \
    --number-balanced-synthetic-0-1 5000 \
    --number-balanced-synthetic-1-0 5000 \
    --number-balanced-synthetic-1-1 5000 \
    --number-imbalanced-synthetic-0-0 1000 \
    --number-imbalanced-synthetic-0-1 9000 \
    --number-imbalanced-synthetic-1-0 9000 \
    --number-imbalanced-synthetic-1-1 1000 \
    --top-k 55 \
