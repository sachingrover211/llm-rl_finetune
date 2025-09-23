#!/bin/bash

#SBATCH -N 1           # number of nodes
#SBATCH -c 4
#SBATCH -t 2-12:00:00   # time in d-hh:mm:ss
#SBATCH -G a100:1
#SBATCH --mem 80G
#SBATCH -p general
#SBATCH -q public
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=all

# Load required modules for job's environment
ml mamba
# Using python, so source activate an appropriate environment
source activate llm
# load cuda and nccl
ml cuda-12.4.1-gcc-12.1.0 nccl-2.22.3-1-gcc-12.1.0

export SCRATCH="/scratch/sgrover6"
export CODE_HOME="/home/sgrover6/avashis9/llm-rl_finetune"
export ACCELERATE_CONFIG="/home/sgrover6/src/accelerate_config_rl.yaml"
export CONFIG_PATH="$CODE_HOME/configs/hopper_numeric_oss_baseline.yaml"
export HF_HOME="$SCRATCH/.cache/huggingface/hub/"
export OPENAI_API_KEY=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
cd $CODE_HOME
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pwd
#echo $SLURM_JOB_ID
#echo $MASTER_ADDR
#echo $SLURM_JOB_NUM_NODES
#echo $SLURM_JOB_NODELIST
#echo $PATH
#scontrol show hostnames "$SLURM_JOB_NODELIST"
#scontrol getaddr "$MASTER_ADDR"

export CMD="--config_file $ACCELERATE_CONFIG main.py --config $CONFIG_PATH"

echo $CMD

accelerate launch $CMD
