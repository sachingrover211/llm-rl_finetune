#!/bin/bash

#SBATCH -N 1           # number of nodes
#SBATCH -c 4
#SBATCH -t 1-12:00:00   # time in d-hh:mm:ss
#SBATCH -G a100:1
#SBATCH --mem 40G
#SBATCH -p general
#SBATCH -q public
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=all

# Load required modules for job's environment
ml mamba
# Using python, so source activate an appropriate environment
source activate py313
# load cuda and nccl
ml cuda-12.4.1-gcc-12.1.0 nccl-2.22.3-1-gcc-12.1.0

export SCRATCH="/scratch/yzhou298"
export CODE_HOME="/home/yzhou298/avashis9/llm-rl_finetune"
export ACCELERATE_CONFIG="/home/yzhou298/accelerate_config_single_gpu.yaml"
export CONFIG_PATH="$CODE_HOME/configs/mc_cont_finetune_oss_baseline.yaml"
export HF_HOME="$SCRATCH/.cache/huggingface/hub/"
export OPENAI_API_KEY=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
cd $CODE_HOME
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

srun $CMD
