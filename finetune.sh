#!/bin/bash

#SBATCH -N 1           # number of nodes
#SBATCH -c 4
#SBATCH -t 4-00:00:00   # time in d-hh:mm:ss
#SBATCH -G a100:1
#SBATCH --mem 80G
#SBATCH -p general
#SBATCH -q public
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=all

# Load required modules for job's environment
# module load mamba/latest
# Using python, so source activate an appropriate environment
source activate gym_domains
# load cuda and nccl
module load cuda-12.4.1-gcc-12.1.0 nccl-2.22.3-1-gcc-12.1.0

export SCRATCH="/scratch/sgrover6"
export CODE_HOME="/home/sgrover6/src/llm-q"
export LOGGING=$CODE_HOME"/logs/finetune/qwen2.5_14B_5_epoch"
export HF_HOME="$SCRATCH/.cache/huggingface/hub/"
export FORCE_TORCHRUN=1

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_USE_CUDA_DSA=1
unset LD_LIBRARY_PATH
echo "################### LD LIBRARY PATH -- CLB $CUDA_LAUNCH_BLOCKING"
echo $LD_LIBRARY_PATH
echo "################### LIBRARY PATH ENDS -- TUCD $TORCH_USE_CUDA_DSA"

cd $CODE_HOME
pwd
#echo $SLURM_JOB_ID
#echo $MASTER_ADDR
#echo $SLURM_JOB_NUM_NODES
#echo $SLURM_JOB_NODELIST
#echo $PATH
#scontrol show hostnames "$SLURM_JOB_NODELIST"
#scontrol getaddr "$MASTER_ADDR"

export CMD="python finetune.py"

echo $CMD

srun $CMD
