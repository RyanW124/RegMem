#!/bin/bash
#SBATCH -A cmsc848n-class
#SBATCH -J memconf
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH -t 03:00:00
# Default is 4 GB host RAM; loading the 8B model needs more.
#SBATCH --mem=32G
#SBATCH -o %x-%j.out
# From ~/scratch/848n (one process per instance; startup costs minutes):
#   sbatch memconflict.sh --instance 0 --out results/final --arm probes extract summarize regmem trustmem --budget 2000
# As a job array, each task runs the instance equal to its array index (omit --instance):
#   sbatch --array=0-29 memconflict.sh --out results/final --arm probes extract --budget 1000 2000 4000 0
# 0.6B runs: sbatch --gpus=a100_1g.5gb:1 -t 00:15:00 memconflict.sh --model Qwen/Qwen3-0.6B [...]
module load python
source ~/scratch/848n/venv/bin/activate
# The python module prepends its own site-packages to PYTHONPATH, shadowing the venv's packages.
unset PYTHONPATH
export HF_HOME=~/scratch/848n/hf_cache HF_HUB_OFFLINE=1
# Long batched prompts fragment GPU memory; expandable segments avoid spurious out-of-memory errors.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/scratch/848n
nvidia-smi -L
python run_memconflict.py ${SLURM_ARRAY_TASK_ID:+--instance $SLURM_ARRAY_TASK_ID} "$@"
