#!/bin/bash
#SBATCH -A cmsc848n-class
#SBATCH -J memconf
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH -t 03:00:00
#SBATCH -o %x-%j.out
# From ~/scratch/848n (one process per instance; startup costs minutes):
#   sbatch memconflict.sh --instance 0 --out results/pilot --arm probes extract summarize regmem --on-reject append retire
# 0.6B runs: sbatch --gpus=a100_1g.5gb:1 -t 00:15:00 memconflict.sh --model Qwen/Qwen3-0.6B [...]
module load python
source ~/scratch/848n/venv/bin/activate
# The python module prepends its own site-packages to PYTHONPATH, shadowing the venv's packages.
unset PYTHONPATH
export HF_HOME=~/scratch/848n/hf_cache HF_HUB_OFFLINE=1
cd ~/scratch/848n
nvidia-smi -L
python run_memconflict.py "$@"
