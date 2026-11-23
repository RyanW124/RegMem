#!/usr/bin/env bash
#SBATCH -A cmsc848n-class
#SBATCH -J regmem-grpo
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
root=${RUN_DIR:-runs/grpo}
phase=${1:?Choose submit, prepare, proposal, mine, repair, or transitions}
# Use the caller's activated environment, inherited by each Slurm job.
python -c 'import sys; sys.exit("GRPO requires Python 3.11+; activate your environment before submitting.") if sys.version_info < (3, 11) else None'
if [[ $phase == submit ]]; then
    exec python grpo_submit.py "$root" "$0"
fi

model=${REGMEM_MODEL:-Qwen/Qwen3-8B}
data="$root/data"
case "$phase" in
    prepare) python -m regmem.grpo_data prepare --model "$model" --data-dir "$data" --local-parquet "${MEMALPHA_PARQUET:-data/memalpha/train.parquet}" ;;
    proposal|repair)
        shift
        python -m regmem.grpo_train --role "$phase" --model "$model" --data-dir "$data" --output-dir "$root/$phase" --resume-if-available "$@" ;;
    mine) python -m regmem.grpo_data mine-repair --model "$model" --data-dir "$data" --proposal-adapter "$root/proposal" ;;
    transitions)
        python evaluate_grpo.py --model "$model" --data-dir "$data" --out "$root/eval" \
            --proposal-adapter "$root/proposal" --repair-adapter "$root/repair" ;;
    *) echo "Unknown phase $phase" >&2; exit 2 ;;
esac
