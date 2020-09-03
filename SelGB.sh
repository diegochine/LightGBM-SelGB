#!/usr/bin/env bash

#SBATCH --job-name=ltr-train.job
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=2

srun --ntasks=1 python3.6 /rhome/hpc20/867637/train.py "lgbm base" &
srun --ntasks=1 python3.6 /rhome/hpc20/867637/train.py "lgbm goss" &
srun --ntasks=1 python3.6 /rhome/hpc20/867637/train.py "fixed" &
srun --ntasks=1 python3.6 /rhome/hpc20/867637/train.py "false_positives" &
srun --ntasks=1 python3.6 /rhome/hpc20/867637/train.py "equal_size" &
wait