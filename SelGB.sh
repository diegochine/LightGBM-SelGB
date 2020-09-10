#!/usr/bin/env bash

#SBATCH --job-name=selgb
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=400000


srun --ntasks=1 --job-name=fixed python3.6 /rhome/hpc20/867637/train.py "fixed" &
srun --ntasks=1 --job-name=falpos python3.6 /rhome/hpc20/867637/train.py "false_positives" &
srun --ntasks=1 --job-name=eqsize python3.6 /rhome/hpc20/867637/train.py "equal_size" &
wait