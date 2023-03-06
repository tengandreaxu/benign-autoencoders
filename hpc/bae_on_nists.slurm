#!/bin/bash -l

#SBATCH --array=0-20
#SBATCH --output=hpc/logs/bae_the two_nists.log
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=1G
#SBATCH --time 2:00:00

source ~/DimensionReductionScience/venv/bin/activate


python3 experiments/bae_on_the_two_nist.py \
    -n 0.0 -n 0.25 -n 0.5 -n 0.75 \
    -dw 0.5 -dw 0.9 \
    -nw 0.001 -nw 0.1 -nw  0.01 \
    -nu 128 \
    -lr 0.001 \
    -idx ${SLURM_ARRAY_TASK_ID} \
    -e 50 \
    -dir 'mnist_20runs' \
    -tp 0 -tp 1 -tp 2 -tp 3;



