#!/bin/bash -l

#SBATCH --job-name=txu-ipython
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:0:0
#SBATCH --mem=50G
#SBATCH --output=hpc/logs/jupyter-logs6.out
#SBATCH --cpus-per-task=20

module load gcc/8.4.0-cuda python/3.7.7 cuda/11.0.2 cudnn/8.1.1.33-11.2-linux-x64

ulimit -c 0
source venv/bin/activate
ipnport=$(shuf -i8000-9999 -n1)

jupyter-notebook --no-browser --port=${ipnport} --ip=$(hostname -i)
