#!/bin/bash
#SBATCH --job-name=naca_optim
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --exclude=gpu-158
#SBATCH --output=optim_%j.out
#SBATCH --error=optim_%j.err

WORKDIR=$PWD

cd "$WORKDIR"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
nvidia-smi
echo ""

#python3 run_optimization.py --max-iter 4 --step-schedule 0.01 0.0025 0.0025 0.0025 --max-backtrack 5

python3 run_optimization.py --restart-iter 2 --max-iter 4 --step 0.0025

#python3 run_optimization.py --skip-baseline --max-iter 4 --step-schedule 0.01 0.005 0.002 0.001
#python3 run_optimization.py --restart-iter 1 --max-iter 4 --step-schedule 0.005 0.0025 0.001 0.0005 --max-backtrack 5
#python3 run_optimization.py --skip-baseline --max-iter 4 --step-schedule 0.01 0.002 0.001 0.0005 --max-backtrack 5

#python3 run_optimization.py --restart-iter 2 --max-iter 3 --step-schedule 0.002 0.001 0.0005 --max-backtrack 5


echo ""
echo "Job finished at: $(date)"
