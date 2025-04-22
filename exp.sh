#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --error=exp.%J.err
#SBATCH --output=exp.%J。out
#SBATCH --mail-type=END
#SBATCH --mail-user=yuechuan_yang@brown.edu
python exp.py
