#!/bin/bash
#SBATCH -J sol-one-hot
#SBATCH -o sol-one-hot_%j.out
#SBATCH -e sol-one-hot_%j.err
#SBATCH --mem=64gb
#SBATCH --time=480:00:00
#-----------------Módulos---------------------------
module load miniconda3
source activate ml_models

sh /home/dmedina/protsol/src/run_scripts/run_one_hot.sh