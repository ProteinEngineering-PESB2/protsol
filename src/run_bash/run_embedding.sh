#!/bin/bash
#SBATCH -J solubility
#SBATCH -o solubility_%j.out
#SBATCH -e solubility_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=127gb

#-----------------Módulos---------------------------
module load miniconda3
source activate encoder_proteins

sh /home/dmedina/protsol/src/run_scripts/run_embedding.sh