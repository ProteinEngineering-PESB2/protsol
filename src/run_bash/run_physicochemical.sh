#!/bin/bash
#SBATCH -J sol-prop
#SBATCH -o sol-prop_%j.out
#SBATCH -e sol-prop_%j.err
#SBATCH --mem=64gb
#SBATCH --time=480:00:00
#-----------------Módulos---------------------------
module load miniconda3
source activate ml_models

sh /home/dmedina/protsol/src/run_scripts/run_properties_fft_models.sh