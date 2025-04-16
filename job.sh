#! /bin/bash

#SBATCH --job-name=DF
#SBATCH -o r_out%j.out
#SBATCH -e r_err%j.err

#SBATCH --mail-user=niting@email.sc.edu
#SBATCH --mail-type=ALL

#SBATCH -p v100-16gb-hiprio
#SBATCH --gres=gpu:1

module load python3/anaconda/2021.07 gcc/12.2.0 cuda/12.1
echo $CONDA_DEFAULT_ENV

hostname
uv run python3 code/main.py data/allrecipes-recipes outs/outs_5
