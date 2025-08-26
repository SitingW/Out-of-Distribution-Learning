#!/bin/bash
# SBATCH --account=def-someuser (what account I should use?)
#SBATCH --mem-per-cpu=1500M      # increase as needed
#SBATCH --time=1:00:00
#SBATCH --job-name=install_python_env
#SBATCH --output=job/install_python_env_%j.out
#SBATCH --error=job/install_python_env_%j.err

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r /home/siting/scratch/Out-of-Distribution-Learning/requirements.txt

python ...
