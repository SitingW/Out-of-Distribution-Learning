
#!/bin/bash
#SBATCH --account=def-szepesva
#SBATCH --nodes=1
#SBATCH --job-name=meow
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mail-user=siting3@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/siting/Out-of-Distribution-Learning/job/meow_%j.out
#SBATCH --error=/home/siting/Out-of-Distribution-Learning/job/meow_%j.err

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module purge # 
module load StdEnv/2023
module load python/3.13

# create local env
# cd ${SLURM_TMPDIR}
virtualenv ${SLURM_TMPDIR}/PYENV
source ${SLURM_TMPDIR}/PYENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch==2.7.1
pip install --no-index tqdm
pip install --no-index panada==3.13.0
pip install --no-index pytest==8.4.1
pip install --no-index h5py==3.13.0

# run python script
cd /home/siting/Out-of-Distribution-Learning/
python torch_modelling/main.py

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running