#!/bin/bash
#SBATCH --account=def-mudl
#SBATCH --nodes 1              
#SBATCH --gres=gpu:1           
#SBATCH --mem=100G            
#SBATCH --time=0-15:00       

module load python/3.6
virtualenv --system-site-packages -p python3 env_ssl
source env_ssl/bin/activate
pip install -r requirements.txt

bash run_experiments.sh
