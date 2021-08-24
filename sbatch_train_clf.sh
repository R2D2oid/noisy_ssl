#!/bin/bash
#SBATCH --account=def-mudl
#SBATCH --nodes 1              
#SBATCH --gres=gpu:1           
#SBATCH --mem=30G            
#SBATCH --time=0-12:00       

noise_type="$1"         #"sym"
noise_rate="$2"         #"0.1"
pretrain_type="$3"      # "barlowtwins" "moco"

bash run_experiments_clf.sh $noise_type $noise_rate $pretrain_type
