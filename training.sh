#!/bin/bash

#SBATCH --job-name=<<job_name>
#SBATCH --nodelist=<<node_list>> 
#SBATCH --gres=gpu:A6000:4  
#SBATCH --time=0-48:00:00                     
#SBATCH --mem=50000MB                         
#SBATCH --cpus-per-task=4      

source PATH
source PATH
conda activate NAME

srun python training.py --path dataset/ --loss_weight 1.0 --loss_option yes --wandb_name NAME --score label --top_n 20 --checkpoints_dirpath PATH