#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sc
#SBATCH --qos=m4
#SBATCH --output=./slurm_files/eval_%j.out
#SBATCH --error=./slurm_files/eval_%j.error
#SBATCH --array=0-49
#SBATCH --exclude=gpu104,gpu094,gpu093,gpu036,gpu109,gpu073,gpu103

export PATH=/pkgs/anaconda3/bin:$PATH
source $HOME/.zshrc
conda init bash

module load cuda-11.0
#conda activate superdiff_proteins
conda activate SE3nv
export HF_HOME=/projects/superdiff/model_checkpoints/huggingface_models


SEED=$SLURM_ARRAY_TASK_ID
LENGTH=$1

# SuperDiff(OR), l=0
logp=0
temp=1
python3 inference.py inference.save_path="./generated_proteins/superdiff_sde_500t_OR_temp${temp}_logp${logp}/length_${LENGTH}/model_outputs/" inference.seed=$SEED inference.stochastic=True inference.sample_length=$LENGTH inference.temp_trans=$temp inference.logp_trans=$logp inference.temp_rots=$temp inference.logp_rots=$logp 

# SuperDiff(OR), l=1
logp=1
temp=1
python3 inference.py inference.save_path="./generated_proteins/superdiff_sde_500t_OR_temp${temp}_logp${logp}/length_${LENGTH}/model_outputs/" inference.seed=$SEED inference.stochastic=True inference.sample_length=$LENGTH inference.temp_trans=$temp inference.logp_trans=$logp inference.temp_rots=$temp inference.logp_rots=$logp 

## SuperDiff(AND)
logp=0
python3 inference.py inference.save_path="./generated_proteins/superdiff_sde_500t_AND_logp${logp}/length_${LENGTH}/model_outputs/" inference.seed=$SEED inference.stochastic=True inference.sample_length=$LENGTH inference.kappa_operator="AND" inference.logp_trans=$logp inference.logp_rots=$logp

## Avg. of outputs
KAPPA=0.5
python3 inference.py inference.save_path="./generated_proteins/mixing_${KAPPA}/length_${LENGTH}/model_outputs/" inference.seed=$SEED inference.mixing_method="mixture" inference.sample_length=$LENGTH inference.kappa=${KAPPA}

## Proteus baseline
python3 inference.py inference.save_path="./generated_proteins/proteus_sde_500t/length_${LENGTH}/model_outputs/" inference.seed=$SEED inference.stochastic=True inference.mixing_method="baseline_proteus" inference.sample_length=$LENGTH

## FrameDiff baseline
python3 inference.py inference.save_path="./generated_proteins/framediff_sde_500t/length_${LENGTH}/model_outputs/" inference.seed=$SLURM_ARRAY_TASK_ID inference.stochastic=True inference.mixing_method="baseline_framediff" inference.sample_length=$LENGTH

