#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sc
#SBATCH --qos=scavenger
#SBATCH --output=./slurm_files/eval_%j.out
#SBATCH --error=./slurm_files/eval_%j.error
#SBATCH --array=0-50

export PATH=/pkgs/anaconda3/bin:$PATH
source $HOME/.zshrc
conda init bash

module load cuda-11.0

conda activate SE3nv
export HF_HOME=/projects/superdiff/model_checkpoints/huggingface_models

#LENGTH=$1
LENGTH=100
#KAPPA=$1
#python3 run_self_consistency.py inference.sample_num=$SLURM_ARRAY_TASK_ID inference.input_dir="../generated_proteins/rfdiff_ns0.1/length_${LENGTH}/model_outputs"
#
#python3 run_self_consistency.py inference.sample_num=$SLURM_ARRAY_TASK_ID inference.input_dir="../superdiff/generated_proteins/kappa_${KAPPA}_fse3only/length_100/model_outputs/"
python3 run_self_consistency.py inference.sample_num=$SLURM_ARRAY_TASK_ID inference.input_dir="../superdiff/generated_proteins/NEGROTKAPPA_composition_bothkappa_sde_500t_temp1_logp0/length_300/model_outputs"
#python3 run_self_consistency.py inference.sample_num=$SLURM_ARRAY_TASK_ID inference.input_dir="/projects/superdiff/generated_proteins/laz_runs_avg/kappa_0.3/length_100/model_outputs"
