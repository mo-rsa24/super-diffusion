#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sd
#SBATCH --qos=m2
#SBATCH --output=./slurm_files/inference_%j.out
#SBATCH --error=./slurm_files/inference_%j.error
##SBATCH --array=0-4
##SBATCH --array=1

#export PATH=/pkgs/anaconda3/bin:$PATH
#source $HOME/.zshrc
#conda init bash
#
#module load cuda-11.0
#
#conda activate SE3nv
#export HF_HOME=/projects/superdiff/model_checkpoints/huggingface_models

LENGTH=$1
python3 se3diff_experiments/inference_se3_diffusion.py inference.name="length_${LENGTH}/model_outputs" inference.samples.min_length=$LENGTH inference.samples.max_length=$LENGTH
