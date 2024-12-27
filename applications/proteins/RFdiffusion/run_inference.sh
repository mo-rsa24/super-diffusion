#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sd
#SBATCH --qos=m3
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
#./scripts/run_inference.py "contigmap.contigs=[${LENGTH}-${LENGTH}]" inference.output_prefix="../generated_proteins/rfdiff_ns0.1/length_${LENGTH}/model_outputs/output" inference.num_designs=50
./scripts/run_inference.py "contigmap.contigs=[${LENGTH}-${LENGTH}]" inference.output_prefix="../generated_proteins/rfdiff_TEST/length_${LENGTH}/model_outputs/output" inference.num_designs=50
