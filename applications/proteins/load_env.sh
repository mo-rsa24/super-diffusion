
module load cuda-11.0

conda activate SE3nv
export HF_HOME=/projects/superdiff/model_checkpoints/huggingface_models 
export PATH=/projects/superdiff/foldseek/bin/:$PATH
export MODELSCOPE_CACHE=/projects/superdiff/model_checkpoints/modelscope/

source ~/.bashrc
zsh

echo "Done loading env :)"
