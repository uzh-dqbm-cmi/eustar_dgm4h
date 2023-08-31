#!/bin/bash

#SBATCH --job-name=no_guid
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=3


#SBATCH --mem=50G

#SBATCH --gres=gpu:rtx3090:1
#SBATCH -o /cluster/work/medinfmk/EUSTAR2/logs/no_g2.out
#


export PYTHONPATH=$PYTHONPATH:/opt/code/install_dir/lib/python3.8/site-packages
export PATH="$HOME/.local/bin:$PATH"
source /cluster/work/medinfmk/EUSTAR2/envir/eustar/bin/activate
export PYTHONPATH=$PYTHONPATH:/opt/code/install_dir/lib/python3.8/site-packages

python3 -u /cluster/work/medinfmk/EUSTAR2/code_ml4h_ct/benchmark_VAE/src/pythae/scripts/cv.py