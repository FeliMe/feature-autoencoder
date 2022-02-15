#!/bin/bash

#SBATCH -J "DFR"   # job name
#SBATCH --output=DFR_run.out
#SBATCH --error=DFR_run.err
#SBATCH --mail-user=felix.meissen@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --time=24:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1  # gpus if needed

## Load python and anaconda
# ml python/anaconda3

## Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

## Run DFR with 5 different random seeds
python trainer.py --seed 0 --batch_size 8 --no_train --resume_path felix-meissen/feature_autoencoder/1l2wn7hb/last.pt --test_dataset wmh
python trainer.py --seed 1 --batch_size 8 --no_train --resume_path felix-meissen/feature_autoencoder/f8e2ptgz/last.pt --test_dataset wmh
python trainer.py --seed 2 --batch_size 8 --no_train --resume_path felix-meissen/feature_autoencoder/7icqlxws/last.pt --test_dataset wmh
python trainer.py --seed 3 --batch_size 8 --no_train --resume_path felix-meissen/feature_autoencoder/34trteqh/last.pt --test_dataset wmh
python trainer.py --seed 4 --batch_size 8 --no_train --resume_path felix-meissen/feature_autoencoder/1cfl4w9y/last.pt --test_dataset wmh
