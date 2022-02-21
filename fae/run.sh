#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run FAE model with 5 different random seeds
python train_fae.py --seed 0
python train_fae.py --seed 1
python train_fae.py --seed 2
python train_fae.py --seed 3
python train_fae.py --seed 4
