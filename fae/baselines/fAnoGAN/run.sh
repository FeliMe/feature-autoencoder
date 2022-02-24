#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run f-AnoGAN with 5 different random seeds
python trainer.py --seed 0 --no_train
python trainer.py --seed 1 --no_train
python trainer.py --seed 2 --no_train
python trainer.py --seed 3 --no_train
python trainer.py --seed 4 --no_train
