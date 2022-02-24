#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run FPI with 5 different random seeds
python trainer.py --seed 0 --interp_fn fpi
python trainer.py --seed 1 --interp_fn fpi
python trainer.py --seed 2 --interp_fn fpi
python trainer.py --seed 3 --interp_fn fpi
python trainer.py --seed 4 --interp_fn fpi

# Run PII with 5 different random seeds
python trainer.py --seed 0 --interp_fn pii
python trainer.py --seed 1 --interp_fn pii
python trainer.py --seed 2 --interp_fn pii
python trainer.py --seed 3 --interp_fn pii
python trainer.py --seed 4 --interp_fn pii
