#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run AE model with MSE loss with 5 different random seeds
python trainer.py --loss_fn mse --seed 0
python trainer.py --loss_fn mse --seed 1
python trainer.py --loss_fn mse --seed 2
python trainer.py --loss_fn mse --seed 3
python trainer.py --loss_fn mse --seed 4

# Run AE model with SSIM loss with 5 different random seeds
python trainer.py --loss_fn ssim --seed 0
python trainer.py --loss_fn ssim --seed 1
python trainer.py --loss_fn ssim --seed 2
python trainer.py --loss_fn ssim --seed 3
python trainer.py --loss_fn ssim --seed 4