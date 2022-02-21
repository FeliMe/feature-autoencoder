#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run FAE with 5 different random seeds
python train_fae.py --seed 0
python train_fae.py --seed 1
python train_fae.py --seed 2
python train_fae.py --seed 3
python train_fae.py --seed 4

# Run VAE with 5 different random seeds
python baselines/VAE/trainer.py --seed 0
python baselines/VAE/trainer.py --seed 1
python baselines/VAE/trainer.py --seed 2
python baselines/VAE/trainer.py --seed 3
python baselines/VAE/trainer.py --seed 4

# Run AE MSE with 5 different random seeds
python baselines/ae/trainer.py --seed 0 --loss_fn mse
python baselines/ae/trainer.py --seed 1 --loss_fn mse
python baselines/ae/trainer.py --seed 2 --loss_fn mse
python baselines/ae/trainer.py --seed 3 --loss_fn mse
python baselines/ae/trainer.py --seed 4 --loss_fn mse

# Run AE SSIM with 5 different random seeds
python baselines/ae/trainer.py --seed 0 --loss_fn ssim
python baselines/ae/trainer.py --seed 1 --loss_fn ssim
python baselines/ae/trainer.py --seed 2 --loss_fn ssim
python baselines/ae/trainer.py --seed 3 --loss_fn ssim
python baselines/ae/trainer.py --seed 4 --loss_fn ssim

# Run f-AnoGAN with 5 different random seeds
python baselines/fAnoGAN/trainer.py --seed 0
python baselines/fAnoGAN/trainer.py --seed 1
python baselines/fAnoGAN/trainer.py --seed 2
python baselines/fAnoGAN/trainer.py --seed 3
python baselines/fAnoGAN/trainer.py --seed 4

# Run FPI with 5 different random seeds
python baselines/fpi/trainer.py --seed 0 --interp_fn fpi
python baselines/fpi/trainer.py --seed 1 --interp_fn fpi
python baselines/fpi/trainer.py --seed 2 --interp_fn fpi
python baselines/fpi/trainer.py --seed 3 --interp_fn fpi
python baselines/fpi/trainer.py --seed 4 --interp_fn fpi

# Run PII with 5 different random seeds
python baselines/fpi/trainer.py --seed 0 --interp_fn pii
python baselines/fpi/trainer.py --seed 1 --interp_fn pii
python baselines/fpi/trainer.py --seed 2 --interp_fn pii
python baselines/fpi/trainer.py --seed 3 --interp_fn pii
python baselines/fpi/trainer.py --seed 4 --interp_fn pii

# Run DFR with 5 different random seeds
python baselines/DFR/trainer.py --seed 0
python baselines/DFR/trainer.py --seed 1
python baselines/DFR/trainer.py --seed 2
python baselines/DFR/trainer.py --seed 3
python baselines/DFR/trainer.py --seed 4

# Run BM
python baselines/BM/trainer.py
