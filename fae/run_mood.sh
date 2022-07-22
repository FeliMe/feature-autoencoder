#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run FAE with 5 different random seeds
python train_fae.py --seed 0 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python train_fae.py --seed 1 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python train_fae.py --seed 2 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python train_fae.py --seed 3 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python train_fae.py --seed 4 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation

# Run VAE with 5 different random seeds
python baselines/VAE/trainer.py --seed 0 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/VAE/trainer.py --seed 1 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/VAE/trainer.py --seed 2 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/VAE/trainer.py --seed 3 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/VAE/trainer.py --seed 4 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation

# Run AE MSE with 5 different random seeds
python baselines/ae/trainer.py --seed 0 --loss_fn mse --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/ae/trainer.py --seed 1 --loss_fn mse --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/ae/trainer.py --seed 2 --loss_fn mse --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/ae/trainer.py --seed 3 --loss_fn mse --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/ae/trainer.py --seed 4 --loss_fn mse --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation

# Run AE SSIM with 5 different random seeds
python baselines/ae/trainer.py --seed 0 --loss_fn ssim --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/ae/trainer.py --seed 1 --loss_fn ssim --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/ae/trainer.py --seed 2 --loss_fn ssim --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/ae/trainer.py --seed 3 --loss_fn ssim --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/ae/trainer.py --seed 4 --loss_fn ssim --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation

# Run f-AnoGAN with 5 different random seeds
python baselines/fAnoGAN/trainer.py --seed 0 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fAnoGAN/trainer.py --seed 1 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fAnoGAN/trainer.py --seed 2 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fAnoGAN/trainer.py --seed 3 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fAnoGAN/trainer.py --seed 4 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation

# Run FPI with 5 different random seeds
python baselines/fpi/trainer.py --seed 0 --interp_fn fpi --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fpi/trainer.py --seed 1 --interp_fn fpi --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fpi/trainer.py --seed 2 --interp_fn fpi --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fpi/trainer.py --seed 3 --interp_fn fpi --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fpi/trainer.py --seed 4 --interp_fn fpi --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation

# Run PII with 5 different random seeds
python baselines/fpi/trainer.py --seed 0 --interp_fn pii --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fpi/trainer.py --seed 1 --interp_fn pii --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fpi/trainer.py --seed 2 --interp_fn pii --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fpi/trainer.py --seed 3 --interp_fn pii --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/fpi/trainer.py --seed 4 --interp_fn pii --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation

# Run DFR with 5 different random seeds
python baselines/DFR/trainer.py --seed 0 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/DFR/trainer.py --seed 1 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/DFR/trainer.py --seed 2 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/DFR/trainer.py --seed 3 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
python baselines/DFR/trainer.py --seed 4 --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation

# Run BM
python baselines/BM/trainer.py --train_dataset mood_train --test_dataset mood_val_test --anomaly_size 10 20 --anomaly_name sink_deformation
