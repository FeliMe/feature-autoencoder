#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# # Run AE model with MSE loss with 5 different random seeds
# python trainer.py --loss_fn mse --seed 0 --no_train --resume_path felix-meissen/feature_autoencoder/1pk858il/last.pt --test_dataset wmh
# python trainer.py --loss_fn mse --seed 1 --no_train --resume_path felix-meissen/feature_autoencoder/33mks5g3/last.pt --test_dataset wmh
# python trainer.py --loss_fn mse --seed 2 --no_train --resume_path felix-meissen/feature_autoencoder/lporc35n/last.pt --test_dataset wmh
# python trainer.py --loss_fn mse --seed 3 --no_train --resume_path felix-meissen/feature_autoencoder/2ykzsopg/last.pt --test_dataset wmh
# python trainer.py --loss_fn mse --seed 4 --no_train --resume_path felix-meissen/feature_autoencoder/4kilf1u2/last.pt --test_dataset wmh

# # Run AE model with SSIM loss with 5 different random seeds
# python trainer.py --loss_fn ssim --seed 0 --no_train --resume_path felix-meissen/feature_autoencoder/30ea01j4/last.pt --test_dataset wmh
# python trainer.py --loss_fn ssim --seed 1 --no_train --resume_path felix-meissen/feature_autoencoder/2wmtgg3u/last.pt --test_dataset wmh
# python trainer.py --loss_fn ssim --seed 2 --no_train --resume_path felix-meissen/feature_autoencoder/3bu52i5r/last.pt --test_dataset wmh
# python trainer.py --loss_fn ssim --seed 3 --no_train --resume_path felix-meissen/feature_autoencoder/1j2ap691/last.pt --test_dataset wmh
# python trainer.py --loss_fn ssim --seed 4 --no_train --resume_path felix-meissen/feature_autoencoder/4c3bxb6y/last.pt --test_dataset wmh

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