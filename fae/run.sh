#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run FAE model with 5 different random seeds
python train_fae.py --seed 0  --no_train --resume_path felix-meissen/feature_autoencoder/2u9bt6ky/last.pt --test_dataset wmh
python train_fae.py --seed 1  --no_train --resume_path felix-meissen/feature_autoencoder/1ovjl78l/last.pt --test_dataset wmh
python train_fae.py --seed 2  --no_train --resume_path felix-meissen/feature_autoencoder/89hdaott/last.pt --test_dataset wmh
python train_fae.py --seed 3  --no_train --resume_path felix-meissen/feature_autoencoder/v6bj34h3/last.pt --test_dataset wmh
python train_fae.py --seed 4  --no_train --resume_path felix-meissen/feature_autoencoder/2g73wpfh/last.pt --test_dataset wmh
