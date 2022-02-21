#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run f-AnoGAN with 5 different random seeds
python trainer.py --seed 0 --no_train --resume_path felix-meissen/feature_autoencoder/w81uw30u/last_encoder.pt --test_dataset wmh
python trainer.py --seed 1 --no_train --resume_path felix-meissen/feature_autoencoder/3e7rtnat/last_encoder.pt --test_dataset wmh
python trainer.py --seed 2 --no_train --resume_path felix-meissen/feature_autoencoder/1uskqsal/last_encoder.pt --test_dataset wmh
python trainer.py --seed 3 --no_train --resume_path felix-meissen/feature_autoencoder/1h5kosk6/last_encoder.pt --test_dataset wmh
python trainer.py --seed 4 --no_train --resume_path felix-meissen/feature_autoencoder/2ynw7u6p/last_encoder.pt --test_dataset wmh
