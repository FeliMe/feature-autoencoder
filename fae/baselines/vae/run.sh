#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run VAE model with 5 different random seeds
python trainer.py --seed 0 --no_train --resume_path felix-meissen/feature_autoencoder/33u9gj1e/last.pt --test_dataset wmh
python trainer.py --seed 1 --no_train --resume_path felix-meissen/feature_autoencoder/mqbclf9q/last.pt --test_dataset wmh
python trainer.py --seed 2 --no_train --resume_path felix-meissen/feature_autoencoder/3cxwlcqx/last.pt --test_dataset wmh
python trainer.py --seed 3 --no_train --resume_path felix-meissen/feature_autoencoder/2ds0bzn0/last.pt --test_dataset wmh
python trainer.py --seed 4 --no_train --resume_path felix-meissen/feature_autoencoder/3vy1282u/last.pt --test_dataset wmh
