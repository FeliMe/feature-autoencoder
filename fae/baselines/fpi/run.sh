#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Run FPI with 5 different random seeds
python trainer.py --seed 0 --interp_fn fpi --no_train --resume_path felix-meissen/feature_autoencoder/r0noiy7s/last.pt --test_dataset wmh
python trainer.py --seed 1 --interp_fn fpi --no_train --resume_path felix-meissen/feature_autoencoder/oo7g80wy/last.pt --test_dataset wmh
python trainer.py --seed 2 --interp_fn fpi --no_train --resume_path felix-meissen/feature_autoencoder/1wbkv3y2/last.pt --test_dataset wmh
python trainer.py --seed 3 --interp_fn fpi --no_train --resume_path felix-meissen/feature_autoencoder/18iwxlm5/last.pt --test_dataset wmh
python trainer.py --seed 4 --interp_fn fpi --no_train --resume_path felix-meissen/feature_autoencoder/34uoj414/last.pt --test_dataset wmh

# Run PII with 5 different random seeds
python trainer.py --seed 0 --interp_fn pii --no_train --resume_path felix-meissen/feature_autoencoder/29n8de9u/last.pt --test_dataset wmh
python trainer.py --seed 1 --interp_fn pii --no_train --resume_path felix-meissen/feature_autoencoder/2l3xywei/last.pt --test_dataset wmh
python trainer.py --seed 2 --interp_fn pii --no_train --resume_path felix-meissen/feature_autoencoder/3cjc91i3/last.pt --test_dataset wmh
python trainer.py --seed 3 --interp_fn pii --no_train --resume_path felix-meissen/feature_autoencoder/2461rt8s/last.pt --test_dataset wmh
python trainer.py --seed 4 --interp_fn pii --no_train --resume_path felix-meissen/feature_autoencoder/j5kd4wcb/last.pt --test_dataset wmh
