#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

python trainer.py --seed 0
python trainer.py --seed 1
python trainer.py --seed 2
python trainer.py --seed 3
python trainer.py --seed 4
python trainer.py --seed 5
