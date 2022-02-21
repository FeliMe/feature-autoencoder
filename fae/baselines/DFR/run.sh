#!/bin/bash

## Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

## Run DFR with 5 different random seeds
python trainer.py --seed 0 --batch_size 4
python trainer.py --seed 1 --batch_size 4
python trainer.py --seed 2 --batch_size 4
python trainer.py --seed 3 --batch_size 4
python trainer.py --seed 4 --batch_size 4
