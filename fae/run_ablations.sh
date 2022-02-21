
#!/bin/bash

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomaly_detection

# Layer 0,1
python train_fae.py --seed 0 --extractor_cnn_layers layer0 layer1
python train_fae.py --seed 1 --extractor_cnn_layers layer0 layer1
python train_fae.py --seed 2 --extractor_cnn_layers layer0 layer1
python train_fae.py --seed 3 --extractor_cnn_layers layer0 layer1
python train_fae.py --seed 4 --extractor_cnn_layers layer0 layer1

# Layer 0,1,2
python train_fae.py --seed 0 --extractor_cnn_layers layer0 layer1 layer2
python train_fae.py --seed 1 --extractor_cnn_layers layer0 layer1 layer2
python train_fae.py --seed 2 --extractor_cnn_layers layer0 layer1 layer2
python train_fae.py --seed 3 --extractor_cnn_layers layer0 layer1 layer2
python train_fae.py --seed 4 --extractor_cnn_layers layer0 layer1 layer2

# Layer 0,1,2,3
python train_fae.py --seed 0 --extractor_cnn_layers layer0 layer1 layer2 layer3
python train_fae.py --seed 1 --extractor_cnn_layers layer0 layer1 layer2 layer3
python train_fae.py --seed 2 --extractor_cnn_layers layer0 layer1 layer2 layer3
python train_fae.py --seed 3 --extractor_cnn_layers layer0 layer1 layer2 layer3
python train_fae.py --seed 4 --extractor_cnn_layers layer0 layer1 layer2 layer3

# Layer 1,2,3
python train_fae.py --seed 0 --extractor_cnn_layers layer1 layer2 layer3
python train_fae.py --seed 1 --extractor_cnn_layers layer1 layer2 layer3
python train_fae.py --seed 2 --extractor_cnn_layers layer1 layer2 layer3
python train_fae.py --seed 3 --extractor_cnn_layers layer1 layer2 layer3
python train_fae.py --seed 4 --extractor_cnn_layers layer1 layer2 layer3

# Layer 1,2
python train_fae.py --seed 0 --extractor_cnn_layers layer1 layer2
python train_fae.py --seed 1 --extractor_cnn_layers layer1 layer2
python train_fae.py --seed 2 --extractor_cnn_layers layer1 layer2
python train_fae.py --seed 3 --extractor_cnn_layers layer1 layer2
python train_fae.py --seed 4 --extractor_cnn_layers layer1 layer2

# Layer 2,3
python train_fae.py --seed 0 --extractor_cnn_layers layer2 layer3
python train_fae.py --seed 1 --extractor_cnn_layers layer2 layer3
python train_fae.py --seed 2 --extractor_cnn_layers layer2 layer3
python train_fae.py --seed 3 --extractor_cnn_layers layer2 layer3
python train_fae.py --seed 4 --extractor_cnn_layers layer2 layer3

# Untrained extractor
python train_fae.py --seed 0 --random_extractor
python train_fae.py --seed 1 --random_extractor
python train_fae.py --seed 2 --random_extractor
python train_fae.py --seed 3 --random_extractor
python train_fae.py --seed 4 --random_extractor

# FAE with MSE loss
python train_fae.py --seed 0 --loss_fn mse
python train_fae.py --seed 1 --loss_fn mse
python train_fae.py --seed 2 --loss_fn mse
python train_fae.py --seed 3 --loss_fn mse
python train_fae.py --seed 4 --loss_fn mse
