# Add the parent directory to sys.path to allow importing from there
import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(this_dir))

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from time import time

import numpy as np
import torch

from model import VAE
from datasets import get_dataloaders
from utils import (
    seed_everything,
    compute_auroc,
    compute_average_precision,
    TensorboardLogger
)


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser()
# General script settings
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')

# Data settings
parser.add_argument('--image_size', type=int, default=128, help='Image size')
parser.add_argument('--sequence', type=str, default='t1', help='MRI sequence')
parser.add_argument('--slice_range', type=int, nargs='+', default=(55, 135), help='Lower and Upper slice index')
parser.add_argument('--normalize', type=bool, default=False, help='Normalize images between 0 and 1')
parser.add_argument('--equalize_histogram', type=bool, default=True, help='Equalize histogram')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

# Logging settings
parser.add_argument('--val_frequency', type=int, default=200, help='Validation frequency')
parser.add_argument('--val_steps', type=int, default=50, help='Steps per validation')
parser.add_argument('--log_frequency', type=int, default=50, help='Logging frequency')
parser.add_argument('--num_images_log', type=int, default=10, help='Number of images to log')
parser.add_argument(
    '--log_dir', type=str, help="Logging directory",
    default=os.path.join(this_dir, 'logs', datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S"))
)

# Hyperparameters
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--kl_weight', type=float, default=0.05, help='Weight of KL loss')  # Very sensitive to this param, 0.01

# Model settings
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256], help='Autoencoder hidden dimensions')
parser.add_argument('--latent_dim', type=int, default=128, help='Size of the latent space')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')

config = parser.parse_args()

# Select training device
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'


""""""""""""""""""""""""""""""" Reproducability """""""""""""""""""""""""""""""
seed_everything(config.seed)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


print("Initializing model...")
model = VAE(config).to(config.device)
# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                             weight_decay=config.weight_decay)  # betas = (0.9, 0.999)
# Print model
print(model.encoder)
print(model.decoder)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, test_loader = get_dataloaders(config)
print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""
writer = TensorboardLogger(config.log_dir, config=config, flush_secs=10)


def train_step(model, optimizer, x, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    rec, mu, logvar = model(x)
    loss_dict = model.loss_function(x, rec, mu, logvar, config.kl_weight)
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict, rec


def val_step(model, x, device):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        rec, mu, logvar = model(x)
        loss_dict = model.loss_function(x, rec, mu, logvar, config.kl_weight)
        anomaly_map, anomaly_score = model.predict_anomaly(x, rec, mu, logvar)
    return loss_dict, anomaly_map.cpu(), anomaly_score.cpu(), rec.cpu()


def validate(model, val_loader, device, i_iter):
    val_losses = defaultdict(list)
    pixel_aps = []
    labels = []
    anomaly_scores = []
    i_val_step = 0

    for x, y in val_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss_dict, anomaly_map, anomaly_score, rec = val_step(model, x, device)

        # Compute metrics
        label = torch.where(y.sum(dim=(1, 2, 3)) > 16, 1, 0)  # TODO: Turn to 0
        pixel_ap = compute_average_precision(anomaly_map, y)

        for k, v in loss_dict.items():
            val_losses[k].append(v.item())
        pixel_aps.append(pixel_ap)
        labels.append(label)
        anomaly_scores.append(anomaly_score)

        i_val_step += 1
        if i_val_step >= config.val_steps:
            break

    # Compute sample-wise average precision and AUROC over all validation steps
    labels = torch.cat(labels)
    anomaly_scores = torch.cat(anomaly_scores)
    sample_ap = compute_average_precision(anomaly_scores, labels)
    sample_auroc = compute_auroc(anomaly_scores, labels)

    # Print validation results
    print("\nValidation results:")
    log_msg = " - ".join([f'val {k}: {np.mean(v):.4f}' for k, v in val_losses.items()])
    log_msg += f"\npixel-wise average precision: {np.mean(pixel_aps):.4f}\n"
    log_msg += f"sample-wise AUROC: {sample_auroc:.4f} - "
    log_msg += f"sample-wise average precision: {sample_ap:.4f} - "
    log_msg += f"Average positive label: {labels.float().mean():.4f}\n"
    print(log_msg)

    # Log to tensorboard
    writer.log({
        f'val/{k}': np.mean(v) for k, v in val_losses.items()
    }, step=i_iter)
    writer.log({
        'val/pixel-ap': np.mean(pixel_aps),
        'val/sample-ap': np.mean(sample_ap),
        'val/sample-auroc': np.mean(sample_auroc),
        'val/input images': x.cpu()[:config.num_images_log],
        'val/reconstructed images': rec.cpu()[:config.num_images_log],
        'val/targets': y.float().cpu()[:config.num_images_log],
        'val/anomaly maps': anomaly_map.cpu()[:config.num_images_log]
    }, step=i_iter)


def train(model, optimizer, train_loader, val_loader, config):
    print('Starting training...')
    i_iter = 0
    i_epoch = 0

    train_losses = defaultdict(list)

    t_start = time()
    while True:
        for x in train_loader:
            i_iter += 1
            loss_dict, rec = train_step(model, optimizer, x, config.device)

            # Add to losses
            for k, v in loss_dict.items():
                train_losses[k].append(v.item())

            if i_iter % config.log_frequency == 0:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k, v in train_losses.items()])
                log_msg = f"Iteration {i_iter} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to tensorboard
                x = x.cpu().detach()
                rec = rec.cpu().detach()
                log_dict = {f'train/{k}': np.mean(v) for k, v in train_losses.items()}
                log_dict['train/input images'] = x[:config.num_images_log]
                log_dict['train/reconstructed images'] = rec[:config.num_images_log]
                log_dict['train/anomaly maps'] = (x - rec).abs()[:config.num_images_log]
                writer.log(log_dict, step=i_iter)

                # Reset
                train_losses = defaultdict(list)

            if i_iter % config.val_frequency == 0:
                validate(model, val_loader, config.device, i_iter)

            if i_iter >= config.max_steps:
                print(f'Reached {config.max_steps} iterations. Finished training.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


if __name__ == '__main__':
    train(model, optimizer, train_loader, test_loader, config)
