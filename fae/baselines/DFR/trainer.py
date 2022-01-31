from argparse import ArgumentParser
from os.path import dirname
from time import time

import numpy as np
import torch
import wandb

from model import Extractor, FeatureAE
from fae.data.datasets import get_dataloaders
from fae.utils.utils import seed_everything
from fae.utils.evaluation import compute_auroc, compute_average_precision
from fae.baselines.DFR.dfr_utils import estimate_latent_channels


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser()
# General script settings
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--resume_path', type=str, help='W&B path to checkpoint to resume training from')

# Data settings
parser.add_argument('--train_dataset', type=str, default='camcan', help='Training dataset name')
parser.add_argument('--test_dataset', type=str, default='brats', help='Test dataset name',
                    choices=['brats', 'mslub', 'msseg', 'wmh'])
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
parser.add_argument('--save_frequency', type=int, default=200, help='Model checkpointing frequency')
parser.add_argument('--num_images_log', type=int, default=10, help='Number of images to log')

# Hyperparameters
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')

# Model settings
parser.add_argument('--latent_channels', type=int, default=None, help='Number of channels in latent space')

args = parser.parse_args()

# Select training device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project="feature_autoencoder", entity="felix-meissen", config=args,
           mode="disabled" if args.debug else "online",
           dir=dirname(dirname(dirname(__file__))))
config = wandb.config


""""""""""""""""""""""""""""""" Reproducability """""""""""""""""""""""""""""""
seed_everything(config.seed)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, test_loader = get_dataloaders(config)
print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


# Estimating number of latent channels
if "latent_channels" not in config or config.latent_channels is None:
    extractor = Extractor(featmap_size=config.image_size)
    config.update(
        {'latent_channels': estimate_latent_channels(extractor, train_loader)},
        allow_val_change=True
    )
    print(f'Latent channels: {config.latent_channels}')
    del(extractor)

print("Initializing model...")
model = FeatureAE(
    img_size=config.image_size,
    latent_channels=config.latent_channels,
).to(config.device)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                             weight_decay=config.weight_decay)  # betas = (0.9, 0.999)
# Print model
print(model.encoder)
print(model.decoder)

if config.resume_path is not None:
    print("Loading model from checkpoint...")
    model.load(config.resume_path)


""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def val_step(model, x, device):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        anomaly_map, anomaly_score, loss = model.predict_anomaly(x)
    return loss.item(), anomaly_map.cpu(), anomaly_score.cpu()


def validate(model, val_loader, device, i_iter):
    val_losses = []
    pixel_aps = []
    labels = []
    anomaly_scores = []
    i_val_step = 0

    for x, y, label in val_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Skip if no anomalies in batch
        if y.sum() == 0:
            continue

        # Compute loss, anomaly map and anomaly score
        loss, anomaly_map, anomaly_score = val_step(model, x, device)

        # Compute metrics
        pixel_ap = compute_average_precision(anomaly_map, y)

        val_losses.append(loss)
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
    log_msg = f"Validation loss: {np.mean(val_losses):.4f}"
    log_msg += f"\npixel-wise average precision: {np.mean(pixel_aps):.4f}\n"
    log_msg += f"sample-wise AUROC: {sample_auroc:.4f} - "
    log_msg += f"sample-wise average precision: {sample_ap:.4f} - "
    log_msg += f"Average positive label: {labels.float().mean():.4f}\n"
    print(log_msg)

    # Log to tensorboard
    wandb.log({
        'val/loss': np.mean(val_losses),
        'val/pixel-ap': np.mean(pixel_aps),
        'val/sample-ap': np.mean(sample_ap),
        'val/sample-auroc': np.mean(sample_auroc),
        'val/input images': wandb.Image(x.cpu()[:config.num_images_log]),
        'val/targets': wandb.Image(y.float().cpu()[:config.num_images_log]),
        'val/anomaly maps': wandb.Image(anomaly_map.cpu()[:config.num_images_log]),
    }, step=i_iter)


def train_step(model, optimizer, x, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    loss = model.loss(x)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, optimizer, train_loader, val_loader, config):
    print('Starting training...')
    i_iter = 0
    i_epoch = 0

    train_losses = []

    t_start = time()
    while True:
        for x in train_loader:
            i_iter += 1
            loss = train_step(model, optimizer, x, config.device)

            # Add to losses
            train_losses.append(loss)

            if i_iter % config.log_frequency == 0:
                # Print training loss
                log_msg = f"Iteration {i_iter} - "
                log_msg += f"train loss: {np.mean(train_losses):.4f}"
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to tensorboard
                x = x.cpu().detach()
                wandb.log({
                    'train/loss': np.mean(train_losses),
                    'train/input images': wandb.Image(x[:config.num_images_log]),
                }, step=i_iter)

                # Reset
                train_losses = []

            if i_iter % config.val_frequency == 0:
                validate(model, val_loader, config.device, i_iter)

            # Save model weights
            if i_iter % config.save_frequency == 0:
                model.save('last.pt')

            if i_iter >= config.max_steps:
                print(f'Reached {config.max_steps} iterations. Finished training.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


if __name__ == '__main__':
    train(model, optimizer, train_loader, test_loader, config)
