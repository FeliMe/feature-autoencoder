from argparse import ArgumentParser
from os.path import dirname
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
import wandb

from fae.baselines.fpi.model import WideResNetAE
from fae.baselines.fpi.fpi_utils import get_dataloaders
from fae.utils.utils import seed_everything
from fae.utils.evaluation import compute_auroc, compute_average_precision


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
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

# Model settings
parser.add_argument('--interp_fn', type=str, default='fpi', help='Interpolation function')

args = parser.parse_args()

# Select training device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project="feature_autoencoder", entity="felix-meissen", config=args,
           mode="disabled" if args.debug else "online",
           dir=dirname(dirname(dirname(__file__))))
config = wandb.config


""""""""""""""""""""""""""""""" Reproducability """""""""""""""""""""""""""""""
seed_everything(config.seed)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


print("Initializing model...")
model = WideResNetAE(config).to(config.device)

# Track model with w&b
wandb.watch(model)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                             weight_decay=config.weight_decay)  # betas = (0.9, 0.999)
# Print model
summary(model, (1, config.image_size, config.image_size))

if config.resume_path is not None:
    print("Loading model from checkpoint...")
    model.load(config.resume_path)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, test_loader = get_dataloaders(config)
print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train_step(model, optimizer, x, y, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    pred = model(x)
    loss = F.binary_cross_entropy(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item(), pred


def val_step(model, x, y, device):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        anomaly_map = model(x)
        loss = F.binary_cross_entropy(anomaly_map, y.float())
        anomaly_map = anomaly_map.cpu()
        anomaly_score = torch.tensor([m[x_ > 0].mean() for m, x_ in zip(anomaly_map, x)])
        pixel_ap = compute_average_precision(anomaly_map, y.cpu())
    return loss.item(), pixel_ap, anomaly_map, anomaly_score


def validate(model, val_loader, device, i_iter):
    val_losses = []
    pixel_aps = []
    labels = []
    anomaly_scores = []
    i_val_step = 0

    for x, y in val_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss, pixel_ap, anomaly_map, anomaly_score = val_step(model, x, y, device)

        # Compute metrics
        label = torch.where(y.sum(dim=(1, 2, 3)) > 16, 1, 0)  # TODO: Turn to 0

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


def train(model, optimizer, train_loader, val_loader, config):
    print('Starting training...')
    i_iter = 0
    i_epoch = 0

    train_losses = []

    t_start = time()
    while True:
        for x, y in train_loader:
            i_iter += 1
            loss, anomaly_map = train_step(model, optimizer, x, y, config.device)

            # Add to losses
            train_losses.append(loss)

            if i_iter % config.log_frequency == 0:
                pixel_ap = compute_average_precision(anomaly_map.detach().cpu(), torch.where(y > 0, 1, 0))
                # Print training loss
                log_msg = f"Iteration {i_iter} - "
                log_msg += f"train loss: {np.mean(train_losses):.4f} - "
                log_msg += f"train pixel-ap: {pixel_ap:.4f} - "
                log_msg += f"time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to tensorboard
                x = x.cpu().detach()
                y = y.cpu().detach()
                anomaly_map = anomaly_map.cpu().detach()
                log_dict = {
                    'train/loss': np.mean(train_losses),
                    'train/pixel-ap': pixel_ap,
                    'train/input images': wandb.Image(x[:config.num_images_log]),
                    'train/anomaly maps': wandb.Image(anomaly_map[:config.num_images_log]),
                    'train/targets': wandb.Image(y[:config.num_images_log]),
                }
                wandb.log(log_dict, step=i_iter)

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
