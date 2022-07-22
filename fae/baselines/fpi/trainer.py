import os
from argparse import ArgumentParser
from time import time
from warnings import warn

import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
import wandb

from fae import WANDBNAME, WANDBPROJECT
from fae.baselines.fpi.model import WideResNetAE
from fae.baselines.fpi.fpi_utils import get_dataloaders
from fae.configs.base_config import base_parser
from fae.utils.utils import seed_everything
from fae.utils import evaluation


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser(
    description="Arguments for training the FPI/PII baseline",
    parents=[base_parser],
    conflict_handler='resolve'
)
# Hyperparameters
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

# Model settings
parser.add_argument('--interp_fn', type=str, default='fpi',
                    help='Interpolation function')

args = parser.parse_args()

args.method = f"patch_interpolation_{args.interp_fn}"

if not args.train and args.resume_path is None:
    warn("Testing untrained model")

# Select training device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb_dir = f"{os.path.expanduser('~')}/wandb/fae/{args.method}"
os.makedirs(wandb_dir, exist_ok=True)
wandb.init(project=WANDBPROJECT, entity=WANDBNAME, config=args,
           mode="disabled" if args.debug else "online",
           dir=wandb_dir)
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
                             weight_decay=config.weight_decay)
# Print model
summary(model, (1, config.image_size, config.image_size))

if config.resume_path is not None:
    print("Loading model from checkpoint...")
    model.load(config.resume_path)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, val_loader, test_loader = get_dataloaders(config)
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

    anomaly_map = pred.detach().cpu()
    return loss.item(), anomaly_map


def train(model, optimizer, train_loader, val_loader, config):
    print('Starting training...')
    i_iter = 0
    i_epoch = 0

    train_losses = []

    t_start = time()
    while True:
        for x, y in train_loader:
            i_iter += 1
            loss, anomaly_map = train_step(
                model, optimizer, x, y, config.device)

            # Add to losses
            train_losses.append(loss)

            if i_iter % config.log_frequency == 0:
                pixel_ap = evaluation.compute_average_precision(
                    anomaly_map.detach().cpu(), torch.where(y > 0, 1, 0))
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
                print("Saving model")
                model.save('last.pt')

            if i_iter >= config.max_steps:
                print(
                    f'Reached {config.max_steps} iterations. Finished training.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


def val_step(model, x, y, device):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        anomaly_map = model(x)
        loss = F.binary_cross_entropy(anomaly_map, y.float())
        anomaly_map = anomaly_map.cpu()
        anomaly_score = torch.tensor(
            [m[x_ > 0].mean() for m, x_ in zip(anomaly_map, x)])
    x = x.cpu()
    y = y.cpu()
    return loss.item(), anomaly_map, anomaly_score


def validate(model, val_loader, device, i_iter):
    val_losses = []
    pixel_aps = []
    labels = []
    anomaly_scores = []
    i_val_step = 0

    for x, y, label in val_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss, anomaly_map, anomaly_score = val_step(model, x, y, device)

        pixel_ap = evaluation.compute_average_precision(anomaly_map, y.cpu())

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
    sample_ap = evaluation.compute_average_precision(anomaly_scores, labels)
    sample_auroc = evaluation.compute_auroc(anomaly_scores, labels)

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


def test(model, test_loader, device, config):
    val_losses = []
    labels = []
    anomaly_scores = []
    segs = []
    anomaly_maps = []

    for x, y, label in test_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss, anomaly_map, anomaly_score = val_step(model, x, y, device)

        val_losses.append(loss)
        labels.append(label)
        anomaly_scores.append(anomaly_score)

        segs.append(y)
        anomaly_maps.append(anomaly_map)

    # Sample-wise metrics
    labels = torch.cat(labels).numpy()
    anomaly_scores = torch.cat(anomaly_scores).numpy()
    sample_ap = evaluation.compute_average_precision(anomaly_scores, labels)
    sample_auroc = evaluation.compute_auroc(anomaly_scores, labels)

    # Pixel-wise metrics
    anomaly_maps = torch.cat(anomaly_maps).numpy()
    segs = torch.cat(segs).numpy()
    pixel_ap = evaluation.compute_average_precision(anomaly_maps, segs)
    pixel_auroc = evaluation.compute_auroc(anomaly_maps, segs)
    iou_at_5fpr = evaluation.compute_iou_at_nfpr(anomaly_maps, segs,
                                                 max_fpr=0.05)
    dice_at_5fpr = evaluation.compute_dice_at_nfpr(anomaly_maps, segs,
                                                   max_fpr=0.05)

    # Print test results
    print("\nTest results:")
    log_msg = f"Validation loss: {np.mean(val_losses):.4f}"
    log_msg += f"\npixel-wise average precision: {pixel_ap:.4f} - "
    log_msg += f"pixel-wise AUROC: {pixel_auroc:.4f}\n"
    log_msg += f"IoU @ 5% fpr: {iou_at_5fpr:.4f} - "
    log_msg += f"Dice @ 5% fpr: {dice_at_5fpr:.4f}\n"
    log_msg += f"sample-wise AUROC: {sample_auroc:.4f} - "
    log_msg += f"sample-wise average precision: {sample_ap:.4f} - "
    log_msg += f"Average positive label: {torch.tensor(segs).float().mean():.4f}\n"
    print(log_msg)

    # Log to tensorboard
    wandb.log({
        'val/loss': np.mean(val_losses),
        'val/pixel-ap': pixel_ap,
        'val/pixel-auroc': pixel_auroc,
        'val/sample-ap': sample_ap,
        'val/sample-auroc': sample_auroc,
        'val/iou-at-5fpr': iou_at_5fpr,
        'val/dice-at-5fpr': dice_at_5fpr,
        'val/input images': wandb.Image(x.cpu()[:config.num_images_log]),
        'val/targets': wandb.Image(y.float().cpu()[:config.num_images_log]),
        'val/anomaly maps': wandb.Image(anomaly_map.cpu()[:config.num_images_log]),
    }, step=config.max_steps + 1)


if __name__ == '__main__':
    if config.train:
        train(model, optimizer, train_loader, val_loader, config)

    # Testing
    print('Testing...')
    test(model, test_loader, config.device, config)
