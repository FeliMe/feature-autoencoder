import os
from argparse import ArgumentParser
from collections import defaultdict
from time import time
from warnings import warn

import numpy as np
import torch
import wandb

from fae import WANDBNAME, WANDBPROJECT
from fae.baselines.ae.model import AE
from fae.configs.base_config import base_parser
from fae.data.datasets import get_dataloaders
from fae.utils.utils import seed_everything
from fae.utils import evaluation


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser(
    description="Arguments for training the Feature Autoencoder",
    parents=[base_parser],
    conflict_handler='resolve'
)
# Logging settings
parser.add_argument('--val_frequency', type=int,
                    default=1000, help='Validation frequency')
parser.add_argument('--log_frequency', type=int,
                    default=200, help='Logging frequency')
parser.add_argument('--save_frequency', type=int, default=1000,
                    help='Model checkpointing frequency')

# Hyperparameters
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

# Model settings
parser.add_argument('--hidden_dims', type=int, nargs='+',
                    default=[16, 32, 64, 128, 256], help='Autoencoder hidden dimensions')
parser.add_argument('--latent_dim', type=int, default=256,
                    help='Size of the latent space')
parser.add_argument('--loss_fn', type=str, default='mse', help='loss function',
                    choices=['l1', 'mse', 'ssim'])

args = parser.parse_args()

args.method = f"AE_{args.loss_fn}"

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
model = AE(config).to(config.device)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                             weight_decay=config.weight_decay)
# Print model
print(model.encoder)
print(model.decoder)

if config.resume_path is not None:
    print("Loading model from checkpoint...")
    model.load(config.resume_path)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, val_loader, test_loader = get_dataloaders(config)
print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train_step(model, optimizer, x, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    rec = model(x)
    loss_dict = model.loss(x, rec)
    loss = loss_dict['rec_loss']
    loss.backward()
    optimizer.step()
    return loss_dict, rec


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
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k,
                                     v in train_losses.items()])
                log_msg = f"Iteration {i_iter} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to tensorboard
                x = x.cpu().detach()
                rec = rec.cpu().detach()
                wandb.log({f'train/{k}': np.mean(v)
                          for k, v in train_losses.items()})
                wandb.log({
                    'train/input images': wandb.Image(x.cpu()[:config.num_images_log]),
                    'train/reconstructed images': wandb.Image(rec.cpu()[:config.num_images_log]),
                }, step=i_iter)

                # Reset
                train_losses = defaultdict(list)

            if i_iter % config.val_frequency == 0:
                validate(model, val_loader, config.device, i_iter)

            # Save model weights
            if i_iter % config.save_frequency == 0:
                model.save('last.pt')

            if i_iter >= config.max_steps:
                print(
                    f'Reached {config.max_steps} iterations. Finished training.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


def val_step(model, x, device):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        rec = model(x)
        loss_dict = model.loss(x, rec)
        anomaly_map, anomaly_score = model.predict_anomaly(x, rec)
    x = x.cpu()
    return loss_dict, anomaly_map.cpu(), anomaly_score.cpu(), rec.cpu()


def validate(model, val_loader, device, i_iter):
    i_val_step = 0
    val_losses = defaultdict(list)
    pixel_aps = []
    labels = []
    anomaly_scores = []

    for x, y, label in val_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss_dict, anomaly_map, anomaly_score, rec = val_step(model, x, device)

        # Compute metrics
        pixel_ap = evaluation.compute_average_precision(anomaly_map, y)

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
    sample_ap = evaluation.compute_average_precision(anomaly_scores, labels)
    sample_auroc = evaluation.compute_auroc(anomaly_scores, labels)

    # Print validation results
    print("\nValidation results:")
    log_msg = " - ".join([f'val {k}: {np.mean(v):.4f}' for k,
                         v in val_losses.items()])
    log_msg += f"\npixel-wise average precision: {np.mean(pixel_aps):.4f}\n"
    log_msg += f"sample-wise AUROC: {sample_auroc:.4f} - "
    log_msg += f"sample-wise average precision: {sample_ap:.4f} - "
    log_msg += f"Average positive label: {labels.float().mean():.4f}\n"
    print(log_msg)

    # Log to tensorboard
    wandb.log({
        f'val/{k}': np.mean(v) for k, v in val_losses.items()
    }, step=i_iter)
    wandb.log({
        'val/pixel-ap': np.mean(pixel_aps),
        'val/sample-ap': np.mean(sample_ap),
        'val/sample-auroc': np.mean(sample_auroc),
        'val/input images': wandb.Image(x.cpu()[:config.num_images_log]),
        'val/reconstructed images': wandb.Image(rec.cpu()[:config.num_images_log]),
        'val/targets': wandb.Image(y.float().cpu()[:config.num_images_log]),
        'val/anomaly maps': wandb.Image(anomaly_map.cpu()[:config.num_images_log]),
    }, step=i_iter)


def test(model, test_loader, device, config):
    val_losses = defaultdict(list)
    labels = []
    anomaly_scores = []
    segs = []
    anomaly_maps = []

    for x, y, label in test_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss_dict, anomaly_map, anomaly_score, rec = val_step(model, x, device)

        for k, v in loss_dict.items():
            val_losses[k].append(v.item())
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
    log_msg = " - ".join([f'val {k}: {np.mean(v):.4f}' for k,
                         v in val_losses.items()])
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
        f'val/{k}': np.mean(v) for k, v in val_losses.items()
    }, step=config.max_steps + 1)
    wandb.log({
        'val/pixel-ap': pixel_ap,
        'val/pixel-auroc': pixel_auroc,
        'val/sample-ap': sample_ap,
        'val/sample-auroc': sample_auroc,
        'val/iou-at-5fpr': iou_at_5fpr,
        'val/dice-at-5fpr': dice_at_5fpr,
        'val/input images': wandb.Image(x.cpu()[:config.num_images_log]),
        'val/reconstructed images': wandb.Image(rec.cpu()[:config.num_images_log]),
        'val/targets': wandb.Image(y.float().cpu()[:config.num_images_log]),
        'val/anomaly maps': wandb.Image(anomaly_map.cpu()[:config.num_images_log]),
    }, step=config.max_steps + 1)


if __name__ == '__main__':
    if config.train:
        train(model, optimizer, train_loader, val_loader, config)

    # Testing
    print('Testing...')
    test(model, test_loader, config.device, config)
