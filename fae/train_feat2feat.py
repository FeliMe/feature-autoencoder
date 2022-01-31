from argparse import ArgumentParser
from collections import defaultdict
from time import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor

from fae.configs.base_config import base_parser
from fae.data import datasets
from fae.models import models
from fae.utils.utils import seed_everything, GANBCELoss, GANNonSaturatingWithR1
from fae.utils.evaluation import compute_average_precision, compute_auroc


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser(
    description="Arguments for training the Feature Autoencoder",
    parents=[base_parser]
)
# parser.add_argument('--discriminator_hidden_dims', nargs='+', type=int,
#                     default=[400, 450, 500, 600])
parser.add_argument('--discriminator_hidden_dims', nargs='+', type=int,
                    default=[400, 450, 500])
parser.add_argument('--lr_d', type=float, default=1e-5)
parser.add_argument('--rec_weight', type=float, default=1.)
parser.add_argument('--adv_weight', type=float, default=0.1)  # 0.3
parser.add_argument('--discpl_weight', type=float, default=0.01)  # 10

args = parser.parse_args()

# Select training device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project="feature_autoencoder", entity="felix-meissen", config=args,
           mode="disabled" if args.debug else "online")
config = wandb.config


""""""""""""""""""""""""""""""" Reproducability """""""""""""""""""""""""""""""
seed_everything(config.seed)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


def get_models(config):
    if config.model in models.__dict__:
        model_cls = models.__dict__[config.model]
    else:
        raise ValueError(f'Model {config.model} not found')

    G = model_cls(config)
    D = models.FeatureDiscriminator(config)
    return G, D


print("Initializing model...")
G, D = get_models(config)
G = G.to(config.device)
D = D.to(config.device)
print(G.ae.enc)
print(G.ae.dec)
print(D)
# Track model with w&b
wandb.watch(G)
wandb.watch(D)
# Init optimizer
optimizer_g = torch.optim.Adam(G.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)  # betas = (0.9, 0.999)
optimizer_d = torch.optim.Adam(D.parameters(), lr=config.lr_d,
                               weight_decay=config.weight_decay)  # betas = (0.9, 0.999)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, test_loader = datasets.get_dataloaders(config)
print(f'Loaded {config.train_dataset} and {config.test_dataset} in '
      f'{time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def set_requires_grad(model, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def feature_matching_loss(feats_real: List[Tensor], feats_fake: List[Tensor]):
    return torch.stack([
        F.mse_loss(rf, ff) for rf, ff in zip(feats_real, feats_fake)
    ]).mean()


def train_step(G, D, optimizer_g, optimizer_d, x, device):
    G.train()
    D.train()
    x = x.to(device)

    gan_loss_fn = GANNonSaturatingWithR1()
    # gan_loss_fn = GANBCELoss()

    # Forward generator
    feats = G.get_feats(x)

    """ 1. Train Discriminator, maximize log(D(x)) + log(1 - D(G(z))) """
    set_requires_grad(D, True)
    set_requires_grad(G, False)
    optimizer_d.zero_grad()

    # Get real and fake features
    # feats, rec = G(x)
    rec = G.get_rec(feats)

    # Discriminator adversarial loss
    gan_loss_fn.pre_discriminator_step(feats)
    pred_real, feats_real = D(feats)
    pred_fake, feats_fake = D(rec.detach())
    adv_loss_d, adv_loss_metrics_d = gan_loss_fn.discriminator_loss(pred_real, pred_fake, feats)

    # Discriminator feature matching loss
    disc_pl_loss = feature_matching_loss(feats_real, feats_fake)

    # Discriminator combine losses and backward
    discriminator_loss = config.adv_weight * adv_loss_d + config.discpl_weight * disc_pl_loss
    discriminator_loss.backward()
    optimizer_d.step()

    """ 2. Train Generator, maximize log(D(G(z))) """
    set_requires_grad(D, False)
    set_requires_grad(G, True)
    optimizer_g.zero_grad()

    # Get real and fake features
    # feats, rec = G(x)
    rec = G.get_rec(feats)

    # Generator adversarial loss
    pred_fake, _ = D(rec)
    adv_loss_g = gan_loss_fn.generator_loss(pred_fake)

    # Generator reconstruction loss
    rec_loss = G.loss_fn(rec, feats).mean()

    # Generator combine losses and backward
    generator_loss = config.adv_weight * adv_loss_g + config.rec_weight * rec_loss
    generator_loss.backward()
    optimizer_g.step()

    return {
        'rec_loss': rec_loss.item(),
        'adv_loss_d': adv_loss_d.item(),
        'adv_loss_g': adv_loss_g.item(),
        'disc_pl_loss': disc_pl_loss.item(),
    } | adv_loss_metrics_d  # combine dicts


def val_step(G, D, x, device):
    G.eval()
    D.eval()
    x = x.to(device)

    gan_loss_fn = GANNonSaturatingWithR1()
    # gan_loss_fn = GANBCELoss()

    with torch.no_grad():
        # Get real and fake features
        feats, rec = G(x)

        # 1.1 Generator adversarial loss
        pred_fake, _ = D(rec)
        adv_loss_g = gan_loss_fn.generator_loss(pred_fake)

        # 1.2 Autoencoder reconstruction loss
        rec_loss = G.loss_fn(rec, feats).mean()

        # 2.1 Discriminator adversarial loss
        gan_loss_fn.pre_discriminator_step(feats)
        pred_real, feats_real = D(feats)
        pred_fake, feats_fake = D(rec.detach())
        adv_loss_d, adv_loss_metrics = gan_loss_fn.discriminator_loss(pred_real, pred_fake, feats)

        # 2.3 Discriminator feature matching loss
        disc_pl_loss = feature_matching_loss(feats_real, feats_fake)

        # Anomaly map and score
        anomaly_map, anomaly_score = G.predict_anomaly(x)

    return {
        'rec_loss': rec_loss.item(),
        'adv_loss_d': adv_loss_d.item(),
        'adv_loss_g': adv_loss_g.item(),
        'disc_pl_loss': disc_pl_loss.item(),
    } | adv_loss_metrics, anomaly_map.cpu(), anomaly_score.cpu()


def validate(G, D, val_loader, device, i_iter):
    val_losses = defaultdict(list)
    pixel_aps = []
    labels = []
    anomaly_scores = []
    i_val_step = 0

    for x, y, label in val_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss_dict, anomaly_map, anomaly_score = val_step(G, D, x, device)

        # Compute metrics
        pixel_ap = compute_average_precision(anomaly_map, y)

        for k, v in loss_dict.items():
            val_losses[k].append(v)
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

    # Log to w&b
    wandb.log({
        f'val/{k}': np.mean(v) for k, v in val_losses.items()
    }, step=i_iter)
    wandb.log({
        'val/pixel-ap': np.mean(pixel_aps),
        'val/sample-ap': np.mean(sample_ap),
        'val/sample-auroc': np.mean(sample_auroc),
        'val/input images': wandb.Image(x.cpu()[:config.num_images_log]),
        'val/targets': wandb.Image(y.float().cpu()[:config.num_images_log]),
        'val/anomaly maps': wandb.Image(anomaly_map.cpu()[:config.num_images_log])
    }, step=i_iter)


def train(G, D, optimizer_g, optimizer_d, train_loader, val_loader, config):
    print('Starting training...')
    i_iter = 0
    i_epoch = 0

    train_losses = defaultdict(list)

    t_start = time()
    while True:
        for x in train_loader:
            i_iter += 1
            loss_dict = train_step(G, D, optimizer_g, optimizer_d, x, config.device)

            # Add to losses
            for k, v in loss_dict.items():
                train_losses[k].append(v)

            if i_iter % config.log_frequency == 0:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k, v in train_losses.items()])
                log_msg = f"Iteration {i_iter} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to w&b
                wandb.log({
                    f'train/{k}': np.mean(v) for k, v in train_losses.items()
                }, step=i_iter)

                # Reset
                train_losses = defaultdict(list)

            if i_iter % config.val_frequency == 0:
                validate(G, D, val_loader, config.device, i_iter)

            if i_iter >= config.max_steps:
                print(f'Reached {config.max_steps} iterations. Finished training.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


if __name__ == '__main__':
    train(G, D, optimizer_g, optimizer_d, train_loader, test_loader, config)
