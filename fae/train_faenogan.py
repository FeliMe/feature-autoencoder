
from argparse import ArgumentParser
from collections import defaultdict
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
import wandb

from fae.models.feature_extractor import Extractor
from fae.models.gan_models import fAnoGAN
from fae.data.datasets import get_dataloaders
from fae.utils.pytorch_ssim import SSIMLoss
from fae.utils.utils import seed_everything, calc_gradient_penalty
from fae.utils.evaluation import compute_auroc, compute_average_precision


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser()
# General script settings
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--resume_path', type=str,
                    help='W&B path to checkpoint to resume training from')

# Data settings
parser.add_argument('--train_dataset', type=str,
                    default='camcan', help='Training dataset name')
parser.add_argument('--test_dataset', type=str, default='brats',
                    help='Test dataset name',
                    choices=['brats', 'mslub', 'msseg', 'wmh'])
parser.add_argument('--image_size', type=int, default=128, help='Image size')
parser.add_argument('--sequence', type=str, default='t1', help='MRI sequence')
parser.add_argument('--slice_range', type=int, nargs='+',
                    default=(55, 135), help='Lower and Upper slice index')
parser.add_argument('--normalize', type=bool, default=False,
                    help='Normalize images between 0 and 1')
parser.add_argument('--equalize_histogram', type=bool,
                    default=True, help='Equalize histogram')
parser.add_argument('--num_workers', type=int,
                    default=4, help='Number of workers')

# Logging settings
parser.add_argument('--log_frequency', type=int,
                    default=100, help='Logging frequency')
parser.add_argument('--val_frequency', type=int,
                    default=200, help='Validation frequency')
parser.add_argument('--save_frequency', type=int, default=200,
                    help='Model checkpointing frequency')
parser.add_argument('--val_steps', type=int, default=50,
                    help='Steps per validation')
parser.add_argument('--num_images_log', type=int,
                    default=10, help='Number of images to log')

# Hyperparameters
parser.add_argument('--lr_g', type=float, default=2e-4,
                    help='Generator learning rate')
parser.add_argument('--lr_d', type=float, default=2e-4,
                    help='Discriminator learning rate')
parser.add_argument('--lr_e', type=float, default=5e-5,
                    help='Encoder learning rate')
parser.add_argument('--gp_weight', type=float, default=10.,
                    help='Gradient penalty weight')
parser.add_argument('--feat_weight', type=float, default=1.,
                    help='Feature reconstruction weight during encoder training')
parser.add_argument('--weight_decay', type=float,
                    default=0.0, help='Weight decay')
parser.add_argument('--max_steps_gan', type=int,
                    default=10000, help='Number of training steps')
parser.add_argument('--max_steps_encoder', type=int,
                    default=10000, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

# Model settings
parser.add_argument('--latent_dim', type=int, default=128,
                    help='Size of the latent space')
parser.add_argument('--generator_hidden_dims', type=int, nargs='+',
                    default=[300, 200, 150, 100],
                    # default=[512, 512, 256, 128],
                    help='Generator hidden dimensions')
parser.add_argument('--discriminator_hidden_dims', type=int, nargs='+',
                    default=[100, 150, 200, 300],
                    # default=[128, 256, 512, 512],
                    help='Discriminator hidden dimensions')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--extractor_cnn_layers', type=str,
                    nargs='+', default=['layer1', 'layer2'])
parser.add_argument('--keep_feature_prop', type=float,
                    default=1.0, help='Proportion of ResNet features to keep')

args = parser.parse_args()

# Select training device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project="feature_autoencoder", entity="felix-meissen", config=args,
           mode="disabled" if args.debug else "online")
config = wandb.config


""""""""""""""""""""""""""""""" Reproducability """""""""""""""""""""""""""""""
seed_everything(config.seed)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


print("Initializing models...")
extractor = Extractor(cnn_layers=config.extractor_cnn_layers,
                      inp_size=config.image_size,
                      keep_feature_prop=config.keep_feature_prop).to(config.device)
extractor.eval()
config.feat_channels = extractor.c_feats
model = fAnoGAN(config).to(config.device)
wandb.watch(model)

# Init optimizers
optimizer_g = torch.optim.Adam(model.G.parameters(), lr=config.lr_g, betas=(0., 0.9),
                               weight_decay=config.weight_decay)
optimizer_d = torch.optim.Adam(model.D.parameters(), lr=config.lr_d, betas=(0., 0.9),
                               weight_decay=config.weight_decay)
optimizer_e = torch.optim.Adam(model.E.parameters(), lr=config.lr_e, betas=(0., 0.9),
                               weight_decay=config.weight_decay)

# Print model
print("Generator:")
summary(model.G, (config.latent_dim,))
print("\nDiscriminator:")
summary(model.D, (config.feat_channels, *[config.image_size // 4] * 2))
print("\nEncoder:")
summary(model.E, (config.feat_channels, *[config.image_size // 4] * 2))

if config.resume_path is not None:
    print("Loading model from checkpoint...")
    model.load(config.resume_path)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, test_loader = get_dataloaders(config)
print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""" GAN Training """""""""""""""""""""""""""""""""


def set_requires_grad(model, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def train_step_gan(model, optimizer_g, optimizer_d, x_real):
    model.train()

    # Generate fake images
    x_fake = model.G(batch_size=x_real.shape[0])

    """ 1. Train Discriminator, maximize log(D(x)) + log(1 - D(G(z))) """
    set_requires_grad(model.D, True)
    set_requires_grad(model.G, False)
    optimizer_d.zero_grad()

    # Discriminator loss (Wasserstein loss)
    loss_real = -model.D(x_real)[0].mean()
    loss_fake = model.D(x_fake.detach())[0].mean()
    adv_loss_d = loss_real + loss_fake

    # Gradient penalty
    loss_gp = calc_gradient_penalty(model.D, x_real, x_fake)

    # Combine losses and backward
    loss_D = adv_loss_d + config.gp_weight * loss_gp
    loss_D.backward()
    optimizer_d.step()

    """ 2. Train Generator, maximize log(D(G(z))) """
    set_requires_grad(model.D, False)
    set_requires_grad(model.G, True)
    optimizer_g.zero_grad()

    # Generator loss
    pred_fake = model.D(x_fake)[0]
    adv_loss_g = -pred_fake.mean()

    loss_G = adv_loss_g
    loss_G.backward()
    optimizer_g.step()

    return {
        'd_loss_real': loss_real.item(),
        'd_loss_fake': loss_fake.item(),
        'adv_loss_d': adv_loss_d.item(),
        'adv_loss_g': adv_loss_g.item(),
        'loss_gp': loss_gp.item(),
    }, x_fake


def train_gan(model, extractor, optimizer_g, optimizer_d, train_loader, config):
    print('Starting training GAN...')
    i_iter = 0
    i_epoch = 0

    train_losses = defaultdict(list)

    t_start = time()
    while True:
        for img_real in train_loader:
            i_iter += 1
            x_real = extractor(img_real.to(config.device))
            loss_dict, x_fake = train_step_gan(model, optimizer_g, optimizer_d,
                                               x_real)

            # Add to losses
            for k, v in loss_dict.items():
                train_losses[k].append(v)

            if i_iter % config.log_frequency == 0:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k,
                                      v in train_losses.items()])
                log_msg = f"Iteration {i_iter} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to tensorboard
                wandb.log(
                    {f'train/{k}': np.mean(v)
                     for k, v in train_losses.items()},
                    step=i_iter
                )
                wandb.log({
                    'train/GAN real images': wandb.Image(x_real.cpu()[:config.num_images_log].mean(1, keepdim=True)),
                    'train/GAN fake images': wandb.Image(x_fake.cpu()[:config.num_images_log].mean(1, keepdim=True)),
                }, step=i_iter)

                # Reset
                train_losses = defaultdict(list)

            # Save model weights
            if i_iter % config.save_frequency == 0:
                model.save('last_gan.pt')

            if i_iter >= config.max_steps_gan:
                print(
                    f'Reached {config.max_steps_gan} iterations. Finished training GAN.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


""""""""""""""""""""""""""""""" Encoder Training """""""""""""""""""""""""""""""


def train_step_encoder(model, optimizer_e, x, config):
    model.train()
    optimizer_e.zero_grad()

    z = model.E(x)  # encode image
    x_rec = model.G(z)  # decode latent vector
    x_feats = model.D.extract_feature(x)  # get features from real image
    # get features from reconstructed image
    x_rec_feats = model.D.extract_feature(x_rec)

    # Reconstruction loss
    # loss_img = F.mse_loss(x_rec, x)
    loss_img = SSIMLoss(size_average=True)(x_rec, x)
    loss_feats = F.mse_loss(x_rec_feats, x_feats)
    loss = loss_img + loss_feats * config.feat_weight

    loss.backward()
    optimizer_e.step()

    return {
        'loss_img_encoder': loss_img.item(),
        'loss_feats_encoder': loss_feats.item(),
        'loss_encoder': loss.item(),
    }, x_rec


def train_encoder(model, extractor, optimizer_e, train_loader, test_loader, config):
    print('Starting training Encoder...')
    i_iter = 0
    i_epoch = 0

    train_losses = defaultdict(list)

    # Generator and discriminator don't require gradients
    set_requires_grad(model.D, False)
    set_requires_grad(model.G, False)

    t_start = time()
    while True:
        for img in train_loader:
            i_iter += 1
            x = extractor(img.to(config.device))
            loss_dict, x_rec = train_step_encoder(
                model, optimizer_e, x, config)

            # Add to losses
            for k, v in loss_dict.items():
                train_losses[k].append(v)

            if i_iter % config.log_frequency == 0:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k,
                                      v in train_losses.items()])
                log_msg = f"Iteration {i_iter} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to tensorboard
                x = x.cpu().detach()
                x_rec = x_rec.cpu().detach()
                wandb.log(
                    {f'train/{k}': np.mean(v)
                     for k, v in train_losses.items()},
                    step=i_iter
                )
                wandb.log({
                    'train/input images': wandb.Image(x.cpu()[:config.num_images_log].mean(1, keepdim=True)),
                    'train/reconstructed images': wandb.Image(x_rec.cpu()[:config.num_images_log].mean(1, keepdim=True)),
                }, step=i_iter)

                # Reset
                train_losses = defaultdict(list)

            if i_iter % config.val_frequency == 0:
                validate_encoder(model, extractor, test_loader, i_iter, config)

            # Save model weights
            if i_iter % config.save_frequency == 0:
                model.save('last_encoder.pt')

            if i_iter >= config.max_steps_encoder:
                print(
                    f'Reached {config.max_steps_encoder} iterations. Finished training encoder.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


def val_step_encoder(model, extractor, img, config):
    model.eval()
    with torch.no_grad():
        x = extractor(img.to(config.device))
        z = model.E(x)  # encode image
        x_rec = model.G(z)  # decode latent vector
        x_feats = model.D.extract_feature(x)  # get features from real image
        # get features from reconstructed image
        x_rec_feats = model.D.extract_feature(x_rec)

        # Reconstruction loss
        # loss_img = F.mse_loss(x_rec, x)
        loss_img = SSIMLoss(size_average=True)(x_rec, x)
        loss_feats = F.mse_loss(x_rec_feats, x_feats)
        loss = loss_img + loss_feats * config.feat_weight

        loss_dict = {
            'loss_img_encoder': loss_img.item(),
            'loss_feats_encoder': loss_feats.item(),
            'loss_encoder': loss.item(),
        }

        # Anomaly map is the residual of the input and the reconstructed image
        # anomaly_map = (x - x_rec).abs().mean(1, keepdim=True)
        anomaly_map = SSIMLoss(size_average=False)(
            x_rec, x).mean(1, keepdim=True)
        anomaly_map = F.interpolate(anomaly_map, img.shape[-2:], mode='bilinear',
                                    align_corners=True)

        # Anomaly score
        # img_diff = (x - x_rec).pow(2).mean((1, 2, 3))
        img_diff = []
        for i in range(img.shape[0]):
            roi = anomaly_map[i][img[i] > 0]
            img_diff.append(roi.mean())
        img_diff = torch.stack(img_diff)
        feat_diff = (x_feats - x_rec_feats).pow(2).mean((1))
        anomaly_score = img_diff + config.feat_weight * feat_diff

    return loss_dict, anomaly_map.cpu(), anomaly_score.cpu(), x.cpu(), x_rec.cpu()


def validate_encoder(model, extractor, test_loader, i_iter, config):
    val_losses = defaultdict(list)
    pixel_aps = []
    labels = []
    anomaly_scores = []
    i_val_step = 0

    for img, y, label in test_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss_dict, anomaly_map, anomaly_score, x, x_rec = val_step_encoder(
            model, extractor, img, config)

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
        'val/input images': wandb.Image(x.cpu()[:config.num_images_log].mean(1, keepdim=True)),
        'val/reconstructed images': wandb.Image(x_rec.cpu()[:config.num_images_log].mean(1, keepdim=True)),
        'val/targets': wandb.Image(y.float().cpu()[:config.num_images_log]),
        'val/anomaly maps': wandb.Image(anomaly_map.cpu()[:config.num_images_log]),
    }, step=i_iter)


if __name__ == '__main__':
    # train_gan(model, extractor, optimizer_g, optimizer_d, train_loader, config)
    train_encoder(model, extractor, optimizer_e,
                  train_loader, test_loader, config)
    # validate_encoder(model, test_loader, config.max_steps_encoder + 1, config):
