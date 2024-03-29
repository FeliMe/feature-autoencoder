import os
from argparse import ArgumentParser
from collections import defaultdict
from time import time
from warnings import warn

import numpy as np
import torch
import wandb

from fae import WANDBNAME, WANDBPROJECT, WANDBDIR
from fae.configs.base_config import base_parser
from fae.data import datasets
from fae.models import models
from fae.utils.utils import seed_everything
from fae.utils import evaluation


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser(
    description="Arguments for training the Feature Autoencoder",
    parents=[base_parser],
    conflict_handler='resolve'
)
config = parser.parse_args()
config.method = "FAE"

if not config.train and config.resume_path is None:
    warn("Testing untrained model")

# Select training device
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'


""""""""""""""""""""""""""""""" Reproducibility """""""""""""""""""""""""""""""
seed_everything(config.seed)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


def get_model(config):
    if config.model in models.__dict__:
        model_cls = models.__dict__[config.model]
    else:
        raise ValueError(f'Model {config.model} not found')

    return model_cls(config)


print("Initializing model...")
model = get_model(config).to(config.device)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                             weight_decay=config.weight_decay)
# Print model
print(model.ae.enc)
print(model.ae.dec)

if config.resume_path is not None:
    print("Loading model from checkpoint...")
    model.load(config.resume_path)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, val_loader, test_loader = datasets.get_dataloaders(config)
print(f'Loaded {config.train_dataset} and {config.test_dataset} in '
      f'{time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""""""" W&B """""""""""""""""""""""""""""""""""""

wandb_dir = f"{WANDBDIR}/fae/{config.method}"
os.makedirs(wandb_dir, exist_ok=True)
wandb.init(project=WANDBPROJECT, entity=WANDBNAME, config=config,
           mode="disabled" if config.debug else "online",
           dir=wandb_dir)
wandb.watch(model)


""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train_step(model, optimizer, x, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    loss_dict = model.loss(x)
    loss = loss_dict['rec_loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def train(model, optimizer, train_loader, val_loader, config):
    print('Starting training...')
    i_iter = 0
    i_epoch = 0

    train_losses = defaultdict(list)

    t_start = time()
    while True:
        for x in train_loader:
            i_iter += 1
            loss_dict = train_step(model, optimizer, x, config.device)

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

                # Log to w&b
                wandb.log({
                    f'train/{k}': np.mean(v) for k, v in train_losses.items()
                }, step=i_iter)

                # Reset
                train_losses = defaultdict(list)

            if i_iter % config.val_frequency == 0:
                validate(model, val_loader, config.device, i_iter)

            # Save model weights
            if i_iter % config.save_frequency == 0:
                model.save(config, 'last.pt')

            if i_iter >= config.max_steps:
                print(
                    f'Reached {config.max_steps} iterations. Finished training.')

                # Final validation
                print("Final validation...")
                validate(model, val_loader, config.device, i_iter)
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


def val_step(model, x, device):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        loss_dict = model.loss(x)
        anomaly_map, anomaly_score = model.predict_anomaly(x)
    return loss_dict, anomaly_map.cpu(), anomaly_score.cpu()


def validate(model, val_loader, device, i_iter):
    val_losses = defaultdict(list)
    pixel_aps = []
    labels = []
    anomaly_scores = []
    i_val_step = 0

    for x, y, label in val_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss_dict, anomaly_map, anomaly_score = val_step(model, x, device)

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


def test(model, test_loader, config):
    val_losses = defaultdict(list)
    labels = []
    anomaly_scores = []
    segs = []
    anomaly_maps = []

    for x, y, label in test_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss_dict, anomaly_map, anomaly_score = val_step(
            model, x, config.device)

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
    iou_at_5fpr = evaluation.compute_iou_at_nfpr(anomaly_maps, segs,
                                                 max_fpr=0.05)
    dice_at_5fpr = evaluation.compute_dice_at_nfpr(anomaly_maps, segs,
                                                   max_fpr=0.05)

    # Print test results
    print("\nTest results:")
    log_msg = " - ".join([f'val {k}: {np.mean(v):.4f}' for k,
                         v in val_losses.items()])
    log_msg += f"\nanomaly-score min/max: {anomaly_scores.min():.4f}/{anomaly_scores.max():.4f}"
    log_msg += f"\npixel-wise average precision: {pixel_ap:.4f}\n"
    log_msg += f"IoU @ 5% fpr: {iou_at_5fpr:.4f} - "
    log_msg += f"Dice @ 5% fpr: {dice_at_5fpr:.4f}\n"
    log_msg += f"sample-wise AUROC: {sample_auroc:.4f} - "
    log_msg += f"sample-wise average precision: {sample_ap:.4f} - "
    log_msg += f"Average positive pixel: {torch.tensor(segs).float().mean():.4f}\n"
    log_msg += f"Average positive label: {torch.tensor(labels).float().mean():.4f}\n"
    print(log_msg)

    # Log to tensorboard
    wandb.log({
        f'val/{k}': np.mean(v) for k, v in val_losses.items()
    }, step=config.max_steps + 1)
    wandb.log({
        'val/pixel-ap': pixel_ap,
        'val/sample-ap': sample_ap,
        'val/sample-auroc': sample_auroc,
        'val/iou-at-5fpr': iou_at_5fpr,
        'val/dice-at-5fpr': dice_at_5fpr,
        'val/input images': wandb.Image(x.cpu()[:config.num_images_log]),
        'val/targets': wandb.Image(y.float().cpu()[:config.num_images_log]),
        'val/anomaly maps': wandb.Image(anomaly_map.cpu()[:config.num_images_log]),
    }, step=config.max_steps + 1)


@torch.no_grad()
def test_pitfalls(model, config):
    import random
    from glob import glob
    from tqdm import tqdm
    from functools import partial
    import torch.nn.functional as F
    from fae.data.data_utils import load_files_to_ram, load_nii_nn
    from fae.data.artificial_anomalies import sample_position, intensity_anomaly
    files = glob('/datasets/MOOD/brain/test_raw/*.nii.gz')
    load_fn = partial(load_nii_nn, slice_range=(128, 129), size=config.image_size)
    imgs = load_files_to_ram(files, load_fn)
    imgs = np.stack([s for vol in imgs for s in vol], axis=0)

    radius = 10
    intensities = np.linspace(0., 1., num=100)

    ap_results = []
    for intensity in tqdm(intensities):
        aps = []
        random.seed(0)

        for img in imgs:
            position = sample_position(img)
            img_anomal, label = intensity_anomaly(img, position, radius, intensity)

            img_ = torch.tensor(img[None]).to(config.device)
            img_anomal_ = torch.tensor(img_anomal[None]).to(config.device)

            # Experiment 3.1
            feats, rec = model(img_)
            feats_anomal, rec_anomal = model(img_anomal_)
            pred = model.loss_fn(rec, feats_anomal).mean(1, keepdim=True)
            pred = F.interpolate(pred, img_.shape[-2:], mode='bilinear',
                                 align_corners=True)
            pred = pred[0, 0].cpu().numpy()

            # Experiment 3.2
            # pred = model.predict_anomaly(img_anomal_)[0][0].detach().cpu().numpy()

            # Compute average precision
            ap = evaluation.compute_average_precision(pred, label)
            aps.append(ap)

        ap_results.append(np.mean(aps))
        print(f'Intensity: {intensity:.4f} - AP: {ap_results[-1]:.4f}')

    ap_results = np.array(ap_results)
    np.save('ex3_1_mood_aps.npy', ap_results)


if __name__ == '__main__':
    # Training
    if config.train:
        train(model, optimizer, train_loader, val_loader, config)

    # test_pitfalls(model, config)

    # Testing
    print('Testing...')
    test(model, test_loader, config)
