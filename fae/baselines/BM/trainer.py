"""
Implementation of the Baseline Model from:
'Simple statistical methods for unsupervised brain anomaly detection on MRI are competitive to deep learning methods'
https://arxiv.org/pdf/2011.12735.pdf
"""
from argparse import ArgumentParser
from functools import partial
from typing import Tuple

import numpy as np

from fae.data.artificial_anomalies import create_artificial_anomalies
from fae.configs.base_config import base_parser
from fae.data.datasets import get_files
from fae.data.data_utils import load_files_to_ram, load_nii_nn, load_segmentation
from fae.utils import evaluation
from fae.utils.utils import seed_everything


""" Config """
parser = ArgumentParser(
    description="Arguments for training the Autoencoder baseline",
    parents=[base_parser],
    conflict_handler='resolve'
)
parser.add_argument('--return_volumes', type=bool, default=True)
config = parser.parse_args()

config.method = "BM"


seed_everything(config.seed)


def get_train_volumes(config):
    files = get_files(config.train_dataset, sequence=config.sequence)

    # Load all files
    volumes = load_files_to_ram(
        files,
        partial(load_nii_nn,
                size=config.image_size,
                slice_range=config.slice_range if 'slice_range' in config else None,
                normalize=config.normalize if 'normalize' in config else False,
                equalize_histogram=config.equalize_histogram if 'equalize_histogram' in config else False)
    )
    if "return_volumes" in config and config.return_volumes:
        return np.stack(volumes, axis=0)[:, :, 0]
    else:
        return np.concatenate(volumes, axis=0)[:, None]


def get_test_volumes(config):
    """Get all image slices and segmentations of the BraTS brain MRI dataset"""
    # Get all files
    files, seg_files = get_files(config.test_dataset, sequence=config.sequence)

    val_size = int(len(files) * config.val_split)
    files = files[val_size:]
    if "mood" not in config.test_dataset:
        seg_files = seg_files[val_size:]

    # Load all files
    volumes = load_files_to_ram(
        files,
        partial(load_nii_nn,
                size=config.image_size,
                slice_range=config.slice_range if 'slice_range' in config else None,
                normalize=config.normalize if 'normalize' in config else False,
                equalize_histogram=config.equalize_histogram if 'equalize_histogram' in config else False)
    )

    # Load all files
    if "mood" in config.test_dataset:
        volumes_ = []
        seg_volumes = []
        for volume in volumes:
            vol, seg_vol = create_artificial_anomalies(
                volume, config.anomaly_name, radius_range=config.anomaly_size)
            volumes_.append(vol)
            seg_volumes.append(seg_vol)
        volumes = volumes_
    else:
        print("Loading segmentations...")
        seg_volumes = load_files_to_ram(
            seg_files,
            partial(load_segmentation,
                    size=config.image_size,
                    slice_range=config.slice_range if 'slice_range' in config else None)
        )

    if "return_volumes" in config and config.return_volumes:
        imgs = np.stack(volumes, axis=0)[:, :, 0]
        segs = np.stack(seg_volumes, axis=0)[:, :, 0]
    else:
        imgs = np.concatenate(volumes, axis=0)[:, None]
        segs = np.concatenate(seg_volumes, axis=0)[:, None]

    return imgs, segs


def train(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the parameters for the distance (mu and std).
    :param x: Training data (shape: [n, c, h, w])
    """
    # Mean
    mu = x.mean(axis=0)

    # Standard deviation
    # ddof=1 to get unbiased std (divide by N-1)
    std = np.std(x, axis=0, ddof=1) + 1e-7

    return mu, std


def predict(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict the anomaly score for the test data.
    :param x: Test data (shape: [n, c, h, w])
    :param mu: Mean of the training data
    :param std: Standard deviation of the training data
    """
    anomaly_maps = np.absolute(x - mu) / std

    # Flatten slices
    anomaly_maps = anomaly_maps.reshape(-1, *anomaly_maps.shape[-2:])
    x = x.reshape(-1, *x.shape[-2:])

    # Compute the anomaly score
    anomaly_scores = np.stack([m[x_ > 0].mean()
                              for m, x_ in zip(anomaly_maps, x)])

    return anomaly_maps, anomaly_scores


def evaluate_pixel(anomaly_maps: np.ndarray, targets: np.ndarray) -> None:
    """
    Compute the mean pixel-wise average precision
    :param anomaly_maps: Anomaly maps (shape: [n, 1, h, w])
    :param targets: Ground truth segmentation maps (shape: [n, 1, h, w])
    """
    # Compute the pixel-wise average precision
    pixel_ap = evaluation.compute_average_precision(anomaly_maps, targets)
    pixel_auroc = evaluation.compute_auroc(anomaly_maps, targets)
    iou_at_5fpr = evaluation.compute_iou_at_nfpr(anomaly_maps, targets,
                                                 max_fpr=0.05)
    dice_at_5fpr = evaluation.compute_dice_at_nfpr(anomaly_maps, targets,
                                                   max_fpr=0.05)

    print(f"Pixel-wise AUROC: {np.mean(pixel_auroc):.4f}")
    print(f"Pixel-wise AP: {np.mean(pixel_ap):.4f}")
    print(f"Pixel-wise IoU at 5 % FPR: {np.mean(iou_at_5fpr):.4f}")
    print(f"Pixel-wise Dice at 5 % FPR: {np.mean(dice_at_5fpr):.4f}")


def evaluate_sample(anomaly_scores: np.ndarray, labels: np.ndarray) -> None:
    """
    Compute the sample-wise average precision and AUROC
    :param anomaly_scores: Anomaly maps (shape: [n, 1, h, w])
    :param labels: Ground truth segmentation maps (shape: [n, 1, h, w])
    """
    sample_auroc = evaluation.compute_auroc(anomaly_scores, labels)
    sample_ap = evaluation.compute_average_precision(anomaly_scores, labels)

    print(f"AUROC: {sample_auroc:.4f}")
    print(f"Sample-wise AP: {sample_ap:.4f}")


if __name__ == '__main__':
    # Load the data
    print("Loading data...")

    train_volumes = get_train_volumes(config)
    print(f"train_images.shape: {train_volumes.shape}")

    test_volumes, test_segs = get_test_volumes(config)
    label = np.where(test_segs.sum(axis=(2, 3)) > 0, 1, 0)
    print(f"test_images.shape: {test_volumes.shape}")

    # Flatten segmentation slices and labels
    test_segs = test_segs.reshape(-1, *test_segs.shape[-2:])
    label = label.reshape(-1)

    # Train the model
    print("Estimating the parameters...")
    mu, std = train(train_volumes)

    # Predict the anomaly score
    print("Predicting the anomaly maps and scores...")
    anomaly_maps, anomaly_scores = predict(test_volumes, mu, std)

    # Also flatten train_volumes
    train_volumes = train_volumes.reshape(-1, *train_volumes.shape[-2:])

    # Evaluate the model
    print("Evaluating...")
    evaluate_sample(anomaly_scores, label)
    evaluate_pixel(anomaly_maps, test_segs)
