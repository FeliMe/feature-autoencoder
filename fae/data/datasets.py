from functools import partial
from glob import glob
import os
import sys
from typing import List, Tuple, Sequence

import numpy as np
from torch.utils.data import Dataset, DataLoader

from fae import (
    CAMCANROOT,
    BRATSROOT,
    MOODROOT,
)
from fae.data.data_utils import (
    load_files_to_ram,
    load_nii_nn,
    load_segmentation,
)
from fae.data.artificial_anomalies import create_artificial_anomalies


def get_camcan_files(path: str = CAMCANROOT, sequence: str = "t1") -> List[str]:
    """Get all CamCAN files in a given sequence (t1, or t2).
    Args:
        path (str): Path to CamCAN root directory
        sequence (str): One of "t1", or "t2"
    Returns:
        files (List[str]): List of files
    """
    files = glob(os.path.join(
        path, f'normal/*/*{sequence.upper()}w_stripped_registered.nii.gz'))
    assert len(files) > 0, "No files found in CamCAN"
    return files


def get_brats_files(path: str = BRATSROOT, sequence: str = "t1") \
        -> Tuple[List[str], List[str]]:
    """Get all BRATS files in a given sequence (t1, t2, or flair).
    Args:
        path (str): Path to BRATS root directory
        sequence (str): One of "t1", "t2", or "flair"
    Returns:
        files (List[str]): List of files
        seg_files (List[str]): List of segmentation files
    """
    files = glob(os.path.join(path, 'MICCAI_BraTS2020_TrainingData/*',
                              f'*{sequence.lower()}*registered.nii.gz'))
    seg_files = [os.path.join(os.path.dirname(
        f), 'anomaly_segmentation.nii.gz') for f in files]
    assert len(files) > 0, "No files found in BraTS"
    return files, seg_files


def get_mood_train_files(path: str = MOODROOT, **kwargs) \
        -> Tuple[List[str], List[str]]:
    """Get MOOD training files.
    Args:
        path (str): Path to MOOD root directory
    Returns:
        files (List[str]): List of files
    """
    train_files = glob(os.path.join(path, 'brain/test_raw/*.nii.gz'))
    train_files.extend(glob(os.path.join(path, 'brain/train/*.nii.gz')))
    train_files = train_files[:round(len(train_files) * 0.8)]
    assert len(train_files) > 0, "No files found in MOOD"
    return train_files


def get_mood_val_test_files(path: str = MOODROOT, **kwargs) \
        -> Tuple[List[str], List[str]]:
    """Get MOOD validation and test files.
    Args:
        sequence (str): One of "t1", or "t2"
        path (str): Path to MOOD root directory
    Returns:
        files (List[str]): List of files
    """
    test_files = glob(os.path.join(path, 'brain/test_raw/*.nii.gz'))
    test_files.extend(glob(os.path.join(path, 'brain/train/*.nii.gz')))
    test_files = test_files[round(len(test_files) * 0.8):]
    assert len(test_files) > 0, "No files found in MOOD"
    return test_files, None


def load_images(files: List[str], config) -> np.ndarray:
    """Load images from a list of files.
    Args:
        files (List[str]): List of files
        config (Namespace): Configuration
    Returns:
        images (np.ndarray): Numpy array of images
    """
    load_fn = partial(load_nii_nn,
                      slice_range=config.slice_range,
                      size=config.image_size,
                      normalize=config.normalize,
                      equalize_histogram=config.equalize_histogram)
    return load_files_to_ram(files, load_fn)


def load_segmentations(seg_files: List[str], config) -> np.ndarray:
    """Load segmentations from a list of files.
    Args:
        seg_files (List[str]): List of files
        config (Namespace): Configuration
    Returns:
        segmentations (np.ndarray): Numpy array of segmentations
    """
    load_fn = partial(load_segmentation,
                      slice_range=config.slice_range,
                      size=config.image_size)
    return load_files_to_ram(seg_files, load_fn)


class TrainDataset(Dataset):
    """
    Training dataset. No anomalies, no segmentation maps.
    """

    def __init__(self, imgs: np.ndarray):
        """
        Args:
            imgs (np.ndarray): Training slices
        """
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]


class TestDataset(Dataset):
    """
    Test dataset. With real anomalies.
    """

    def __init__(self, imgs: np.ndarray, segs: np.ndarray):
        super().__init__()
        self.imgs = imgs
        self.segs = segs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]  # (c, h, w)
        seg = self.segs[idx]  # (1, h, w)
        label = np.where(seg.sum(axis=(1, 2)) > 0, 1, 0)  # (1,)
        return img, seg, label


def get_files(ds_name: str, sequence: str):
    if f"get_{ds_name}_files" in sys.modules[__name__].__dict__:
        get_files_fn = sys.modules[__name__].__dict__[
            f"get_{ds_name}_files"]
    else:
        raise ValueError(f'Dataset {ds_name} not found')

    return get_files_fn(sequence=sequence)


def val_test_split(files: Sequence, val_size: float, shuffle: bool = True) \
        -> Tuple[List, List]:
    """Split a list of files into validation and test sets"""
    if shuffle:
        np.random.shuffle(files)

    # Split
    val_size = int(len(files) * val_size)
    val_files = files[:val_size]
    test_files = files[val_size:]

    return val_files, test_files


def get_dataloaders(config):
    """Returns the train-, val- and testloader.
    Args:
        config (Namespace): Configuration
    Returns:
        train_loader (torch.utils.data.DataLoader): Training loader
        test_loader (torch.utils.data.DataLoader): Test loader
    """
    train_files = get_files(config.train_dataset, config.sequence)
    test_files, test_seg_files = get_files(
        config.test_dataset, config.sequence)

    print(f"Found {len(train_files)} training files")
    print(f"Found {len(test_files)} test files")

    print("Loading images...")
    if not config.train:
        train_imgs = np.random.randn(1000, 1, 128, 128)
    else:
        train_imgs = np.concatenate(load_images(train_files, config))
    test_imgs = np.concatenate(load_images(test_files, config))

    if "mood" in config.test_dataset:
        print("Creating artificial anomalies...")
        assert config.anomaly_name is not None
        test_imgs, test_segs = create_artificial_anomalies(
            test_imgs, config.anomaly_name, radius_range=config.anomaly_size)
    else:
        print("Loading segmentations...")
        test_segs = np.concatenate(load_segmentations(test_seg_files, config))

    # Split into validation and test sets
    val_size = int(len(test_imgs) * config.val_split)
    val_imgs = test_imgs[:val_size]
    test_imgs = test_imgs[val_size:]
    val_segs = test_segs[:val_size]
    test_segs = test_segs[val_size:]

    # Shuffle validation and test data
    val_perm = np.random.permutation(len(val_imgs))
    val_imgs = val_imgs[val_perm]
    val_segs = val_segs[val_perm]

    test_perm = np.random.permutation(len(test_imgs))
    test_imgs = test_imgs[test_perm]
    test_segs = test_segs[test_perm]

    # Create dataloaders
    train_loader = DataLoader(TrainDataset(train_imgs),
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(TestDataset(val_imgs, val_segs),
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)
    test_loader = DataLoader(TestDataset(test_imgs, test_segs),
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers)
    print(f"Len train_loader: {len(train_loader)}")
    print(f"Len val_loader: {len(val_loader)}")
    print(f"Len test_loader: {len(test_loader)}")

    return train_loader, val_loader, test_loader
