from functools import partial
from glob import glob
import os
import sys
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader

from fae import (
    CAMCANROOT,
    BRATSROOT,
    MSLUBROOT,
    MSSEGROOT,
    WMHROOT
)
from fae.data.artificial_anomalies import (
    source_deformation_anomaly,
    sink_deformation_anomaly,
    pixel_shuffle_anomaly,
    random_anomaly
)
from fae.data.data_utils import (
    load_files_to_ram,
    load_nii_nn,
    load_segmentation,
    train_val_split,
)


def get_camcan_files(path: str = CAMCANROOT, sequence: str = "t1") -> List[str]:
    """Get all CamCAN files in a given sequence (t1, or t2).
    Args:
        path (str): Path to CamCAN root directory
        sequence (str): One of "t1", or "t2"
    Returns:
        files (List[str]): List of files
    """
    files = glob(os.path.join(path, 'normal/*/',
                              f'*{sequence.upper()}w_stripped_registered.nii.gz'))
    assert len(files) > 0, "No files found in CamCAN"
    return files


def get_brats_files(path: str = BRATSROOT, sequence: str = "t1") -> Tuple[List[str], List[str]]:
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
    seg_files = [os.path.join(os.path.dirname(f), 'anomaly_segmentation.nii.gz') for f in files]
    assert len(files) > 0, "No files found in BraTS"
    return files, seg_files


def get_mslub_files(path: str = MSLUBROOT, sequence: str = "t1") -> Tuple[List[str], List[str]]:
    """Get all MSLUB files in a given sequence (t1, t2, or flair).
    Args:
        path (str): Path to MSLU-B root directory
        sequence (str): One of "t1", "t2", or "flair"
    Returns:
        files (List[str]): List of files
        seg_files (List[str]): List of segmentation files
    """
    if sequence.lower().startswith('t'):
        sequence += 'w'
    files = glob(os.path.join(path, 'lesion/*',
                              f'{sequence.upper()}*stripped_registered.nii.gz'))
    seg_files = [os.path.join(os.path.dirname(f), 'anomaly_segmentation.nii.gz') for f in files]
    assert len(files) > 0, "No files found in MSLUB"
    return files, seg_files


def get_msseg_files(path: str = MSSEGROOT, sequence: str = "t2") -> Tuple[List[str], List[str]]:
    """Get all MSSEG files in a given sequence (t2, or flair).
    Args:
        path (str): Path to MSSEG root directory
        sequence (str): One of "t2", or "flair"
    Returns:
        files (List[str]): List of files
        seg_files (List[str]): List of segmentation files
    """
    files = glob(os.path.join(path, 'training/training*/training*/',
                              f'{sequence.lower()}*registered.nii'))
    seg_files = [os.path.join(os.path.dirname(f), 'anomaly_segmentation.nii.gz') for f in files]
    assert len(files) > 0, "No files found in MSSEG2015"
    return files, seg_files


def get_wmh_files(path: str = WMHROOT, sequence: str = "t1") -> Tuple[List[str], List[str]]:
    """Get all WMH files in a given sequence (t1, or flair).
    Args:
        path (str): Path to WMH root directory
        sequence (str): One of "t1", or "flair"
    Returns:
        files (List[str]): List of files
        seg_files (List[str]): List of segmentation files
    """
    files = glob(os.path.join(path, '*/*/orig/',
                              f'{sequence.upper()}*stripped_registered.nii.gz'))
    seg_files = [os.path.join(os.path.dirname(f), 'anomaly_segmentation.nii.gz') for f in files]
    assert len(files) > 0, "No files found in WMH"
    return files, seg_files


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


class ValidationDataset(Dataset):
    """
    Validation dataset. With artificial anomalies.
    """
    def __init__(self, imgs: np.ndarray, anomaly_size: Tuple[int, int]):
        super().__init__()
        self.imgs = imgs
        self.anomaly_size = anomaly_size

    def __len__(self):
        return len(self.imgs)

    def create_anomaly(self, img: np.ndarray):
        anomaly_fn = np.random.choice([source_deformation_anomaly, sink_deformation_anomaly,
                                       pixel_shuffle_anomaly])
        return random_anomaly(img, radius_range=self.anomaly_size,
                              anomaly_fn=anomaly_fn)

    def __getitem__(self, idx):
        img, seg = self.create_anomaly(self.imgs[idx])
        return [img, seg]


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
        return [self.imgs[idx], self.segs[idx]]


def get_dataloaders(config):
    """Returns the train-, val- and testloader.
    Args:
        config (Namespace): Configuration
    Returns:
        train_loader (torch.utils.data.DataLoader): Training loader
        val_loader (torch.utils.data.DataLoader): Validation loader
        test_loader (torch.utils.data.DataLoader): Test loader
    """
    def get_files(ds_name, sequence):
        if f"get_{ds_name}_files" in sys.modules[__name__].__dict__:
            get_files_fn = sys.modules[__name__].__dict__[f"get_{ds_name}_files"]
        else:
            raise ValueError(f'Dataset {ds_name} not found')

        return get_files_fn(sequence=sequence)

    train_files = get_files(config.train_dataset, config.sequence)
    train_files, val_files = train_val_split(train_files, config.val_split)
    test_files, test_seg_files = get_files(config.test_dataset, config.sequence)

    train_imgs = np.concatenate(load_images(train_files, config))[:, None]
    val_imgs = np.concatenate(load_images(val_files, config))[:, None]
    test_imgs = np.concatenate(load_images(test_files, config))[:, None]
    test_segs = np.concatenate(load_segmentations(test_seg_files, config))[:, None]

    # Shuffle test data
    perm = np.random.permutation(len(test_imgs))
    test_imgs = test_imgs[perm]
    test_segs = test_segs[perm]

    train_loader = DataLoader(TrainDataset(train_imgs),
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(ValidationDataset(val_imgs, config.anomaly_size),
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


if __name__ == "__main__":
    from argparse import Namespace
    from fae.data.data_utils import show
    config = Namespace()
    config.slice_range = (55, 135)
    config.image_size = 224
    config.normalize = True
    config.equalize_histogram = False
    config.batch_size = 32

    # Test finding files
    camcan_files = get_camcan_files(sequence='t1')
    print(len(camcan_files))

    brats_files, brats_seg_files = get_brats_files(sequence='t1')
    print(len(brats_files), len(brats_seg_files))

    mslub_files, mslub_seg_files = get_mslub_files(sequence='t1')
    print(len(mslub_files), len(mslub_seg_files))

    msseg_files, msseg_seg_files = get_msseg_files(sequence='t2')
    print(len(msseg_files), len(msseg_seg_files))

    wmh_files, wmh_seg_files = get_wmh_files(sequence='t1')
    print(len(wmh_files), len(wmh_seg_files))

    # Test training dataset
    imgs = np.concatenate(load_images(camcan_files[:10], config))[:, None]
    train_ds = TrainDataset(imgs[40:])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size)
    x = next(iter(train_loader))
    print(x.shape)
    show(x[0, 0])

    # Test validation dataset
    imgs = np.concatenate(load_images(camcan_files[:10], config))[:, None]
    print(imgs.shape)
    val_ds = ValidationDataset(imgs[40:], (5, 50))
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    x, y = next(iter(val_loader))
    print(x.shape, y.shape)
    show([x[0, 0], y[0, 0]])

    # Test test dataset
    imgs = np.concatenate(load_images(brats_files[:10], config))[:, None]
    segs = np.concatenate(load_segmentations(brats_seg_files[:10], config))[:, None]
    test_ds = TestDataset(imgs[40:], segs[40:])
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)
    x, y = next(iter(test_loader))
    print(x.shape, y.shape)
    show([x[0, 0], y[0, 0]])
