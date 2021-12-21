import os
from glob import glob
from typing import List, Tuple

from fae import (
    CAMCANROOT,
    BRATSROOT,
    MSLUBROOT,
    MSSEGROOT,
    WMHROOT
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
                              f'*{sequence.upper()}w*registered_stripped.nii.gz'))
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
    seg_files = [os.path.join(os.path.dirname(f), 'anomal_segmentation.nii.gz') for f in files]
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
    seg_files = [os.path.join(os.path.dirname(f), 'anomal_segmentation.nii.gz') for f in files]
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
    seg_files = [os.path.join(os.path.dirname(f), 'anomal_segmentation.nii.gz') for f in files]
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
    seg_files = [os.path.join(os.path.dirname(f), 'anomal_segmentation.nii.gz') for f in files]
    return files, seg_files


if __name__ == "__main__":
    camcan_files = get_camcan_files()
    print(len(camcan_files))
    brats_files, brats_seg_files = get_brats_files()
    print(len(brats_files), len(brats_seg_files))
    mslub_files, mslub_seg_files = get_mslub_files()
    print(len(mslub_files), len(mslub_seg_files))
    msseg_files, msseg_seg_files = get_msseg_files()
    print(len(msseg_files), len(msseg_seg_files))
    wmh_files, wmh_seg_files = get_wmh_files()
    print(len(wmh_files), len(wmh_seg_files))
    import IPython; IPython.embed(); exit(1)
