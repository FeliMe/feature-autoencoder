from multiprocessing import Pool, cpu_count
from typing import Callable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.exposure import equalize_hist
from skimage.transform import resize


def load_nii(path: str, size: int = None, primary_axis: int = 0,
             dtype: str = "float32") -> Tuple[np.ndarray, np.ndarray]:
    """Load a neuroimaging file with nibabel, [w, h, slices]
    https://nipy.org/nibabel/reference/nibabel.html
    Args:
        path (str): Path to nii file
        size (int): Optional. Output size for h and w. Only supports rectangles
        primary_axis (int): Primary axis (the one to slice along, usually 2)
        dtype (str): Numpy datatype
    Returns:
        volume (np.ndarray): Of shape [w, h, slices]
        affine (np.ndarray): Affine coordinates (rotation and translation),
                             shape [4, 4]
    """
    # Load file
    data = nib.load(path, keep_file_open=False)
    volume = data.get_fdata(caching='unchanged')  # [w, h, slices]
    affine = data.affine

    # Squeeze optional 4th dimension
    if volume.ndim == 4:
        volume = volume.squeeze(-1)

    # Resize if size is given and if necessary
    if size is not None and (volume.shape[0] != size or volume.shape[1] != size):
        volume = resize(volume, [size, size, size])

    # Convert
    volume = volume.astype(np.dtype(dtype))

    # Move primary axis to first dimension
    volume = np.moveaxis(volume, primary_axis, 0)

    return volume, affine


def load_nii_nn(path: str, size: int = None,
                slice_range: Tuple[int, int] = None,
                normalize: bool = False,
                equalize_histogram: bool = False,
                dtype: str = "float32") -> np.ndarray:
    """
    Load a file for training. Slices should be first dimension, volumes are in
    MNI space and center cropped to the shorter side, then resized to size.

    Args:
        path: Path to nii file
        slice_range: Indices of lower and upper slices to return
        normalize: Normalize between 0 and 1
        equalize_histogram: Perform histogram equalization on the volume
        dtype:
    Returns:
        vol: The loaded and preprocessed volume
    """
    vol = load_nii(path, primary_axis=2, dtype=dtype)[0]

    if slice_range is not None:
        vol = vol[slice_range[0]:slice_range[1]]
        assert vol.shape[0] > 0

    vol = rectangularize(vol)

    if size is not None:
        vol = resize(vol, [vol.shape[0], size, size])

    if normalize:
        vol = normalize_percentile(vol, 98)

    if equalize_histogram:
        vol = histogram_equalization(vol)

    # Expand channel dimension
    vol = vol[:, None]

    return vol


def load_segmentation(path: str, size: int = None,
                      slice_range: Tuple[int, int] = None,
                      threshold: float = 0.4):
    """Load a segmentation file"""
    vol = load_nii_nn(path, size=size, slice_range=slice_range,
                      normalize=False, equalize_histogram=False)
    return np.where(vol > threshold, 1, 0)


def load_png(path: str, size: int = None,
             normalize: bool = False,
             equalize_histogram: bool = False,
             dtype: str = "float32"):
    """Load images that are stored as PNGs"""
    img = Image.open(path).convert("L")

    if size is not None and (img.size[0] != size or img.size[1] != size):
        img = img.resize((size, size))

    img = np.array(img, dtype=dtype)[None]

    if normalize:
        img = (img / 255).astype(img.dtype)

    if equalize_histogram and np.any(img > 0):
        img = histogram_equalization(img)

    return img


def load_files_to_ram(files: Sequence, load_fn: Callable = load_nii_nn,
                      num_processes: int = min(12, cpu_count())) -> List[np.ndarray]:
    with Pool(num_processes) as pool:
        results = pool.map(load_fn, files)
    return results


def histogram_equalization(img):
    mask = np.where(img > 0, 1, 0)  # Create equalization mask
    img = equalize_hist(img, nbins=256, mask=mask)  # Equalize
    img *= mask  # Assure that background is still 0
    return img


def normalize_percentile(img: np.ndarray, percentile: float = 98) -> np.ndarray:
    """Normalize an image to a percentile.
    Args:
        img (np.ndarray): Image to normalize
        percentile (float): Percentile to normalize to
    Returns:
        img (np.ndarray): Normalized image
    """
    # Get upper and lower bounds
    maxi = np.percentile(img, percentile)
    mini = np.min(img)
    # Normalize
    img = (img.astype(np.float32) - mini) / (maxi - mini)
    return img


def rectangularize(img: np.ndarray) -> np.ndarray:
    """
    Center crop the image to the shorter side

    Args:
        img (np.ndarray): Image to crop, shape [slices, w, h]
    Returns:
        img (np.ndarray): Cropped image
    """
    w, h = img.shape[1:]

    if w < h:
        # Center crop height to width
        img = img[:, :, (h - w) // 2:(h + w) // 2]
    elif h < w:
        # Center crop width to height
        img = img[:, (w - h) // 2:(w + h) // 2, :]

    return img
