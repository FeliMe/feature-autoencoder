from functools import partial
from multiprocessing import Pool
import os
import random
import sys

import numpy as np
from skimage.draw import disk

from typing import Callable, Tuple, Optional


def sample_position(img: np.ndarray) -> Tuple[int, int]:
    """Sample a random position in the brain

    Args:
        img: The image to sample the position on
    Returns:
        position: A tuple of the x,y position
    """
    obj_inds = np.where(img > 0)
    location_idx = random.randint(0, len(obj_inds[0]) - 1)
    position = (obj_inds[-2][location_idx], obj_inds[-1][location_idx])
    return position


def intensity_anomaly(
    img: np.ndarray,
    position: Tuple[int, int],
    radius: int,
    intensity: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Pixels are assigned a (random) uniform intensity value.

    Args:
        img: Image to be augmented, shape [c, h, w]
        position: Center pixel of the mask
        radius: Radius
    Returns:
        img_deformed: img with (random) uniform intensity, shape [c, h, w]
        label: target segmentation mask, shape [1, h, w]
    """
    # Create label mask
    rr, cc = disk(position, radius)
    rr = rr.clip(0, img.shape[-2] - 1)
    cc = cc.clip(0, img.shape[-1] - 1)
    label = np.zeros(img.shape, dtype=np.uint8)
    label[..., rr, cc] = 1

    # Sample intensity
    if intensity is None:
        intensity = np.random.uniform(0, 1)

    # Create deformed image
    img_deformed = img.copy()
    img_deformed[..., rr, cc] = intensity

    return img_deformed, label


def source_deformation(img: np.ndarray, position: Tuple[int, int], radius: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Pixels are shifted away from the center of the sphere.

    Args:
        img: Image to be augmented, shape [c, h, w]
        position: Center pixel of the mask
        radius: Radius
    Returns:
        img_deformed: img with source deformation, shape [c, h, w]
        label: target segmentation mask, shape [1, h, w]
    """
    # Create label mask
    rr, cc = disk(position, radius)
    rr = rr.clip(0, img.shape[-2] - 1)
    cc = cc.clip(0, img.shape[-1] - 1)
    label = np.zeros(img.shape, dtype=np.uint8)
    label[..., rr, cc] = 1

    # Remove anomaly at background pixels
    mask = img > 0
    label *= mask

    # Center voxel of deformation
    C = np.array(position)

    # Create copy of image for reference
    img_deformed = img.copy()
    copy = img.copy()

    # Iterate over indices of all voxels in mask
    inds = np.where(label > 0)
    for x, y in zip(*inds[-2:]):
        # Voxel at current location
        I = np.array([x, y])

        # Source pixel shift
        s = np.square(np.linalg.norm(I - C, ord=2) / radius)
        V = np.round(C + s * (I - C)).astype(np.int)
        x_, y_ = V

        # Assure that z_, y_ and x_ are valid indices
        x_ = max(min(x_, img.shape[-1] - 1), 0)
        y_ = max(min(y_, img.shape[-2] - 1), 0)

        if img_deformed[..., x, y] > 0:
            img_deformed[..., x, y] = copy[..., x_, y_]

    return img_deformed, label


def sink_deformation(img: np.ndarray, position: Tuple[int, int], radius: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Pixels are shifted toward from the center of the sphere.

    Args:
        img: Image to be augmented, shape [c, h, w]
        position: Center pixel of the mask
        radius: Radius
    Returns:
        img_deformed: img with sink deformation, shape [c, h, w]
        label: target segmentation mask, shape [1, h, w]
    """
    # Create label mask
    rr, cc = disk(position, radius)
    rr = rr.clip(0, img.shape[-2] - 1)
    cc = cc.clip(0, img.shape[-1] - 1)
    label = np.zeros(img.shape, dtype=np.uint8)
    label[..., rr, cc] = 1

    # Remove anomaly at background pixels
    mask = img > 0
    label *= mask

    # Center voxel of deformation
    C = np.array(position)

    # Create copy of image for reference
    img_deformed = img.copy()
    copy = img.copy()

    # Iterate over indices of all voxels in mask
    inds = np.where(label > 0)
    for x, y in zip(*inds[-2:]):
        # Voxel at current location
        I = np.array([x, y])

        # Sink pixel shift
        s = np.square(np.linalg.norm(I - C, ord=2) / radius)
        V = np.round(I + (1 - s) * (I - C)).astype(np.int)
        x_, y_ = V

        # Assure that z_, y_ and x_ are valid indices
        x_ = max(min(x_, img.shape[-2] - 1), 0)
        y_ = max(min(y_, img.shape[-1] - 1), 0)

        if img_deformed[..., x, y] > 0:
            img_deformed[..., x, y] = copy[..., x_, y_]

    return img_deformed, label


def create_artificial_anomaly(img: np.ndarray, anomaly_fn: Callable,
                              radius_range: Optional[Tuple[int, int]] = None,
                              p: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Create an artificial anomaly on the image.

    Args:
        img: Image to be augmented, shape [c, h, w]
        anomaly_fn: Function that creates the anomaly
        radius_range: Range in which radius of anomaly is sampled uniformly
        p: Probability of creating an anomaly
    Returns:
        img_deformed: img with sink deformation, shape [c, h, w]
        label: target segmentation mask, shape [1, h, w]
    """
    # Sample position
    position = sample_position(img)

    # Sample radius
    if radius_range is None:
        radius_range = (round(img.shape[0] / 10), round(img.shape[0] / 5))
    radius = random.randint(*radius_range)

    if random.random() < p:
        img_deformed, label = anomaly_fn(img, position, radius)
    else:
        img_deformed = img.copy()
        label = np.zeros(img.shape, dtype=np.uint8)

    return img_deformed, label


def create_artificial_anomalies(imgs: np.ndarray, anomaly_name: str,
                                radius_range: Optional[Tuple[int, int]] = None,
                                num_processes: int = min(os.cpu_count(), 12)) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Create artificial anomalies on the images.

    Args:
        imgs: Images to be augmented, shape [n, c, h, w]
        radius_range: Range in which radius of anomaly is sampled uniformly
        num_processes: Number of threads used for multiprocessing
    Returns:
        imgs_deformed: imgs with sink deformation, shape [n, c, h, w]
        labels: target segmentation masks, shape [n, 1, h, w]
    """
    if anomaly_name in sys.modules[__name__].__dict__:
        anomaly_fn = sys.modules[__name__].__dict__[anomaly_name]
    else:
        raise ValueError(f'Anomaly function {anomaly_name} not found')

    map_fn = partial(create_artificial_anomaly, radius_range=radius_range,
                     p=0.5, anomaly_fn=anomaly_fn)
    with Pool(num_processes) as pool:
        results = pool.map(map_fn, imgs)

    imgs_deformed = np.array([r[0] for r in results])
    labels = np.array([r[1] for r in results])

    return imgs_deformed, labels
