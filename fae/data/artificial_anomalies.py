import random
import numpy as np
from skimage.draw import disk

from typing import Callable, Tuple


def sample_position(img: np.ndarray) -> Tuple[int, int]:
    """Sample a random position in the brain
    :param: img: The image to sample the position on
    :return: A tuple of the x,y position
    """
    obj_inds = np.where(img > 0)
    location_idx = random.randint(0, len(obj_inds[0]) - 1)
    position = (obj_inds[0][location_idx], obj_inds[1][location_idx])
    return position


def disk_anomaly(img: np.ndarray, position: Tuple[int, int], radius: int,
                 intensity: float) -> np.ndarray:
    """Draw a disk on a grayscale image.

    Args:
        img (np.ndarray): Grayscale image
        position (Tuple[int, int]): Position of disk
        radius (int): Radius of disk
        intensity (float): Intensity of pixels inside the disk
    Returns:
        disk_img (np.ndarray): img with ball drawn on it
        label (np.ndarray): target segmentation mask
    """
    assert img.ndim == 2, f"Invalid shape {img.shape}. Use a grayscale image"
    # Create disk
    rr, cc = disk(position, radius)
    rr = rr.clip(0, img.shape[0] - 1)
    cc = cc.clip(0, img.shape[1] - 1)

    # Draw disk on image
    disk_img = img.copy()
    disk_img[rr, cc] = intensity

    # Create label
    label = np.zeros(img.shape, dtype=np.uint8)
    label[rr, cc] = 1

    # Remove anomaly at background pixels
    mask = img > 0
    disk_img *= mask
    label *= mask

    return disk_img, label


def source_deformation_anomaly(img: np.ndarray, position: Tuple[int, int],
                               radius: int):
    """Pixels are shifted away from the center of the sphere.
    Args:
        img (np.ndarray): Image to be augmented, shape [h, w]
        position (Tuple[int int]): Center pixel of the mask
        radius (int): Radius
    Returns:
        img_deformed (np.ndarray): img with source deformation
        label (np.ndarray): target segmentation mask
    """
    # Create label mask
    rr, cc = disk(position, radius)
    rr = rr.clip(0, img.shape[0] - 1)
    cc = cc.clip(0, img.shape[1] - 1)
    label = np.zeros(img.shape, dtype=np.uint8)
    label[rr, cc] = 1

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


def sink_deformation_anomaly(img: np.ndarray, position: Tuple[int, int],
                             radius: int):
    """Pixels are shifted toward from the center of the sphere.
    Args:
        img (np.ndarray): Image to be augmented, shape [h, w]
        position (Tuple[int int]): Center pixel of the mask
        radius (int): Radius
    Returns:
        img_deformed (np.ndarray): img with sink deformation
        label (np.ndarray): target segmentation mask
    """
    # Create label mask
    rr, cc = disk(position, radius)
    rr = rr.clip(0, img.shape[0] - 1)
    cc = cc.clip(0, img.shape[1] - 1)
    label = np.zeros(img.shape, dtype=np.uint8)
    label[rr, cc] = 1

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


def pixel_shuffle_anomaly(img: np.ndarray, position: Tuple[int, int],
                          radius: int):
    """Pixels in the label mask are randomly shuffled
    Args:
        img (np.ndarray): Image to be augmented, shape [h, w]
        position (Tuple[int int]): Center pixel of the mask
        radius (int): Radius
    Returns:
        img_deformed (np.ndarray): img with sink deformation
        label (np.ndarray): target segmentation mask
    """
    # Create label mask
    rr, cc = disk(position, radius)
    rr = rr.clip(0, img.shape[0] - 1)
    cc = cc.clip(0, img.shape[1] - 1)
    label = np.zeros(img.shape, dtype=np.uint8)
    label[rr, cc] = 1

    # Remove anomaly at background pixels
    mask = img > 0
    label *= mask

    # Create copy of image for reference
    img_deformed = img.copy()

    # Create permutation of indices in label mask
    inds = np.where(label > 0)
    perm = np.random.permutation(len(inds[0]))
    inds_shuffled = [axis[perm] for axis in inds]

    # Apply permutation
    for x, y, x_, y_ in zip(*inds, *inds_shuffled):
        img_deformed[x, y] = img[x_, y_]

    return img_deformed, label


def random_anomaly(img: np.ndarray, radius_range: Tuple[int, int],
                   anomaly_fn: Callable):
    """Create an anomaly at a random location and size in the image.
    Args:
        img (np.ndarray): Image to be augmented, shape [h, w]
        radius_range Tuple[int, int]: Range to sample the radius from
        anomaly_fn (Callable): Anomaly function
    Returns:
        img_deformed (np.ndarray): img with sink deformation
        label (np.ndarray): target segmentation mask
    """
    position = sample_position(img)
    radius = np.random.randint(radius_range[0], radius_range[1] + 1)
    return anomaly_fn(img, position, radius)
