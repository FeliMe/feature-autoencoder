"""Create artificial anomalies via patch interpolation or poisson image interpolation.
Adapted from: https://github.com/jemtan/FPI/blob/master/self_sup_task.py
         and: https://github.com/jemtan/PII/blob/main/poissonBlend.py"""

from typing import Callable, List, Tuple

import numpy as np
import scipy
from scipy.sparse.linalg import spsolve
import torch
from torch.utils.data import DataLoader

import fae.data.datasets as fae_datasets


def show(img):
    import matplotlib.pyplot as plt
    if img.ndim == 4:
        from torchvision.utils import make_grid
        img = make_grid(torch.tensor(img), nrow=8, padding=2)
    if img.ndim == 3:
        img = img[0]
    plt.imshow(img, cmap='gray')
    plt.show()


def sample_location(img: np.ndarray, core_percent: float = 0.8) -> np.ndarray:
    """
    Sample a location for the core of the patch.
    The location is an array of shape (2,) where N is the number
    of patches.

    :param img: image tensor of shape (H, W) or (C, H, W)
    :param core_percent: percentage of the core region
    :return: array of shape (2,)
    """

    dims = np.array(img.shape)
    core = core_percent * dims
    offset = (1 - core_percent) * dims / 2

    # Sample x-coordinates
    lo = int(np.floor(offset[-1]))
    hi = int(np.ceil(offset[-1] + core[-1]))
    cx = np.random.randint(lo, hi)

    # Sample y-coordinates
    lo = int(np.floor(offset[-2]))
    hi = int(np.ceil(offset[-2] + core[-2]))
    cy = np.random.randint(lo, hi)

    return np.array([cx, cy])


def sample_width(img: np.ndarray, min_p: float = 0.1, max_p: float = 0.4) -> float:
    """
    Sample a width for the patch.
    The width is a float between min_p and max_p of the image width

    :param img: image tensor of shape (H, W) or (C, H, W)
    :param min_p: minimum width percentage
    :param max_p: maximum width percentage
    :return: width
    """
    img_width = img.shape[-1]
    min_width = round(min_p * img_width)
    max_width = round(max_p * img_width)
    return np.random.randint(min_width, max_width)


def create_patch_mask(img: np.ndarray) -> np.ndarray:
    """
    Create a mask for the given image.
    The mask is a tensor of shape (C, H, W) where C is the number of channels,
    H is the height of the image and W is the width of the image.
    The mask is a binary tensor with values 0 and 1.
    The mask is 1 if the patch is inside the image and 0 otherwise.

    :param img: image tensor of shape (H, W) (C, H, W)
    :param patch_size: size of the patch
    :return: mask tensor of shape (H, W) or (C, H, W)
    """
    dims = img.shape

    # Center of the patch
    center = sample_location(img)

    # Width of the patch
    width = sample_width(img)

    # Compute patch coordinates
    coor_min = center - width // 2
    coor_max = center + width // 2

    # Clip coordinates to within image dims
    coor_min = np.clip(coor_min, 0, dims[-2:])
    coor_max = np.clip(coor_max, 0, dims[-2:])

    # Create mask
    mask = np.zeros(img.shape, dtype=np.float32)
    mask[:, coor_min[0]:coor_max[0], coor_min[1]:coor_max[1]] = 1

    return mask


def insert_laplacian_indexed(laplacian_op, mask: np.ndarray, central_val: float = 4):
    dims = np.shape(mask)  # (H, W) or (C, H, W)
    mask_flat = mask.flatten()
    inds = np.array((mask_flat > 0).nonzero())
    laplacian_op[..., inds, inds] = central_val
    laplacian_op[..., inds, inds + 1] = -1
    laplacian_op[..., inds, inds - 1] = -1
    laplacian_op[..., inds, inds + dims[-2]] = -1
    laplacian_op[..., inds, inds - dims[-2]] = -1
    laplacian_op = laplacian_op.tocsc()
    return laplacian_op


def pii(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs poisson image interpolation between two batches of images and
    returns the resulting images and the corresponding masks.

    :param img1: image tensor of shape (H, W) or (C, H, W)
    :param img2: image tensor of shape (H, W) or (C, H, W)
    :return: (img_pii, mask)
    """
    # Create mask
    patch_mask = create_patch_mask(img1)  # shape (H, W) or (C, H, W)
    interp = np.random.uniform(0.05, 0.95)
    mask = patch_mask * interp

    # Assure borders are 0 for poisson blending
    border_mask = np.zeros_like(mask)
    border_mask[..., 1:-1, 1:-1] = 1
    mask = border_mask * mask

    dims = np.shape(img1)
    clip_vals1 = (np.min(img1), np.max(img1))
    identity_matrix = scipy.sparse.identity(dims[-2] * dims[-1]).tolil()

    # Flatten images
    img1_flat = img1.flatten()  # (H, W) -> (H * W)
    img2_flat = img2.flatten()  # (H, W) -> (H * W)
    mask_flat = mask.flatten()  # (H, W) -> (H * W)

    # Discrete approximation of gradient
    grad_matrix = insert_laplacian_indexed(identity_matrix, mask, central_val=0)
    grad_matrix.eliminate_zeros()  # Get rid of central, only identity or neighbours
    grad_mask = grad_matrix != 0  # (H * W, H * W)

    img1_grad = grad_matrix.multiply(img1_flat)  # Negative neighbour values
    img1_grad = img1_grad + scipy.sparse.diags(img1_flat).dot(grad_mask)  # Add center value to sparse elements to get difference
    img2_grad = grad_matrix.multiply(img2_flat)
    img2_grad = img2_grad + scipy.sparse.diags(img2_flat).dot(grad_mask)

    # Mixing, favor the stronger gradient to improve blending
    alpha = np.max(mask_flat)
    img1_greater_mask = (1 - alpha) * np.abs(img1_grad) > alpha * np.abs(img2_grad)
    img1_guide = alpha * img2_grad - img1_greater_mask.multiply(alpha * img2_grad) + img1_greater_mask.multiply((1 - alpha) * img1_grad)

    img1_guide = np.squeeze(np.array(np.sum(img1_guide, 1)))
    img1_guide[mask_flat == 0] = img1_flat[mask_flat == 0]

    partial_laplacian = insert_laplacian_indexed(identity_matrix, mask,
                                                 central_val=4)
    x1 = spsolve(partial_laplacian, img1_guide)
    x1 = np.clip(x1, clip_vals1[0], clip_vals1[1])

    img_pii = np.reshape(x1, img1.shape)

    valid_label = (patch_mask * img1)[..., None] != (patch_mask * img2)[..., None]
    valid_label = np.any(valid_label, axis=-1)
    label = valid_label * mask

    return img_pii.astype(np.float32), label


def fpi(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs patch interpolation between two batches of images and returns
    the resulting images and the corresponding masks.

    :param img1: image tensor of shape (H, W) or (C, H, W)
    :param img2: image tensor of shape (H, W) or (C, H, W)
    :return: (patchex, label)
    """
    # Create mask
    patch_mask = create_patch_mask(img1)

    # Interpolation factor
    alpha = np.random.uniform(0.05, 0.95)

    # Create interpolation mask
    mask = patch_mask * alpha

    # Inverse mask
    mask_inv = patch_mask - mask
    zero_mask = 1 - patch_mask

    # Interpolate between patches
    patch_set = mask * img1 + mask_inv * img2
    img_fpi = img1 * zero_mask + patch_set

    # Create label masks
    valid_label = (patch_mask * img1)[..., None] != (patch_mask * img2)[..., None]
    valid_label = np.any(valid_label, axis=-1)
    label = valid_label * mask_inv

    return img_fpi, label


class PatchSwapDataset(torch.utils.data.Dataset):
    def __init__(self, volumes: List[np.ndarray], interp_fn: Callable):
        """
        :param volumes: list of volumes of shape [slices, 1, h, w]
        :param interp_fn: function to interpolate between two slices (fpi or pii)
        """
        self.interp_fn = interp_fn
        self.volumes = volumes
        self.n_vols = len(volumes)
        self.n_slices = len(volumes[0])

    def __len__(self):
        return self.n_vols * self.n_slices

    def convert_idx(self, idx):
        vol_idx = idx // self.n_slices
        slice_idx = idx % self.n_slices
        return vol_idx, slice_idx

    def __getitem__(self, idx):
        # Convert index to volume and slice index
        vol_idx, slice_idx = self.convert_idx(idx)

        # Get volume index of second volume
        vol_idx2 = torch.randint(0, self.n_vols - 1, (1,)).item()
        if vol_idx2 >= vol_idx:
            vol_idx2 += 1

        img1 = self.volumes[vol_idx][slice_idx]
        img2 = self.volumes[vol_idx2][slice_idx]

        # Patch swap
        patchex, label = self.interp_fn(img1, img2)

        return patchex, label


def get_dataloaders(config):
    """Returns the train-, val- and testloader.
    Args:
        config (Namespace): Configuration
    Returns:
        train_loader (torch.utils.data.DataLoader): Training loader
        test_loader (torch.utils.data.DataLoader): Test loader
    """
    def get_files(ds_name, sequence):
        if f"get_{ds_name}_files" in fae_datasets.__dict__:
            get_files_fn = fae_datasets.__dict__[f"get_{ds_name}_files"]
        else:
            raise ValueError(f'Dataset {ds_name} not found')

        return get_files_fn(sequence=sequence)

    train_files = get_files(config.train_dataset, config.sequence)
    test_files, test_seg_files = get_files(config.test_dataset, config.sequence)

    # Split into validation and test sets
    val_size = int(len(test_files) * config.val_split)
    val_files = test_files[:val_size]
    test_files = test_files[val_size:]
    val_seg_files = test_seg_files[:val_size]
    test_seg_files = test_seg_files[val_size:]

    print(f"Found {len(train_files)} training files files")
    print(f"Found {len(val_files)} validation files files")
    print(f"Found {len(test_files)} test files files")

    train_imgs = np.stack(fae_datasets.load_images(train_files, config))
    val_imgs = np.concatenate(fae_datasets.load_images(val_files, config))
    val_segs = np.concatenate(fae_datasets.load_segmentations(val_seg_files, config))
    test_imgs = np.concatenate(fae_datasets.load_images(test_files, config))
    test_segs = np.concatenate(fae_datasets.load_segmentations(test_seg_files, config))

    # Shuffle test data
    perm = np.random.permutation(len(test_imgs))
    test_imgs = test_imgs[perm]
    test_segs = test_segs[perm]

    train_loader = DataLoader(PatchSwapDataset(train_imgs, interp_fn=eval(config.interp_fn)),
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(fae_datasets.TestDataset(val_imgs, val_segs),
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)
    test_loader = DataLoader(fae_datasets.TestDataset(test_imgs, test_segs),
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers)
    print(f"Len train_loader: {len(train_loader)}")
    print(f"Len val_loader: {len(val_loader)}")
    print(f"Len test_loader: {len(test_loader)}")

    return train_loader, val_loader, test_loader
