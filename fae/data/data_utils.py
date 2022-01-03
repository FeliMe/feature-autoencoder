from multiprocessing import Pool, cpu_count
from typing import Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.exposure import equalize_hist
from skimage.transform import resize


def load_nii(path: str, size: int = None, primary_axis: int = 0,
             dtype: str = "float32"):
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


def load_nii_nn(path: str, size: int = 224,
                slice_range: Tuple[int, int] = None,
                normalize: bool = False,
                equalize_histogram: bool = False,
                dtype: str = "float32"):
    """
    Load a file for training. Slices should be first dimension, volumes are in
    MNI space and center cropped to the shorter side, then resized to size
    """
    vol = load_nii(path, primary_axis=2, dtype=dtype)[0]

    if slice_range is not None:
        vol = vol[slice_range[0]:slice_range[1]]

    vol = rectangularize(vol)

    if size is not None:
        vol = resize(vol, [vol.shape[0], size, size])

    if normalize:
        vol = normalize_percentile(vol, 98)

    if equalize_histogram:
        vol = histogram_equalization(vol)

    return vol


def load_segmentation(path: str, size: int = 224,
                      slice_range: Tuple[int, int] = None,
                      threshold: float = 0.4):
    """Load a segmentation file"""
    vol = load_nii_nn(path, size=size, slice_range=slice_range,
                      normalize=False, equalize_histogram=False)
    return np.where(vol > threshold, 1, 0)


def load_files_to_ram(files: Sequence, load_fn: Callable = load_nii_nn,
                      num_processes: int = cpu_count()) -> List[np.ndarray]:
    pool = Pool(num_processes)
    results = []

    results = pool.map(load_fn, files)

    pool.close()
    pool.join()

    return results


def histogram_equalization(img):
    # Create equalization mask
    mask = np.where(img > 0, 1, 0)
    # Equalize
    img = equalize_hist(img, nbins=256, mask=mask)
    # Assure that background still is 0
    img *= mask

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
    # Get image shape
    w, h = img.shape[1:]

    if w < h:
        # Center crop height to width
        img = img[:, :, (h - w) // 2:(h + w) // 2]
    elif h < w:
        # Center crop width to height
        img = img[:, (w - h) // 2:(w + h) // 2, :]
    else:
        # No cropping
        pass

    return img


def train_val_split(files: Sequence, val_size: float):
    """Split a list of files into training and validation sets"""
    # Shuffle
    np.random.shuffle(files)

    # Split
    val_size = int(len(files) * val_size)
    train_files = files[val_size:]
    val_files = files[:val_size]

    return train_files, val_files


def show(imgs: List[np.ndarray], seg: List[np.ndarray] = None,
         path: str = None) -> None:

    if not isinstance(imgs, list):
        imgs = [imgs]
    n = len(imgs)
    fig = plt.figure()

    for i in range(n):
        fig.add_subplot(1, n, i + 1)
        plt.imshow(imgs[i], cmap="gray")

        if path is not None:
            plt.axis('off')

        if seg is not None:
            plt.imshow(seg[i], cmap="jet", alpha=0.3)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def volume_viewer(volume, initial_position=None, slices_first=True):
    """Plot a volume of shape [x, y, slices]
    Useful for MR and CT image volumes
    Args:
        volume (torch.Tensor or np.ndarray): With shape [slices, h, w]
        initial_position (list or tuple of len 3): (Optional)
        slices_first (bool): If slices are first or last dimension in volume
    """
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def previous_slice(ax):
        volume = ax.volume
        d = volume.shape[0]
        ax.index = (ax.index + 1) % d
        ax.images[0].set_array(volume[ax.index])
        ax.texts.pop()
        ax.text(5, 15, f"Slice: {d - ax.index}", color="white")

    def next_slice(ax):
        volume = ax.volume
        d = volume.shape[0]
        ax.index = (ax.index - 1) % d
        ax.images[0].set_array(volume[ax.index])
        ax.texts.pop()
        ax.text(5, 15, f"Slice: {d - ax.index}", color="white")

    def process_key(event):
        fig = event.canvas.figure
        # Move axial (slices)
        if event.key == 'k':
            next_slice(fig.axes[0])
        elif event.key == 'j':
            previous_slice(fig.axes[0])
        # Move coronal (h)
        elif event.key == 'u':
            previous_slice(fig.axes[1])
        elif event.key == 'i':
            next_slice(fig.axes[1])
        # Move saggital (w)
        elif event.key == 'h':
            previous_slice(fig.axes[2])
        elif event.key == 'l':
            next_slice(fig.axes[2])
        fig.canvas.draw()

    def prepare_volume(volume, slices_first):
        # Omit batch dimension
        if volume.ndim == 4:
            volume = volume[0]

        # If image is not loaded with slices_first, put slices dimension first
        if not slices_first:
            volume = np.moveaxis(volume, 2, 0)

        # Pad slices
        if volume.shape[0] < volume.shape[1]:
            pad_size = (volume.shape[1] - volume.shape[0]) // 2
            pad = [(0, 0)] * volume.ndim
            pad[0] = (pad_size, pad_size)
            volume = np.pad(volume, pad)

        # Flip directions for display
        volume = np.flip(volume, (0, 1, 2))

        return volume

    def plot_ax(ax, volume, index, title):
        ax.volume = volume
        shape = ax.volume.shape
        d = shape[0]
        ax.index = d - index
        aspect = shape[2] / shape[1]
        ax.imshow(ax.volume[ax.index], aspect=aspect)
        ax.set_title(title)
        ax.text(5, 15, f"Slice: {d - ax.index}", color="white")

    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'

    remove_keymap_conflicts({'h', 'j', 'k', 'l'})

    volume = prepare_volume(volume, slices_first)

    if initial_position is None:
        initial_position = np.array(volume.shape) // 2

    # Volume shape [slices, h, w]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    plot_ax(ax[0], np.transpose(volume, (0, 2, 1)), initial_position[2],
            "axial")  # axial [slices, h, w]
    plot_ax(ax[1], np.transpose(volume, (2, 0, 1)), initial_position[1],
            "coronal")  # saggital [h, slices, w]
    plot_ax(ax[2], np.transpose(volume, (1, 0, 2)), initial_position[0],
            "sagittal")  # coronal [w, slices, h]
    fig.canvas.mpl_connect('key_press_event', process_key)
    print("Plotting volume, navigate:"
          "\naxial with 'j', 'k'"
          "\ncoronal with 'u', 'i'"
          "\nsaggital with 'h', 'l'")
    plt.show()


if __name__ == '__main__':
    file = "/home/felix/datasets/CamCAN/normal/sub-CC110033/sub-CC110033_T1w_registered_stripped.nii.gz"
    vol = load_nii_nn(file, size=224)
    print(vol.shape)
    volume_viewer(vol)
