from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Tuple

import numpy as np
from skimage import measure
from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve


CPUS = cpu_count()


def compute_average_precision(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Average Precision.

    :param preds: An array of predicted anomaly scores (shape: [N, 1, H, W]).
    :param targets: An array of ground truth labels (shape: [N, 1, H, W]).
    """
    preds, targets = np.array(preds), np.array(targets)
    # Check if targets are binary
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError('Targets must be binary')

    ap = average_precision_score(targets.reshape(-1), preds.reshape(-1))
    return ap


def compute_auroc(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve.

    :param preds: An array of predicted anomaly scores (shape: [N, 1, H, W]).
    :param targets: An array of ground truth labels (shape: [N, 1, H, W]).
    """
    preds, targets = np.array(preds), np.array(targets)
    # Check if targets are binary
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError('Targets must be binary')

    auc = roc_auc_score(targets.reshape(-1), preds.reshape(-1))
    return auc


def _compute_pro(pred_bin: np.ndarray, target: np.ndarray) -> float:
    # Find each connected gt region, compute the overlapped pixels between the gt region and predicted region
    image_pros = []
    label_map = measure.label(target, connectivity=2)
    props = measure.regionprops(label_map)
    for prop in props:
        # find the bounding box of an anomaly region
        x_min, y_min, x_max, y_max = prop.bbox
        cropped_pred_bin = pred_bin[x_min:x_max, y_min:y_max]
        cropped_target = prop.filled_image
        intersection = np.logical_and(
            cropped_pred_bin, cropped_target).astype(np.float32).sum()
        image_pros.append(intersection / prop.area)

    return np.mean(image_pros)


def compute_pro_auc(preds: np.ndarray, targets: np.ndarray,
                    max_fpr: float = 0.3, n_thresh: int = 100,
                    num_processes: int = 16) -> float:
    """Compute the normalized per-region-overlap (PRO-score).

    For each connected component within the ground truth, the relative overlap
    with the thresholded anomaly region is computed. We evaluate the PRO value
    for a large number of increasing thresholds until an average per-pixel
    false-positive rate of 30% for the entire dataset is reached and use the
    area under the PRO curve as a measure of anomaly detection performance.

    :param preds: An array of predicted anomaly scores (shape: [N, 1, H, W]).
    :param targets: An array of ground truth labels (shape: [N, 1, H, W]).
    :param max_fpr: Maximum false positive rate.
    :param n_threshs: Maximum number of thresholds to check.
    :param num_processes: Number of processes to use for multiprocessing.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Check if targets are binary
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError('Targets must be binary')

    # Squeeze away channel dimension
    preds = preds.squeeze(1)
    targets = targets.squeeze(1)

    pros_mean = []
    threds = []
    fprs = []

    for t in np.linspace(preds.max(), preds.min(), n_thresh):
        # Binarize predictions
        pred_bins = np.where(preds > t, 1, 0)

        # Compute false positive rate
        fpr = np.sum(pred_bins * (1 - targets)) / np.sum(1 - targets)
        if fpr > max_fpr:
            break

        with Pool(num_processes) as pool:
            # pro = pool.starmap(_compute_pro, [(pred_bin, target)
            #                                   for pred_bin, target in zip(pred_bins, targets)])
            pro = pool.starmap(_compute_pro, [(pred_bin, target) for pred_bin, target in zip(
                pred_bins, targets) if np.sum(target) > 0])

        pros_mean.append(np.mean(pro))
        threds.append(t)
        fprs.append(fpr)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)

    # rescale fpr [0, 0.3] -> [0, 1]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc_score = auc(fprs, pros_mean)

    return pro_auc_score


def compute_iou(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the Intersection over Union.

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Check if predictions and targets are binary
    if not np.all(np.logical_or(preds == 0, preds == 1)):
        raise ValueError('Predictions must be binary')
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError('Targets must be binary')

    # Compute IoU
    intersection = np.logical_and(preds, targets).sum()
    union = np.logical_or(preds, targets).sum()
    iou = intersection / union

    return iou


def compute_iou_at_nfpr(preds: np.ndarray, targets: np.ndarray,
                        max_fpr: float = 0.05) -> float:
    """
    Computes the Intersection over Union at 5% FPR.

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param max_fpr: Maximum false positive rate.
    :param n_thresh: Maximum number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Find threshold for 5% FPR
    fpr, _, thresholds = roc_curve(targets.reshape(-1), preds.reshape(-1))
    t = thresholds[max(0, fpr.searchsorted(max_fpr, 'right') - 1)]

    # Compute IoU
    return compute_iou(np.where(preds > t, 1, 0), targets)


def _iou_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                         threshold: float) -> float:
    return compute_iou(np.where(preds > threshold, 1, 0), targets)


def compute_best_iou(preds: np.ndarray, targets: np.ndarray,
                     n_thresh: float = 100,
                     num_processes: int = 4) -> Tuple[float, float]:
    """
    Compute the best dice score for n_thresh thresholds.

    :param predictions: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param n_thresh: Number of thresholds to check.
    :param num_processes: Number of processes to use for multiprocessing.
    """
    preds, targets = np.array(preds), np.array(targets)

    thresholds = np.linspace(preds.max(), preds.min(), n_thresh)

    with Pool(num_processes) as pool:
        fn = partial(_iou_multiprocessing, preds, targets)
        scores = pool.map(fn, thresholds)

    scores = np.stack(scores, 0)
    max_dice = scores.max()
    max_thresh = thresholds[scores.argmax()]
    return max_dice, max_thresh


def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the Sorensen-Dice coefficient:

    dice = 2 * TP / (2 * TP + FP + FN)

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Check if predictions and targets are binary
    if not np.all(np.logical_or(preds == 0, preds == 1)):
        raise ValueError('Predictions must be binary')
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError('Targets must be binary')

    # Compute Dice
    dice = 2 * np.sum(preds[targets == 1]) / \
        (np.sum(preds) + np.sum(targets))

    return dice


def compute_dice_at_nfpr(preds: np.ndarray, targets: np.ndarray,
                         max_fpr: float = 0.05) -> float:
    """
    Computes the Sorensen-Dice score at 5% FPR.

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param max_fpr: Maximum false positive rate.
    :param n_threshs: Maximum number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Find threshold for 5% FPR
    fpr, _, thresholds = roc_curve(targets.reshape(-1), preds.reshape(-1))
    t = thresholds[max(0, fpr.searchsorted(max_fpr, 'right') - 1)]
    print(f"Threshold at 5% FPR: {t}")

    # Compute Dice
    return compute_dice(np.where(preds > t, 1, 0), targets)


def _dice_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    return compute_dice(np.where(preds > threshold, 1, 0), targets)


def compute_best_dice(preds: np.ndarray, targets: np.ndarray,
                      n_thresh: float = 100,
                      num_processes: int = 4) -> Tuple[float, float]:
    """
    Compute the best dice score for n_thresh thresholds.

    :param predictions: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param n_thresh: Number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)

    thresholds = np.linspace(preds.max(), preds.min(), n_thresh)

    with Pool(num_processes) as pool:
        fn = partial(_dice_multiprocessing, preds, targets)
        scores = pool.map(fn, thresholds)

    scores = np.stack(scores, 0)
    max_dice = scores.max()
    max_thresh = thresholds[scores.argmax()]
    return max_dice, max_thresh


if __name__ == '__main__':
    from time import perf_counter
    anomaly_maps = np.load('./temp/anomaly_maps.npy')
    segs = np.load('./temp/segs.npy')
    size = (30000, 1, 128, 128)
    anomaly_maps = np.random.random(size)
    segs = np.random.randint(0, 2, size)  # Binary

    import torch
    anomaly_maps = torch.tensor(anomaly_maps)
    segs = torch.tensor(segs)

    # PRO-score
    start = perf_counter()
    print(f"PRO-score: {compute_pro_auc(anomaly_maps, segs)}")
    print(f"Time: {perf_counter() - start:.2f}s")

    # # AP
    # start = perf_counter()
    # print(f"AP: {compute_average_precision(anomaly_maps, segs)}")
    # print(f"Time: {perf_counter() - start:.2f}s")

    # # AUROC
    # start = perf_counter()
    # print(f"AUROC: {compute_auroc(anomaly_maps, segs)}")
    # print(f"Time: {perf_counter() - start:.2f}s")

    # # IOU at 5% FPR
    # start = perf_counter()
    # print(
    #     f"IoU at 5% FPR: {compute_iou_at_nfpr(anomaly_maps, segs, max_fpr=0.05)}")
    # print(f"Time: {perf_counter() - start:.2f}s")

    # # Best IOU
    # start = perf_counter()
    # print(f"Best IoU: {compute_best_iou(anomaly_maps, segs)}")
    # print(f"Time: {perf_counter() - start:.2f}s")

    # # Dice at 5% FPR
    # start = perf_counter()
    # print(
    #     f"Dice at 5% FPR: {compute_dice_at_nfpr(anomaly_maps, segs, max_fpr=0.05)}")
    # print(f"Time: {perf_counter() - start:.2f}s")

    # # Best Dice
    # start = perf_counter()
    # print(f"Best Dice: {compute_best_dice(anomaly_maps, segs)}")
    # print(f"Time: {perf_counter() - start:.2f}s")

    import IPython
    IPython.embed()
    exit(1)
