from sklearn.metrics import average_precision_score, roc_auc_score


def compute_average_precision(predictions, targets):
    """Compute Average Precision
    Args:
        predictions (torch.Tensor): Anomaly scores
        targets (torch.Tensor): Segmentation map or target label, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AP must be binary")
    ap = average_precision_score(targets.reshape(-1), predictions.reshape(-1))
    return ap


def compute_auroc(predictions, targets) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve
    Args:
        predictions (torch.Tensor): Anomaly scores
        targets (torch.Tensor): Segmentation map or target label, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AUROC must be binary")
    auc = roc_auc_score(targets.reshape(-1), predictions.reshape(-1))
    return auc
