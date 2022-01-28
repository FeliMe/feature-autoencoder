import numpy as np
import torch

from sklearn.decomposition import PCA


def estimate_latent_channels(extractor, train_loader):
    """
    Estimate the number of latent channels for the Feature Autoencoder
    by performing a PCA over the features extracted from the train set.
    """
    device = next(extractor.parameters()).device
    feats = []
    i_samples = 0
    for i, normal_img in enumerate(train_loader):
        # Extract features
        with torch.no_grad():
            feat = extractor(normal_img.to(device))  # b, c, h, w
        # Reshape
        b, c = feat.shape[:2]
        feat = feat.permute(0, 2, 3, 1).reshape(-1, c)  # b*h*w, c
        # Add to feature tensor
        feats.append(feat.cpu().numpy())
        i_samples += b
        if i_samples > 20:
            break
    # Concatenate feats
    feats = np.concatenate(feats, axis=0)
    # Estimate parameters for mlp
    pca = PCA(n_components=0.9)
    pca.fit(feats)
    latent_channels, _ = pca.components_.shape
    return latent_channels
