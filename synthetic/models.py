"""PCA baseline and compact convolutional autoencoder for 16 x 32 x 20-phase flow planes.

Input per plane: (20 phases x 3 components) = 60 channels on the 16 (radius) x
32 (angle) polar grid, in m/s with a single global scale. Convolutions wrap
around in angle (circular padding) and are zero-padded in radius. The loss is
mean squared error over lumen points only (lumen fraction >= 0.5).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

SCALE = 1.0   # m/s, single global factor (keeps differences in flow magnitude)


def to_input(X):
    """(n, 20, 16, 32, 3) -> (n, 60, 16, 32) float32."""
    X = np.asarray(X, np.float32) / SCALE
    return np.ascontiguousarray(np.moveaxis(X, 4, 1).reshape(X.shape[0], 60, X.shape[2], X.shape[3]))


class PolarConv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=0)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")      # angle wraps around
        x = F.pad(x, (0, 0, 1, 1))                       # radius: zero padding
        return self.conv(x)


class AE(nn.Module):
    def __init__(self, latent=8, width=64):
        super().__init__()
        self.enc = nn.Sequential(PolarConv(60, width), nn.GELU(),
                                 PolarConv(width, width, 2), nn.GELU(),      # 8 x 16
                                 PolarConv(width, width, 2), nn.GELU(),      # 4 x 8
                                 PolarConv(width, 32, 2), nn.GELU())         # 2 x 4
        self.to_z = nn.Linear(32 * 2 * 4, latent)
        self.from_z = nn.Linear(latent, 32 * 2 * 4)
        self.dec = nn.Sequential(nn.Upsample(scale_factor=2), PolarConv(32, width), nn.GELU(),
                                 nn.Upsample(scale_factor=2), PolarConv(width, width), nn.GELU(),
                                 nn.Upsample(scale_factor=2), PolarConv(width, width), nn.GELU(),
                                 PolarConv(width, 60))

    def encode(self, x):
        return self.to_z(self.enc(x).flatten(1))

    def forward(self, x):
        z = self.encode(x)
        return self.dec(self.from_z(z).view(-1, 32, 2, 4)), z


def masked_mse(xhat, x, w):
    """w: (n, 1, 16, 32) lumen weight."""
    return (((xhat - x) ** 2) * w).sum() / (w.sum() * x.shape[1] + 1e-9)


def train_ae(Xtr, Wtr, latent=8, seed=0, epochs=40, batch=128, lr=1e-3, Xval=None, Wval=None, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AE(latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Wt = torch.from_numpy(Xtr), torch.from_numpy(Wtr)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            xhat, _ = model(Xt[b])
            loss = masked_mse(xhat, Xt[b], Wt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def encode(model, X, batch=512):
    return np.concatenate([model.encode(torch.from_numpy(X[i:i + batch])).numpy() for i in range(0, len(X), batch)])


@torch.no_grad()
def recon_error(model, X, W, batch=512):
    num = den = 0.0
    for i in range(0, len(X), batch):
        x, w = torch.from_numpy(X[i:i + batch]), torch.from_numpy(W[i:i + batch])
        xh, _ = model(x)
        num += float((((xh - x) ** 2) * w).sum()); den += float(w.sum() * x.shape[1])
    return num / den


def fit_pca(Xtr, Wtr, k=8, seed=0):
    p = PCA(k, random_state=seed)
    p.fit((Xtr * Wtr).reshape(len(Xtr), -1))
    return p


def pca_encode(p, X, W):
    return p.transform((X * W).reshape(len(X), -1))


def pca_recon_error(p, X, W):
    Z = pca_encode(p, X, W)
    Xh = p.inverse_transform(Z).reshape(X.shape)
    return float((((Xh - X * W) ** 2) * W).sum() / (W.sum() * X.shape[1]))
