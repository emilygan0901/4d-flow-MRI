"""Cross-fitted feature extraction on the synthetic data (PCA and autoencoder).

Cases (not planes) are split into 5 folds. For each fold and seed a model is
trained on the other 4 folds and features are extracted for the held-out
fold, so no case's features come from a model trained on it -- as planned for
the study. Fold models are aligned to the fold-0 model by orthogonal
Procrustes rotation on the invariance set, which plays the role of the open
reference data. The invariance set is also encoded by every model.

Training uses every second plane per case (3,000 planes) to fit CPU time;
features are extracted for all 20 planes.
Writes synthetic/data/features.npz.  Run:  python synthetic/train_features.py
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.models import AE, to_input, masked_mse, encode, recon_error, fit_pca, pca_encode, pca_recon_error

torch.set_num_threads(2)
DATA = ROOT / "synthetic" / "data"
LATENT, FOLDS, SEEDS, EPOCHS = 8, 5, 3, 40


def load(name):
    d = np.load(DATA / f"{name}.npz")
    X, M = d["X"], d["M"]
    n, p = X.shape[:2]
    return to_input(X.reshape(n * p, *X.shape[2:])), (M.reshape(n * p, 1, 16, 32) >= 0.5).astype(np.float32), n, p


def train(Xtr, Wtr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AE(LATENT, 32)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    steps = EPOCHS * ((len(Xtr) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    Xt, Wt = torch.from_numpy(Xtr), torch.from_numpy(Wtr)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]
            xh, _ = model(Xt[b]); loss = masked_mse(xh, Xt[b], Wt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    return model


def standardise(Z, ref):
    return (Z - ref.mean(0)) / (ref.std(0) + 1e-9)


def main():
    X, W, n, p = load("main")
    Xi, Wi, ni, pi = load("invariance")
    case = np.repeat(np.arange(n), p)
    plane = np.tile(np.arange(p), n)
    fold_of_case = np.random.default_rng(0).permutation(n) % FOLDS
    fold = fold_of_case[case]
    out = {k: np.zeros((SEEDS, n * p, LATENT), np.float32) for k in ("ae", "pca")}
    inv = {k: np.zeros((SEEDS, FOLDS, ni * pi, LATENT), np.float32) for k in ("ae", "pca")}
    err = {k: np.zeros((SEEDS, FOLDS)) for k in ("ae", "pca")}
    t0 = time.time()
    for s in range(SEEDS):
        ref = {}
        for k in range(FOLDS):
            tr = np.where((fold != k) & (plane % 2 == 0))[0]
            te = np.where(fold == k)[0]
            m = train(X[tr], W[tr], seed=100 * s + k)
            pc = fit_pca(X[tr], W[tr], LATENT, seed=s)
            for name, enc_te, enc_inv, e in (
                    ("ae", encode(m, X[te]), encode(m, Xi), recon_error(m, X[te], W[te])),
                    ("pca", pca_encode(pc, X[te], W[te]), pca_encode(pc, Xi, Wi), pca_recon_error(pc, X[te], W[te]))):
                # Procrustes: rotate this fold's latent space onto fold 0's, using the reference (invariance) set
                zi = standardise(enc_inv, enc_inv)
                if k == 0:
                    ref[name] = zi
                    R = np.eye(LATENT)
                else:
                    R, _ = orthogonal_procrustes(zi, ref[name])
                out[name][s, te] = standardise(enc_te, enc_inv) @ R
                inv[name][s, k] = zi @ R
                err[name][s, k] = e
            print(f"seed {s} fold {k}: AE err {err['ae'][s, k]:.6f}  PCA err {err['pca'][s, k]:.6f}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(DATA / "features.npz", ae=out["ae"], pca=out["pca"], inv_ae=inv["ae"], inv_pca=inv["pca"],
                        err_ae=err["ae"], err_pca=err["pca"], fold=fold, case=case, plane=plane)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
