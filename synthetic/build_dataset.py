"""Build the synthetic datasets used to test the PCA baseline and the autoencoder.

  main.npz        N cases with all factors drawn independently (20 planes each)
  invariance.npz  M flow configurations, each acquired K times with different
                  acquisition settings (same underlying 3D flow), to test
                  whether features stay stable when only the acquisition changes

Each file holds X (cases, planes, 20, 16, 32, 3) float16 in m/s, the lumen
fraction on the grid, and a JSON table of input factors and per-plane geometry.
Run from the repository root:
    python synthetic/build_dataset.py --n 300 --m 40 --k 4
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.generator3d import simulate, draw_factors, ACQUISITION, RANGES

OUT = ROOT / "synthetic" / "data"


def save(name, X, M, F, P):
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", X=np.stack(X).astype(np.float16), M=np.stack(M).astype(np.float16))
    json.dump(dict(factors=F, planes=P), open(OUT / f"{name}.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    X, M, F, P = [], [], [], []
    t0 = time.time()
    for i in range(a.n):
        fct = draw_factors(rng)
        x, m, meta = simulate(fct, seed=int(rng.integers(1 << 31)))
        X.append(x); M.append(m); F.append(fct); P.append(meta)
        if (i + 1) % 25 == 0:
            print(f"main {i + 1}/{a.n}  {time.time() - t0:.0f} s", flush=True)
    save("main", X, M, F, P)
    X, M, F, P = [], [], [], []
    for j in range(a.m):
        base = draw_factors(rng)
        for k in range(a.k):
            fct = dict(base)
            for key in ACQUISITION:                                          # new acquisition, same flow
                lo, hi = RANGES[key]
                fct[key] = int(rng.integers(lo, hi + 1)) if key == "frames" else float(rng.uniform(lo, hi))
            fct["config"] = j
            x, m, meta = simulate(fct, seed=int(rng.integers(1 << 31)))
            X.append(x); M.append(m); F.append(fct); P.append(meta)
        if (j + 1) % 10 == 0:
            print(f"invariance {j + 1}/{a.m}  {time.time() - t0:.0f} s", flush=True)
    save("invariance", X, M, F, P)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
