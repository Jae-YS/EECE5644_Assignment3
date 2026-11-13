"""Q2: GMM model-order selection with 10-fold CV log-likelihood, repeated many times."""
import sys, warnings, numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

QUICK = "--quick" in sys.argv
REPS = 5 if QUICK else 100
SIZES = [10, 100, 1000]
KMAX, FOLDS = 10, 10
rng = np.random.default_rng(1)

# ---- True GMM: components 0 and 1 overlap heavily (mean distance ~ comparable to covariance eigenvalues)
alpha = np.array([0.30, 0.30, 0.20, 0.20])
mu = np.array([[0, 0], [2.0, 1.0], [-5, 5], [6, -4]], float)
Sig = np.array([[[2.0, 0.8], [0.8, 1.5]],
                [[1.5, -0.6], [-0.6, 2.0]],
                [[1.0, 0.0], [0.0, 2.5]],
                [[2.5, 1.0], [1.0, 1.0]]])
for i in (0, 1):
    print(f"comp {i} eigvals:", np.linalg.eigvalsh(Sig[i]).round(2))
print("distance between overlapping means:", np.linalg.norm(mu[0]-mu[1]).round(2))

def sample(n):
    z = rng.choice(4, n, p=alpha)
    return np.array([rng.multivariate_normal(mu[k], Sig[k]) for k in z])

def cv_loglik(X, K, folds):
    """Mean validation log-likelihood (per sample) of a K-component GMM over the folds."""
    tot = 0.0
    for k in range(FOLDS):
        va = folds[k]; tr = np.hstack([folds[i] for i in range(FOLDS) if i != k])
        if len(tr) < K: return -np.inf                      # cannot fit K components with fewer samples
        try:
            g = GaussianMixture(K, covariance_type="full", reg_covar=1e-3, n_init=2,
                                max_iter=200, random_state=int(rng.integers(1 << 30))).fit(X[tr])
            tot += g.score_samples(X[va]).sum()
        except Exception:
            return -np.inf
    return tot / len(X)

counts = {N: np.zeros(KMAX, int) for N in SIZES}
mean_ll = {N: np.zeros(KMAX) for N in SIZES}
for N in SIZES:
    for r in range(REPS):
        X = sample(N)
        folds = np.array_split(rng.permutation(N), FOLDS)
        ll = np.array([cv_loglik(X, K, folds) for K in range(1, KMAX + 1)])
        counts[N][np.argmax(ll)] += 1
        mean_ll[N] += np.where(np.isfinite(ll), ll, np.nan if False else -1e3) / REPS
    print(f"N={N} done; selection rates:", (counts[N] / REPS).round(2))

# ---- Report: table + figures
orders = np.arange(1, KMAX + 1)
header = "N\\K " + " ".join(f"{k:5d}" for k in orders)
print("\nSelection rate (%) of each GMM order\n" + header)
for N in SIZES: print(f"{N:5d} " + " ".join(f"{100*c/REPS:5.0f}" for c in counts[N]))
np.savetxt("q2_selection_rates.csv", np.array([counts[N] / REPS for N in SIZES]), delimiter=",",
           header="rows N=10,100,1000; cols K=1..10", fmt="%.3f")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for a, N in zip(ax, SIZES):
    a.bar(orders, counts[N] / REPS, color="steelblue"); a.axvline(4, color="r", ls="--", lw=1)
    a.set_title(f"N = {N}"); a.set_xlabel("GMM order K"); a.set_xticks(orders)
ax[0].set_ylabel("Selection rate"); fig.suptitle(f"Order selected by 10-fold CV ({REPS} repetitions); red = true K=4")
plt.tight_layout(); plt.savefig("q2_selection_rates.png", dpi=150)

plt.figure(figsize=(5, 3.8))
plt.imshow(np.array([counts[N] / REPS for N in SIZES]), cmap="viridis", aspect="auto")
plt.xticks(range(KMAX), orders); plt.yticks(range(3), SIZES); plt.colorbar(label="rate")
plt.xlabel("K"); plt.ylabel("N"); plt.tight_layout(); plt.savefig("q2_heatmap.png", dpi=150)
