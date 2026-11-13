"""Q1: MLP class-posterior approximation (4 classes, 3-D Gaussians), numpy + scipy L-BFGS.
MLP: x -> ELU hidden layer (P units) -> softmax. Trained by minimizing average cross-entropy (= ML)."""
import sys, numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUICK = "--quick" in sys.argv
rng = np.random.default_rng(0)
C, D = 4, 3

# ---------------- True data distribution ----------------
base_means = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [1.5, 1.5, 3]], float)
covs = [np.array([[1.0, .3, 0], [.3, 1.0, .2], [0, .2, 1.0]]),
        np.array([[1.5, -.4, 0], [-.4, 1.0, 0], [0, 0, 1.2]]),
        np.array([[1.0, 0, .3], [0, 1.6, .3], [.3, .3, 1.0]]),
        np.array([[1.2, .2, .2], [.2, 1.2, -.3], [.2, -.3, 1.5]])]

def sample(n, means):
    y = rng.integers(0, C, n)                       # uniform priors
    x = np.empty((n, D))
    for c in range(C):
        idx = y == c
        x[idx] = rng.multivariate_normal(means[c], covs[c], idx.sum())
    return x, y

def bayes_predict(x, means):
    lik = np.stack([mvn(means[c], covs[c]).pdf(x) for c in range(C)], 1)   # uniform priors -> argmax pdf
    return lik.argmax(1)

# scale mean separation so optimal P(error) lands in 10-20%
scale = 1.0
for _ in range(25):
    m = base_means * scale
    xt, yt = sample(50000, m)
    e = np.mean(bayes_predict(xt, m) != yt)
    if 0.12 <= e <= 0.18: break
    scale *= 1.1 if e > 0.18 else 0.93
means = base_means * scale
x_test, y_test = sample(100000 if not QUICK else 20000, means)
p_opt = np.mean(bayes_predict(x_test, means) != y_test)
print(f"mean scale={scale:.3f}; optimal (true-pdf MAP) test P(error) = {p_opt:.4f}")

# ---------------- MLP (manual forward/backward) ----------------
def unpack(th, P):
    i = 0
    W1 = th[i:i+P*D].reshape(P, D); i += P*D
    b1 = th[i:i+P]; i += P
    W2 = th[i:i+C*P].reshape(C, P); i += C*P
    b2 = th[i:i+C]
    return W1, b1, W2, b2

def forward(th, X, P):
    W1, b1, W2, b2 = unpack(th, P)
    z = X @ W1.T + b1
    h = np.where(z > 0, z, np.expm1(np.minimum(z, 0)))      # ELU
    a = h @ W2.T + b2
    a -= a.max(1, keepdims=True)
    p = np.exp(a); p /= p.sum(1, keepdims=True)              # softmax
    return z, h, p

def loss_grad(th, X, Y, P, l2=1e-5):
    N = len(X)
    W1, b1, W2, b2 = unpack(th, P)
    z, h, p = forward(th, X, P)
    nll = -np.mean(np.log(p[np.arange(N), Y] + 1e-300))
    dl = p.copy(); dl[np.arange(N), Y] -= 1; dl /= N
    gW2 = dl.T @ h; gb2 = dl.sum(0)
    dz = (dl @ W2) * np.where(z > 0, 1.0, np.exp(np.minimum(z, 0)))
    gW1 = dz.T @ X; gb1 = dz.sum(0)
    g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
    return nll + 0.5*l2*np.sum(th**2) - 0.5*l2*0, g + l2*th   # tiny L2 only for numerical stability

def train(X, Y, P, restarts=3, maxiter=400):
    """Multiple random restarts; keep the solution with the lowest training CE (= highest log-likelihood)."""
    best_th, best_f = None, np.inf
    n = P*D + P + C*P + C
    for _ in range(restarts):
        th0 = rng.normal(0, 0.5, n)
        r = minimize(loss_grad, th0, args=(X, Y, P), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        if r.fun < best_f: best_f, best_th = r.fun, r.x
    return best_th

def predict(th, X, P):
    return forward(th, X, P)[2].argmax(1)                    # MAP rule on estimated posteriors

# sanity check: analytic gradient vs finite differences
P_ = 5; Xg, Yg = sample(50, means); th = rng.normal(0, .5, P_*D+P_+C*P_+C)
f0, g0 = loss_grad(th, Xg, Yg, P_); gn = np.zeros_like(th)
for i in range(len(th)):
    t = th.copy(); t[i] += 1e-6; gn[i] = (loss_grad(t, Xg, Yg, P_)[0] - f0) / 1e-6
print("gradient check, max abs diff:", np.abs(g0 - gn).max())

# ---------------- Cross-validation, final training, test ----------------
sizes = [100, 500, 1000, 5000, 10000] if not QUICK else [100, 500]
P_grid = [1, 2, 4, 8, 16, 32, 64] if not QUICK else [1, 4, 16]
K = 10
cv_restarts, final_restarts = (1, 5) if not QUICK else (1, 2)

def kfold_idx(n, K):
    perm = rng.permutation(n); return np.array_split(perm, K)

results = []
cv_tables = {}
for N in sizes:
    X, Y = sample(N, means)
    folds = kfold_idx(N, K)
    perr = np.zeros(len(P_grid))
    for j, P in enumerate(P_grid):
        errs = []
        for k in range(K):
            va = folds[k]; tr = np.hstack([folds[i] for i in range(K) if i != k])
            th = train(X[tr], Y[tr], P, restarts=cv_restarts)
            errs.append(np.mean(predict(th, X[va], P) != Y[va]))
        perr[j] = np.mean(errs)
    P_best = P_grid[int(np.argmin(perr))]
    th = train(X, Y, P_best, restarts=final_restarts, maxiter=800)
    test_err = np.mean(predict(th, x_test, P_best) != y_test)
    results.append((N, P_best, test_err)); cv_tables[N] = perr
    print(f"N={N:6d}  best P={P_best:3d}  CV P(err)={perr.min():.3f}  TEST P(err)={test_err:.4f}")

print("\nN, P*, test P(error), (optimal =", round(p_opt, 4), ")")
for r in results: print(r)
print("\nCV P(error) per candidate P:", P_grid)
for N, v in cv_tables.items(): print(N, np.round(v, 3))

plt.figure(figsize=(6, 4))
plt.semilogx([r[0] for r in results], [r[2] for r in results], "o-", label="Trained MLP (CV-selected P)")
plt.axhline(p_opt, color="r", ls="--", label="Theoretically optimal")
for N, P, e in results: plt.annotate(f"P={P}", (N, e), textcoords="offset points", xytext=(0, 6), ha="center")
plt.xlabel("# training samples"); plt.ylabel("Test P(error)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("q1_perror_vs_samples.png", dpi=150)
