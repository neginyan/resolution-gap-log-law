"""Stage AB1: F = ln(sigma^2 J_ww) + Lambda/4 on finite configurations (sigma = 1), with the 2-d grid evaluator of stage Z (fields of p only):
  (a) the main line's examples: diamond n x n (b1 = (3.5, 3.34), b2 = (3.5, -3.34)), n = 3 ... 11; a chain of 12 equal steps Delta = (3.871, 3.228);
  (b) adversarial search over K free Gaussian components (weights, positions, widths, tilts), K = 3, 6, 9 -- the non-lattice families.
-> results/ab_finite.json (restartable by key).  Python 3.11."""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize
from z_tools import grid_sums


def F_of(T, h=0.08, ext=8.0):
    mv = [t[1][0].real for t in T]; mw = [t[1][1].real for t in T]; sv = [np.sqrt(t[2][0][0] + 1) for t in T]; sw = [np.sqrt(t[2][1][1] + 1) for t in T]
    vn = np.arange(min(m - ext * s for m, s in zip(mv, sv)), max(m + ext * s for m, s in zip(mv, sv)) + h / 2, h)
    wn = np.arange(min(m - ext * s for m, s in zip(mw, sw)), max(m + ext * s for m, s in zip(mw, sw)) + h / 2, h)
    E, L, m = grid_sums(T, 1.0, vn, wn); Lam = E['lam'] / E['sww']
    return {'F': float(np.log(E['sww']) + Lam / 4), 'J_ww': float(E['sww']), 'Lambda': float(Lam), 'mass': float(m), 'n_points': len(vn) * len(wn)}


def points(P, w=None):
    w = np.full(len(P), 1.0 / len(P)) if w is None else np.asarray(w) / np.sum(w)
    return [(complex(np.log(wi)), np.array(p, complex), np.zeros((2, 2))) for wi, p in zip(w, P)]


def build(x, K):
    x = np.asarray(x).reshape(K, 6); w = np.exp(x[:, 0] - x[:, 0].max()); w /= w.sum(); T = []
    for k in range(K):
        sa, sb = np.exp(np.clip(x[k, 3], -6, 1.5)), np.exp(np.clip(x[k, 4], -6, 1.5)); th = x[k, 5]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); S = R @ np.diag([sa * sa, sb * sb]) @ R.T
        T.append((complex(np.log(max(w[k], 1e-300))), np.array([3 * x[k, 1], 3 * x[k, 2]], complex), S))
    return T


if __name__ == '__main__':
    fn = 'results/ab_finite.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    def save(): json.dump(out, open(fn, 'w'), indent=1, default=float)
    b1, b2 = np.array([3.5, 3.34]), np.array([3.5, -3.34])
    for n in (3, 5, 7, 9, 11):
        k = 'diamond_n%d' % n
        if k not in out:
            P = [i * b1 + j * b2 for i in range(n) for j in range(n)]; out[k] = F_of(points(P), h=0.1); save(); print(k, out[k], flush=True)
    if 'chain12' not in out:
        D = np.array([3.871, 3.228]); out['chain12'] = F_of(points([i * D for i in range(12)]), h=0.08); save(); print('chain12', out['chain12'], flush=True)
    rng = np.random.default_rng(3)
    for K in (3, 6, 9):
        for st in range(3):
            key = 'adv_K%d_start%d' % (K, st)
            if key in out: continue
            if st == 0: x0 = np.column_stack([np.zeros(K), np.arange(K) * 3.871 / 3, np.arange(K) * 3.228 / 3, np.full(K, -4.0), np.full(K, -4.0), np.zeros(K)])
            elif st == 1:
                g = int(np.ceil(np.sqrt(K))); P = [(i * b1 + j * b2) / 3 for i in range(g) for j in range(g)][:K]
                x0 = np.column_stack([np.zeros(K), [p[0] for p in P], [p[1] for p in P], np.full(K, -4.0), np.full(K, -4.0), np.zeros(K)])
            else: x0 = np.column_stack([rng.normal(0, 0.5, K), rng.uniform(-3, 3, K), rng.uniform(-3, 3, K), rng.uniform(-4, 0, K), rng.uniform(-4, 0, K), rng.uniform(0, 3, K)])
            t0 = time.time(); best = {'F': -1e9}
            def f(x):
                try: r = F_of(build(x, K), h=0.12, ext=7.0)
                except (ValueError, np.linalg.LinAlgError, MemoryError): return 1e3
                if r['n_points'] > 3e6 or not np.isfinite(r['F']) or abs(r['mass'] - 1) > 1e-3: return 1e3
                if r['F'] > best['F']: best.update(r, x=list(map(float, np.ravel(x))))
                return -r['F']
            minimize(f, x0.ravel(), method='Nelder-Mead', options={'maxfev': 1200 if K < 9 else 900, 'xatol': 1e-5, 'fatol': 1e-8})
            best['fine'] = F_of(build(best['x'], K), h=0.05, ext=9.0); best['seconds'] = time.time() - t0; out[key] = best; save()
            print('%-18s F %.4f (fine %.4f) (%.0fs)' % (key, best['F'], best['fine']['F'], best['seconds']), flush=True)
