"""Stage AA2: adversarial maximisation of K_Lambda = xbar_w(sqrt p)/xbar_w(p) (and of F = sigma^2 J_ww exp(Lambda/4), the test quantity of the
Fisher-form smoothing lemma) over Gaussian mixtures in (v, w) coordinates relative to the top.
Usage:  python aa_klam.py <sigma> -> results/aa_klam_s<sigma>.json  (restartable: one entry per (objective, K, start) run; checkpoint after each run)
Objectives:  KL      max ln K_Lambda                      subject to y/sigma <= 1/2
             KL_gap  max ln K_Lambda                      subject to y/sigma <= 1/2 and sqrt(P_w) <= 1  (the small-gap regime: gap/sigma ~ sqrt(P_w))
             F       max ln(sigma^2 J_ww) + Lambda/4      subject to y/sigma <= 1/2
Evaluation: z_tools.grid_sums (fields from p, grad p, Hess p) on the adaptive node sets of z_chain, closed-form L2 integrals (aa_tools.pair_l2).
Search: Nelder-Mead (scipy), 3 starts per (objective, K), 700 evaluations each (350 for K = 8), coarse grid; the best state is re-evaluated on a fine grid.
Python 3.11."""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize
from z_tools import grid_sums
from aa_tools import pair_l2
from z_chain import mixture_grid


def build(x, K, sig):
    s = np.sqrt(sig); x = np.asarray(x).reshape(K, 6); wl = x[:, 0] - x[:, 0].max(); w = np.exp(wl); w /= w.sum(); T = []
    for k in range(K):
        mv, mw = 5 * x[k, 1] * sig, 5 * x[k, 2] * s
        sv = sig * np.exp(np.clip(x[k, 3], np.log(0.05), np.log(40))); sw = s * np.exp(np.clip(x[k, 4], np.log(0.05), np.log(40))); r = 0.98 * np.tanh(x[k, 5])
        S = np.array([[sv * sv, r * sv * sw], [r * sv * sw, sw * sw]]); T.append((complex(np.log(max(w[k], 1e-300))), np.array([mv, mw], complex), S))
    return T


def evaluate(T, sig, n=6):
    s2 = sig * sig; vn, wn = mixture_grid(T, sig, n)
    if len(vn) * len(wn) > 4e6: return None
    E, L, m = grid_sums(T, sig, vn, wn); I0, Iww, Ivvww, _ = pair_l2(T, s2 * np.eye(2))
    Jww, Jvv = E['sww'], E['svv']; a = s2 * Jvv; Lam = s2 * E['lam'] / Jww; xw = s2 * Ivvww / Iww
    Pw = 4 * s2 * E['sww_w'] * E['phi_svv'] / a ** 2; yos = sig * Jww / (s2 * E['jpos'])
    return {'K_Lambda': Lam / (4 * xw), 'Lambda': Lam, 'xbar_w': xw, 'J_ww': Jww, 'a': a, 'P_w': Pw, 'y_over_sigma': yos, 'F_log': float(np.log(s2 * Jww) + Lam / 4),
            'grid_mass': m, 'n_points': len(vn) * len(wn)}


def objective(kind, r):
    if r is None or not np.isfinite(r['K_Lambda']): return 1e6
    pen = 50 * max(r['y_over_sigma'] - 0.5, 0) + 50 * abs(r['grid_mass'] - 1) * (abs(r['grid_mass'] - 1) > 1e-3)
    if kind == 'KL_gap': pen += 20 * max(np.sqrt(r['P_w']) - 1.0, 0)
    val = np.log(r['K_Lambda']) if kind in ('KL', 'KL_gap') else r['F_log']
    return -val + pen


def starts(K, rng):
    out = []
    # (a) a wide smooth component plus small narrow components (low-density structures); (b) random; (c) hourglass-like (needle + side blobs)
    a = np.zeros((K, 6)); a[0] = [3, 0, 0, np.log(30), np.log(20), 0]
    for k in range(1, K): a[k] = [-3, rng.uniform(-6, 6), rng.uniform(-6, 6), np.log(0.5), np.log(1.0), 0]
    out.append(a.ravel()); out.append(np.column_stack([rng.normal(0, 1, K), rng.normal(0, 1.5, K), rng.normal(0, 1.5, K), rng.uniform(-1, 3, K), rng.uniform(-1, 3, K), rng.normal(0, 0.5, K)]).ravel())
    c = np.zeros((K, 6)); c[0] = [2, 0, 0, np.log(0.3), np.log(3), 0]
    for k in range(1, K): c[k] = [-1, rng.uniform(-1, 1), (1 if k % 2 else -1) * rng.uniform(4, 8) / 5, np.log(rng.uniform(2, 12)), np.log(1.0), 0]
    out.append(c.ravel()); return out


if __name__ == '__main__':
    sig = float(sys.argv[1]); fn = 'results/aa_klam_s%g.json' % sig; out = json.load(open(fn)) if os.path.exists(fn) else {}
    rng = np.random.default_rng(int(-np.log10(sig)))
    for kind in ('KL', 'KL_gap', 'F'):
        for K in (4, 8):
            for i, x0 in enumerate(starts(K, rng)):
                key = '%s_K%d_start%d' % (kind, K, i)
                if key in out: continue
                t0 = time.time(); best = {'f': 1e9}
                def f(x):
                    try: r = evaluate(build(x, K, sig), sig)
                    except (np.linalg.LinAlgError, ValueError, FloatingPointError): return 1e6
                    v = objective(kind, r)
                    if v < best['f']: best.update(f=v, x=list(map(float, x)), r=r)
                    return v
                minimize(f, x0, method='Nelder-Mead', options={'maxfev': 700 if K == 4 else 350, 'xatol': 1e-4, 'fatol': 1e-6})
                rec = {'kind': kind, 'K': K, 'start': i, 'sigma': sig, 'seconds': time.time() - t0}
                if 'x' in best:
                    fine = evaluate(build(best['x'], K, sig), sig, n=12)
                    rec.update({'x': best['x'], 'coarse': best['r'], 'fine': fine, 'objective': best['f']})
                out[key] = rec; json.dump(out, open(fn, 'w'), indent=1, default=float)
                fr = rec.get('fine') or {}
                print('%-22s K_Lambda %.4g  F_log %.3g  sqrtPw %.3g  y/s %.3g  (%.0fs)' % (key, fr.get('K_Lambda', np.nan), fr.get('F_log', np.nan), np.sqrt(fr.get('P_w', np.nan)), fr.get('y_over_sigma', np.nan), rec['seconds']), flush=True)
