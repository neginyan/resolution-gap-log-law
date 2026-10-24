"""Stage W (main-line request 3): g_min(sigma) of the ripple strip computed ENTIRELY with the separate evaluator w_indep (two-scale, phase-based;
valid when the strip holds many ripple periods, Lv >> sigma/k0, i.e. sigma <= 1e-6 here).  Same family and constraint as w1_gmin
(k0, lam, M, eta = Lv/ell, c; y/sigma <= 0.5; Lw = M lam sqrt(sigma) <= 2), one start per sigma (continuation from the next larger sigma; at 1e-6 the stage-V optimum), Nelder-Mead.
It doubles as a cross-check of w1_gmin (different evaluator, different optimiser path) and extends the sigma range to 1e-96 for the log-law fit.
usage: python3 w1_gmin_indep.py  -> results/w1_gmin_indep_opt.json (restartable by sigma).  Python 3.11."""
import json, os
import numpy as np
from scipy.optimize import minimize
from w_indep import strip_gap

FN = 'results/w1_gmin_indep_opt.json'
SIGMAS = [1e-6, 1e-12, 1e-24, 1e-48, 1e-96]


def state(x):
    k0, lam, M, eta = np.exp(x[:4]); c = 1 / (1 + np.exp(-x[4])); return k0, lam, M, eta, c


def obj(x, sg):
    k0, lam, M, eta, c = state(x)
    if not (0.3 < k0 < 12 and 0.1 < lam < 50 and 1 < M < 1e100 and M * lam * np.sqrt(sg) < 2.0 and eta * lam / np.sqrt(sg) > 30 / k0): return 60.0
    try: r = strip_gap(sg, k0, lam, M, eta, c)
    except (FloatingPointError, OverflowError, ValueError): return 70.0
    g = r['gap_over_sigma']
    if not np.isfinite(g) or r['y_over_sigma'] > 0.5: return 40.0
    return g


if __name__ == '__main__':
    np.seterr(all='ignore')
    R = json.load(open(FN)) if os.path.exists(FN) else {}
    for sg in SIGMAS:
        key = '%g' % sg
        if key in R: continue
        L = np.log(1 / sg); best = None
        prev = [R[k] for k in R if R[k]['sigma'] > sg]
        if prev:                                    # continuation from the previous (larger) sigma: k0^2 grows like ln(1/sigma)
            p = min(prev, key=lambda r: r['sigma']); k0 = p['k0'] * np.sqrt(L / p['ln_inv_sigma'])
            M = min(p['M'] * np.exp(k0 ** 2 - p['k0'] ** 2), 0.3 / (p['lam'] * np.sqrt(sg)))
            starts = [np.log([k0, p['lam'], M, p['Lv_over_ell']]).tolist() + [np.log(p['c'] / (1 - p['c'] + 1e-12) + 1e-12)]]
        else:                                       # the stage-V finite-strip optimum at sigma = 1e-6
            q = json.load(open('results/v2_gapmin_s1e-06_seed0.json'))['best']['par']
            starts = [[np.log(q['k0']), np.log(q['lam']), np.log(q['Aw'] / q['lam']), np.log(q['Lv_s'] * np.sqrt(sg) / q['lam']), 3.0]]
        for x0 in starts:
            x0 = np.array(x0, float); r = None
            for rnd in range(2):
                r = minimize(obj, x0, args=(sg,), method='Nelder-Mead', options={'maxfev': 250, 'xatol': 1e-4, 'fatol': 1e-7, 'adaptive': True}); x0 = r.x
            if best is None or r.fun < best[0]: best = (float(r.fun), r.x)
        k0, lam, M, eta, c = state(best[1]); f = strip_gap(sg, k0, lam, M, eta, c, n_theta=96, n_slow=24, n_gh=64, n_gl=16, win_panels_per_ell=5, bg_panels=96)
        R[key] = {'sigma': sg, 'ln_inv_sigma': L, 'opt_gap_over_sigma': best[0], 'final_gap': f['gap_over_sigma'], 'y_over_sigma': f['y_over_sigma'], 'P_sharp': f['P_sharp'],
                  'k0': k0, 'lam': lam, 'M': M, 'Lv_over_ell': eta, 'c': c, 'x': list(map(float, best[1])), 'feasible': bool(f['y_over_sigma'] <= 0.5)}
        json.dump(R, open(FN, 'w'), indent=1); print(key, {k: R[key][k] for k in ('final_gap', 'k0', 'lam', 'M', 'Lv_over_ell', 'c')}, flush=True)
