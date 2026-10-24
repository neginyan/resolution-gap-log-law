"""Stage W1: the corrected sigma -> 0 strip limit, per k0 (replaces the stage-V table v3_klimit / v3_split, which carried the background
artefact).  One-period V-limit at sigma = 1e-80 with Lv = ell/100 (background focus cost and V-spread cost both negligible there), minimised
over (lambda, M) by Nelder-Mead; then the window / background split (w2_tools, window lines omega >= a) on the optimum.
-> results/w1_klimit.json.  Python 3.11."""
import json
import numpy as np
from scipy.optimize import minimize
from w_tools import evaluate, terms_of, w_nodes
from w2_tools import line_sums, analyse

SIG = 1e-80


def ev(k0, lam, M, fine=1):
    bg = [(1.0, 0.0, lam * M)]; win = [(1.0, 0.0, lam, 1)]
    return evaluate(bg, win, k0, sig=SIG, Lv_sig=1e-2 * lam / np.sqrt(SIG), nper=40 * fine, coarse=400 * fine, nv_per=32 * fine)


def run(k0):
    def f(x):
        lam, M = np.exp(x)
        if not (0.2 < lam < 20 and 10 < M < 1e60): return 50.0
        o = ev(k0, lam, M); return o['gap_over_sigma'] if o['y_over_sigma'] <= 0.5 else 40.0
    x0 = np.log([1.2, 1e3 * np.exp(k0 ** 2)]); r = minimize(f, x0, method='Nelder-Mead', options={'maxfev': 160, 'xatol': 1e-3, 'fatol': 1e-6})
    lam, M = np.exp(r.x); o = ev(k0, lam, M, fine=2)
    bg = [(1.0, 0.0, lam * M)]; win = [(1.0, 0.0, lam, 1)]; T = terms_of(bg, win, k0, SIG, 1e-2 * lam / np.sqrt(SIG)); wn = w_nodes(bg, win, SIG, 40, 400)
    per = 2 * np.pi * SIG / k0; nv = 32; vn = (np.arange(nv) + 0.5) * per / nv - per / 2
    A = analyse(line_sums(T, SIG, vn, np.full(nv, per / nv), wn), wn, SIG, 1.0)
    return {'k0': k0, 'opt_gap_over_sigma': float(r.fun), 'lam': float(lam), 'M': float(M), 'final_gap': float(o['gap_over_sigma']),
            'window_part': A['G_win'], 'background_part': A['G_bg'], 'sqrtPsharp_win': A['sqrtPsharp_win'], 'P_sharp': float(o['P_sharp']), 'P': float(o['P']),
            'F': float(o['F']), 'y_over_sigma': float(o['y_over_sigma']), 'feasible': bool(o['y_over_sigma'] <= 0.5 and abs(o['mass_ratio'] - 1) < 1e-6),
            'one_over_2k0': 1 / (2 * k0)}


if __name__ == '__main__':
    rows = []
    for k0 in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0):
        r = run(k0); rows.append(r); print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        json.dump({'sigma': SIG, 'Lv_over_ell': 0.01, 'rows': rows}, open('results/w1_klimit.json', 'w'), indent=1)
