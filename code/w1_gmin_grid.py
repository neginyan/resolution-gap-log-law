"""Stage W (main-line request 3) at LARGE sigma (1e-2, 1e-4, 1e-5), where the scales of the strip cannot be separated (Lv ~ a few ripple
periods ~ ell): g_min(sigma) of the same ripple-strip family (ripple filling the strip, c free) by the full 2-d grid evaluator of stage U
(u_cgm.evaluate: own state construction, oscillating N-form, tensor grid, valid at any scale; <= 8e6 points).
Constraint y/sigma <= 0.5, |grid mass - 1| < 1e-4, Lw <= 2, Lv >= 3 sigma.  Starts: the stage-V finite-strip optimum (v2_gapmin, nearest sigma)
and k0 = 1.5.  Final: a 1.5x finer grid (feasibility re-checked there).
usage: python3 w1_gmin_grid.py SIGMA [MAXEVAL]  -> results/w1_gmin_grid_opt_s<SIGMA>.json.  Python 3.11."""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize
from u_cgm import ripple_terms, nodes, evaluate

SIG = float(sys.argv[1]); MAXEV = int(sys.argv[2]) if len(sys.argv) > 2 else 200
OUT = 'results/w1_gmin_grid_opt_s%g.json' % SIG


def state(x):
    k0, lam, M, Lv_s = np.exp(x[:4]); c = 1 / (1 + np.exp(-x[4])); return k0, lam, M, Lv_s, c


def ev(x, fine=1.0):
    k0, lam, M, Lv_s, c = state(x); sg = SIG; s = np.sqrt(sg); ell = lam * s; Lw = M * ell; Lv = Lv_s * sg
    T = ripple_terms(sg, Lv, Lw, c, k0, 1e30, ell, 0.0)
    hv = min(0.35 * sg, 2 * np.pi * sg / k0 / 12) / fine; Tv, Tw = np.sqrt(Lv ** 2 + sg ** 2), np.sqrt(Lw ** 2 + sg ** 2)
    vn = nodes(6.5 * Tv, hv, 6.5 * Tv, hv); wn = nodes(min(7 * ell, 6.5 * Tw), ell / 40 / fine, 6.5 * Tw, max(Tw / 120, ell / 40) / fine)
    while len(vn) * len(wn) > 8.0e6: hv *= 1.05; vn = nodes(6.5 * Tv, hv, 6.5 * Tv, hv)
    return evaluate(T, sg, vn, wn)[1]


def objective(x, log):
    k0, lam, M, Lv_s, c = state(x)
    if not (0.3 < k0 < 6 and 0.2 < lam < 50 and 0.5 < M < 1e6 and M * lam * np.sqrt(SIG) < 2.0 and 3 < Lv_s < 3000): return 60.0
    try: d = ev(x)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return 70.0
    if not np.isfinite(d['gap_over_sigma']) or abs(d['grid_mass'] - 1) > 1e-4: return 50.0
    if d['y_over_sigma'] > 0.5: return 40.0
    log.append(d['gap_over_sigma']); return d['gap_over_sigma']


if __name__ == '__main__':
    np.seterr(all='ignore')
    st = json.load(open(OUT)) if os.path.exists(OUT) else {'sigma': SIG, 'runs': [], 'done': [], 'best': None}
    near = min([1e-2, 1e-3, 1e-6], key=lambda t: abs(np.log(t / SIG)))
    p = json.load(open('results/v2_gapmin_s%g_seed0.json' % near))['best']['par']
    starts = [np.array([np.log(p['k0']), np.log(p['lam']), np.log(max(p['Aw'] / p['lam'], 0.6)), np.log(p['Lv_s']), np.log(p['c'] / (1 - p['c']))]),
              np.array([np.log(1.5), np.log(1.5), np.log(min(10.0, 0.3 / (1.5 * np.sqrt(SIG)))), np.log(60.0), 3.0])]
    for i, x0 in enumerate(starts):
        if i in st['done']: continue
        log = []; t0 = time.time(); r = None
        for rnd in range(2):
            r = minimize(objective, x0, args=(log,), method='Nelder-Mead', options={'maxfev': MAXEV // 2, 'xatol': 1e-3, 'fatol': 1e-5, 'adaptive': True}); x0 = r.x
        k0, lam, M, Lv_s, c = state(r.x)
        rec = {'start': i, 'opt_gap_over_sigma': float(r.fun), 'x': list(map(float, r.x)), 'k0': k0, 'lam': lam, 'M': M, 'Lv_over_sigma': Lv_s, 'c': c, 'seconds': time.time() - t0}
        st['runs'].append(rec); st['done'].append(i)
        if st['best'] is None or rec['opt_gap_over_sigma'] < st['best']['opt_gap_over_sigma']: st['best'] = rec
        json.dump(st, open(OUT, 'w'), indent=1); print('sigma %g start %d: gap %.5f k0 %.3f lam %.3f M %.3g Lv/sigma %.3g c %.3f' % (SIG, i, r.fun, k0, lam, M, Lv_s, c), flush=True)
    b = st['best']; f = ev(np.array(b['x']), fine=1.5)
    b['final'] = {k: float(f[k]) for k in ('gap_over_sigma', 'y_over_sigma', 'P', 'F', 'grid_mass', 'n_points')}
    b['feasible_on_final_grids'] = bool(f['y_over_sigma'] <= 0.5 and abs(f['grid_mass'] - 1) < 1e-4)
    st['finished'] = True; json.dump(st, open(OUT, 'w'), indent=1); print('final', SIG, b['final'], flush=True)
