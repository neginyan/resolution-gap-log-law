"""Stage V2: adversarial MINIMISATION of the true gap (1 - R)/sigma over the ripple-strip family (v2_strip.strip; stable R of u_cgm),
y/sigma <= 0.5 held exactly (only feasible evaluations count), grid-mass guard, <= 8e6 points (~1.3 GB).
Parameters: k0 in [0.8, 2.5], c in (0, 1], lambda = ell/sqrt(sigma) in [0.5, 4], A_w = Lw/sqrt(sigma) in [4, 40], Lv/sigma in [30, 300].
Optimised values carry 'opt' (loose).  Checkpointed.  usage: python3 v2_gapmin.py <sigma> [iters] [seed]   Python 3.11 compatible."""
import json, os, sys
import numpy as np
from scipy.optimize import minimize
from v2_strip import strip

CK = 'results/v2_gapmin_ckpt'; os.makedirs(CK, exist_ok=True); Y0 = 0.5
sg_ = lambda t: 1 / (1 + np.exp(-t))


def unpack_s(z):
    return dict(k0=0.8 + 1.7 * sg_(z[0]), c=sg_(z[1]), lam=0.5 + 3.5 * sg_(z[2]), Aw=4 + 36 * sg_(z[3]), Lv_s=30 + 270 * sg_(z[4]))


def job(sig, iters=100, seed=0):
    key = 's%g_seed%d' % (sig, seed); ck = os.path.join(CK, key + '.json')
    st = json.load(open(ck)) if os.path.exists(ck) else {'n': 0, 'best': None}
    def obj(z):
        p = unpack_s(z)
        try: d = strip(sig, p['k0'], p['lam'], p['Aw'], p['Lv_s'], c=p['c'])
        except (MemoryError, ValueError, np.linalg.LinAlgError): return 1e3
        st['n'] += 1
        if st['n'] % 5 == 0: json.dump(st, open(ck, 'w'), default=float)
        if not np.isfinite(d['gap_over_sigma']) or abs(d['grid_mass'] - 1) > 1e-4: return 1e3
        if d['y_over_sigma'] > Y0 * (1 - 1e-3): return 1e2 + d['y_over_sigma']
        if st['best'] is None or d['gap_over_sigma'] < st['best']['gap_over_sigma']: st['best'] = dict(d, par=p, z=list(map(float, z)))
        return d['gap_over_sigma']
    lg = lambda t: float(np.log(t / (1 - t)))
    z0 = np.array([lg((1.25 - 0.8) / 1.7), lg(0.95), lg((1.65 - 0.5) / 3.5), lg((12 - 4) / 36), lg((100 - 30) / 270)])     # the best strip of v2_strip
    if st['best']: z0 = np.array(st['best']['z'])
    elif seed > 0: z0 = z0 + np.random.default_rng(seed).normal(0, 0.6, 5)
    minimize(obj, z0, method='Nelder-Mead', options={'maxfev': iters, 'xatol': 1e-3, 'fatol': 1e-4, 'initial_simplex': z0 + np.vstack([np.zeros(5), np.eye(5) * 0.7])})
    json.dump(st, open(ck, 'w'), default=float)
    out = {'sigma': sig, 'seed': seed, 'n_evals': st['n'], 'best': None}
    b = st['best']
    if b:
        p = b['par']; fine = strip(sig, p['k0'], p['lam'], p['Aw'], p['Lv_s'], c=p['c'], fine_w=2.0)
        out['best'] = {'opt_gap_over_sigma': b['gap_over_sigma'], 'par': p, 'search': {k: b[k] for k in ('P', 'F', 'y_over_sigma', 'B2', 'B1', 'P_sharp', 'delta', 'a', 'grid_mass')},
                       'finer_w': {k: fine[k] for k in ('gap_over_sigma', 'P', 'F', 'y_over_sigma', 'B2', 'grid_mass', 'n_points')},
                       'feasible_on_finer_grid': bool(fine['y_over_sigma'] <= Y0)}
        print(key, 'gap', round(b['gap_over_sigma'], 4), 'finer', round(fine['gap_over_sigma'], 4), 'P', round(b['P'], 4), 'F', round(b['F'], 4), 'y', round(b['y_over_sigma'], 3), p, flush=True)
        # accurate final values (stage V, after the main line's request): cancellation-free N on the same grid, and Monte Carlo without a grid
        from v3_tools import grid_sums, mc_sums
        from u_cgm import ripple_terms, nodes
        sq = np.sqrt(sig); Lv = p['Lv_s'] * sig; Lw = p['Aw'] * sq; ell = p['lam'] * sq
        T = ripple_terms(sig, Lv, Lw, p['c'], p['k0'], 10 * Lv, ell, 0.0); hv = min(0.35 * sig, 2 * np.pi * sig / p['k0'] / 12)
        vn = nodes(6.5 * Lv, hv, 6.5 * Lv, hv); wn = nodes(7 * ell, ell / 80, 6.5 * Lw, max(Lw / 240, ell / 80))
        A = grid_sums(T, sig, vn, wn, rows=16); B = mc_sums(T, sig, dict(Lv=Lv, Lw=Lw, c=p['c'], k0=p['k0'], Rv=10 * Lv, ell=ell), n=4_000_000)
        out['best']['accurate'] = {'gap_ibp_grid': A['gap_over_sigma_ibp'], 'gap_stable_grid': A['gap_over_sigma'], 'gap_mc': B['gap_over_sigma'], 'gap_mc_se': B['gap_over_sigma_se'],
                                   'P_grid': A['P'], 'F_grid': A['F'], 'y_grid': A['y_over_sigma'], 'P_sharp_grid': A['P_sharp'], 'P_mc': B['P'], 'F_mc': B['F']}
        print(key, 'accurate', out['best']['accurate'], flush=True)
    json.dump(out, open('results/v2_gapmin_%s.json' % key, 'w'), indent=1, default=float)


if __name__ == '__main__':
    a = sys.argv[1:]; job(float(a[0]), int(a[1]) if len(a) > 1 else 100, int(a[2]) if len(a) > 2 else 0)
