"""Stage AB2: the tail correction eps_pi of step 1 without H0, on every mixture-type state of stage Z (x1 30 + stored 377).
eps_pi^2 := E[s_v^2 m_z^2 1{|m_z| > pi}] / E[s_v^2 m_z^2]   (step 1 then reads  P_w >= (2/pi^2) P# (1 - eps_pi^2)  for every state).
Grid: the adaptive node sets of stage Z (z_chain.mixture_grid, n_per_sd = 8), fields from p, grad p only (z_tools.fields); m = y + sigma^2 s.
Also Var_rho(z) (closed form, hypothesis H0'': Var(X_z) <= pi^2).
Usage: python ab_tail.py x1|stored  -> results/ab_tail_<mode>.json (restartable by key).  Python 3.11."""
import sys, os, json, time
import numpy as np
from z_tools import fields, _prep, moments
from z_chain import mixture_grid, jobs_stored
from aa_chain import terms_x1


def tail(T, sig, n=8, rows=64):
    pre = _prep(T, sig); s2 = sig * sig; vn, wn = mixture_grid(T, sig, n)
    wv = np.zeros_like(vn); d = np.diff(vn); wv[1:] += d / 2; wv[:-1] += d / 2
    ww = np.zeros_like(wn); d = np.diff(wn); ww[1:] += d / 2; ww[:-1] += d / 2
    S = {'m': 0.0, 'all': 0.0, 'tail': 0.0, 'pt': 0.0, 'zmax_grid': float(max(abs(vn).max(), 0) + abs(wn).max()) / np.sqrt(2)}
    for a in range(0, len(vn), rows):
        V, W = np.meshgrid(vn[a:a + rows], wn, indexing='ij'); x = np.stack([V.ravel(), W.ravel()], 1); dA = np.outer(wv[a:a + rows], ww).ravel()
        p, g, _ = fields(pre, x); ok = p > 0; p = np.where(ok, p, 1e-300); sv, sw = g[:, 0] / p, g[:, 1] / p
        mz = (x[:, 0] + x[:, 1] + s2 * (sv + sw)) / np.sqrt(2); wt = p * dA * ok; f = wt * sv * sv * mz * mz; t = np.abs(mz) > np.pi
        S['m'] += wt.sum(); S['all'] += f.sum(); S['tail'] += f[t].sum(); S['pt'] += wt[t].sum()
    return {'eps_pi': float(np.sqrt(S['tail'] / S['all'])), 'tail_mass': S['pt'] / S['m'], 'grid_mass': S['m'], 'Var_rho_z': float(moments(T)), 'n_points': len(vn) * len(wn),
            'grid_reaches_pi': S['zmax_grid'] >= np.pi}


if __name__ == '__main__':
    mode = sys.argv[1]; fn = 'results/ab_tail_%s.json' % mode; out = json.load(open(fn)) if os.path.exists(fn) else {}
    if mode == 'x1':
        Z = json.load(open('results/z_chain_x1.json')); jobs = [(k, terms_x1(k, z['sigma']), z['sigma']) for k, z in Z.items() if not k.startswith('strip')]
    else:
        jobs = jobs_stored()
    for k, T, sg in jobs:
        if k in out: continue
        t0 = time.time()
        try: r = tail(T, sg)
        except (MemoryError, ValueError, np.linalg.LinAlgError) as e: r = {'error': str(e)}
        r['seconds'] = time.time() - t0; out[k] = r; json.dump(out, open(fn, 'w'), indent=1, default=float)
        print('%-44s eps_pi %.3g tail %.3g Var(z) %.3g (%.0fs)' % (k, r.get('eps_pi', np.nan), r.get('tail_mass', np.nan), r.get('Var_rho_z', np.nan), r['seconds']), flush=True)
