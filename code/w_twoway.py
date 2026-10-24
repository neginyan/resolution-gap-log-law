"""Stage W1: two independent evaluations of the corrected (stage-W) strip gaps.
  A  w_full.full_sums  : grids (one V-period per v0 node, trapezoid over the true V-envelope), cancellation-free point fields;
  B  v3_tools.mc_delta_sums: the finite strip, no grid: Monte Carlo (importance sampling of the window, common random numbers against the pure
                         Gaussian background, whose sums come from a separable grid).
Both use the stage-W point fields (exact d_p m_z = -sigma^2 ez.T^-1.ep and difference-based F); these were checked point by point against an
independent 320-digit evaluation (w_mpcheck).  -> results/w_twoway.json.  Python 3.11."""
import json, os, time, sys
import numpy as np
from w_tools import terms_of, w_nodes
from w_full import full_sums
from v3_tools import mc_delta_sums

ROWS = [(3.0, 1.5, 8.1e4, 0.05, 1e-24), (2.5, 1.5, 1e5, 1 / 150, 1e-24)]
# Both evaluators here work with absolute positions (v0 + a fraction of a period; Monte Carlo y = x + sigma Z), which keep the phase k v / sigma
# only while Lv/sigma <~ 1e12.  A first run also tried (k0 4, sigma 1e-80, Lv/sigma ~ 1e38): both failed (31.7 / 186), the phase-based w_indep gives
# 0.12500 there, as does the one-period limit.  That row is therefore not a valid test of these two evaluators and is not used.
if os.path.exists('results/w1_gmin_opt_s1e-24.json'):
    _b = json.load(open('results/w1_gmin_opt_s1e-24.json')).get('best')
    if _b: ROWS.append(('gmin', _b['k0'], _b['lam'], _b['M'], _b['Lv_over_ell'], 1e-24, _b['c']))


def row(k0, lam, M, eta, sg, c=1.0, n=160_000_000, seed=17):
    s = np.sqrt(sg); ell = lam * s; Lw = M * ell; Lvs = eta * lam / np.sqrt(sg); Lv = Lvs * sg
    bg = [(1.0, 0.0, lam * M)]; win = [(c, 0.0, lam, 1)]
    A = full_sums(terms_of(bg, win, k0, sg, Lvs), sg, k0, np.sqrt(Lv ** 2 + sg ** 2), w_nodes(bg, win, sg, 40, 400), nv=32, n0=21, ext=6.5)
    t = time.time(); B = mc_delta_sums(sg, dict(Lv=Lv, Lw=Lw, c=c, k0=k0, Rv=1e30, ell=ell), n=n, seed=seed)
    r = {'k0': k0, 'lam': lam, 'M': M, 'Lv_over_ell': eta, 'sigma': sg, 'c': c, 'A_gap': float(A['gap_over_sigma']), 'A_P_sharp': float(A['P_sharp']),
         'A_P': float(A['P']), 'A_F': float(A['F']), 'A_y': float(A['y_over_sigma']),
         'B_gap': B['gap_over_sigma'], 'B_gap_se': B['gap_over_sigma_se'], 'B_P_sharp': B['P_sharp'], 'B_P': B['P'], 'B_F': B['F'], 'B_y': B['y_over_sigma'],
         'z_score': (B['gap_over_sigma'] - float(A['gap_over_sigma'])) / max(B['gap_over_sigma_se'], 1e-300), 'seconds_B': time.time() - t}
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True); return r


if __name__ == '__main__':
    fn = 'results/w_twoway.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(o['k0'], o['sigma']) for o in out}
    out = [o for o in out if o['sigma'] >= 1e-30]          # drop the invalid sigma = 1e-80 row of the first run (see above)
    for r_ in ROWS:
        tag = r_[0] == 'gmin'; r_ = r_[1:] if tag else r_
        if (r_[0], r_[4]) in done and not tag: continue
        if tag and any(o.get('tag') == 'gmin_1e-24' for o in out): continue
        o = row(*r_)
        if tag: o['tag'] = 'gmin_1e-24'
        out.append(o); json.dump(out, open(fn, 'w'), indent=1)
