"""Stage W (main-line request 1): every corrected strip value checked by an implementation that shares nothing with the main evaluator.
  main evaluator  : w_tools state + v3_tools point fields (d_p D form of N) + w_full / one-period grids;
  check (sigma <= 1e-6): w_indep (closed-form density, posterior-covariance form of N, two-scale Gauss quadrature; numpy only);
  check (sigma >= 1e-4, where the two scales are not separated): u_cgm.evaluate (stage-U code: own state construction, own fields with the
                   oscillating N-form, tensor grid) -- independent of v3_tools / w_full.
Rows: the sigma -> 0 table (w1_klimit, k0 <= 6), the Monte-Carlo rows (w_twoway), and the g_min(sigma) optima.  -> results/w_indep_check.json.
Python 3.11."""
import json, glob, os
import numpy as np
from w_indep import strip_gap

out = {'klimit': [], 'twoway': [], 'gmin': []}
for r in json.load(open('results/w1_klimit.json'))['rows']:
    if r['k0'] > 6: continue
    g = strip_gap(1e-80, r['k0'], r['lam'], r['M'], 0.01, 1.0)
    out['klimit'].append({'k0': r['k0'], 'main': r['final_gap'], 'indep': g['gap_over_sigma'], 'rel': abs(g['gap_over_sigma'] / r['final_gap'] - 1)})
    print(out['klimit'][-1], flush=True)
if os.path.exists('results/w_twoway.json'):
    for r in json.load(open('results/w_twoway.json')):
        g = strip_gap(r['sigma'], r['k0'], r['lam'], r['M'], r['Lv_over_ell'], r.get('c', 1.0))
        out['twoway'].append({'k0': r['k0'], 'sigma': r['sigma'], 'main_grid': r['A_gap'], 'mc': r['B_gap'], 'mc_se': r['B_gap_se'], 'indep': g['gap_over_sigma'],
                              'rel_grid': abs(g['gap_over_sigma'] / r['A_gap'] - 1)}); print(out['twoway'][-1], flush=True)
for fn in sorted(glob.glob('results/w1_gmin_opt_s*.json')):            # main evaluator (w_full) optima, sigma <= 1e-6  ->  w_indep
    o = json.load(open(fn))
    if not o.get('finished'): continue
    b = o['best']; sg = o['sigma']; row = {'sigma': sg, 'source': 'w1_gmin (w_full)', 'main': b['final']['gap_over_sigma']}
    g = strip_gap(sg, b['k0'], b['lam'], b['M'], b['Lv_over_ell'], b['c'], n_theta=96, n_slow=24); row.update(method='w_indep', indep=g['gap_over_sigma'], y_indep=g['y_over_sigma'])
    row['rel'] = abs(row['indep'] / row['main'] - 1); out['gmin'].append(row); print(row, flush=True)
from v3_tools import ripple_terms_k, grid_sums
from u_cgm import nodes
for fn in sorted(glob.glob('results/w1_gmin_grid_opt_s*.json')):       # u_cgm optima, sigma >= 1e-5  ->  v3 state constructor + v3 point fields + own grid
    o = json.load(open(fn))
    if not o.get('finished'): continue
    b = o['best']; sg = o['sigma']; s = np.sqrt(sg); ell = b['lam'] * s; Lw = b['M'] * ell; Lv = b['Lv_over_sigma'] * sg
    T = ripple_terms_k(sg, Lv, Lw, b['c'], b['k0'], 1e30, ell); Tv, Tw = np.sqrt(Lv ** 2 + sg ** 2), np.sqrt(Lw ** 2 + sg ** 2)
    hv = min(0.25 * sg, 2 * np.pi * sg / b['k0'] / 16); vn = np.arange(-6.5 * Tv, 6.5 * Tv + hv / 2, hv)
    wn = np.unique(np.concatenate([np.linspace(-6.5 * Tw, 6.5 * Tw, 801), np.arange(-min(8 * ell, 6.5 * Tw), min(8 * ell, 6.5 * Tw), ell / 60)]))
    A = grid_sums(T, sg, vn, wn, rows=16)
    row = {'sigma': sg, 'source': 'w1_gmin_grid (u_cgm)', 'main': b['final']['gap_over_sigma'], 'method': 'v3 point fields, own grid', 'indep': A['gap_over_sigma_ibp']}
    if Lv * b['k0'] / (2 * np.pi * sg) >= 10:                           # enough periods for the two-scale evaluator: a third, separate value
        row['w_indep'] = strip_gap(sg, b['k0'], b['lam'], b['M'], Lv / ell, b['c'], n_theta=96, n_slow=32)['gap_over_sigma']
        row['rel_w_indep_vs_v3grid'] = abs(row['w_indep'] / row['indep'] - 1)
    row['rel'] = abs(row['indep'] / row['main'] - 1); out['gmin'].append(row); print(row, flush=True)
if os.path.exists('results/w1_gmin_indep_opt.json'):                   # w_indep optima, sigma <= 1e-24  ->  one-period limit (w_tools + v3; no V-spread term)
    from w_tools import evaluate
    for r in json.load(open('results/w1_gmin_indep_opt.json')).values():
        if r['sigma'] > 1e-24: continue
        o = evaluate([(1.0, 0.0, r['lam'] * r['M'])], [(r['c'], 0.0, r['lam'], 1)], r['k0'], sig=r['sigma'], Lv_sig=r['Lv_over_ell'] * r['lam'] / np.sqrt(r['sigma']), nper=40, coarse=400)
        vcost = (r['Lv_over_ell'] * r['lam']) ** 2 / 8          # the V-spread cost the one-period limit leaves out (estimate eta^2 lam^2 / 8)
        row = {'sigma': r['sigma'], 'source': 'w1_gmin_indep (w_indep)', 'main': r['final_gap'], 'method': 'one-period limit (w_tools/v3) + V-spread estimate',
               'indep': float(o['gap_over_sigma']) + vcost, 'vspread_estimate': vcost}
        row['rel'] = abs(row['indep'] / row['main'] - 1); out['gmin'].append(row); print(row, flush=True)
json.dump(out, open('results/w_indep_check.json', 'w'), indent=1)
