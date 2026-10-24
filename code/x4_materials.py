"""Stage X4: materials for an independent check by the main line: the best ripple-strip states at sigma = 1e-4, 1e-6, 1e-12, 1e-24, every term of
1 - R = (2 sigma^2 J_ww + N)/(sigma^2 (J_vv + J_ww)), N = E[s_p D] = -E[d_p D], each computed by two evaluators that share no code:
  A: x_indep (closed-form blurred density, posterior-covariance form of N, two-scale Gauss quadrature);
  B: sigma = 1e-4, 1e-6: v3_tools.grid_sums on a full 2-d grid with v3_tools.ripple_terms_k states; sigma = 1e-12, 1e-24: w_full (true V-envelope,
     w_tools states, v3 point fields, d_p D form of N; its O(period/Lv) aliasing error is ~1e-5 and ~1e-11 there).
-> results/x4_states_for_main.json.  Python 3.11."""
import json
import numpy as np
from x_indep import multi_gap

GI = json.load(open('results/w1_gmin_indep_opt.json')); GG = json.load(open('results/w1_gmin_grid_opt_s0.0001.json'))['best']
states = []
b = GG; sg = 1e-4; s = np.sqrt(sg)
states.append({'sigma': sg, 'k0': b['k0'], 'lam': b['lam'], 'M': b['M'], 'c': b['c'], 'Lv_over_sigma': b['Lv_over_sigma'], 'origin': 'w1_gmin_grid_opt_s0.0001 (u_cgm full-grid search)'})
for key in ('1e-06', '1e-12', '1e-24'):
    r = GI[key]; sg = r['sigma']
    states.append({'sigma': sg, 'k0': r['k0'], 'lam': r['lam'], 'M': r['M'], 'c': r['c'], 'Lv_over_sigma': r['Lv_over_ell'] * r['lam'] / np.sqrt(sg), 'origin': 'w1_gmin_indep_opt (separate-evaluator search)'})
KEYS = ('gap_over_sigma', 'J_vv', 'J_ww', 'N', 'a', 'j', 'y_over_sigma')
out = {'state_definition': {
    'coordinates': 'pendulum H = p^2/2 - cos q (kappa = 1), top at (q, p) = (pi, 0); v = (q - pi + p)/sqrt2, w = (q - pi - p)/sqrt2; z := q - pi = (v + w)/sqrt2, p = (v - w)/sqrt2',
    'rho': 'rho(v, w) = N(v; 0, Lv^2) N(w; 0, Lw^2) [1 + c exp(-w^2 / (2 ell^2)) cos(k0 v / sigma)],  ell = lam sqrt(sigma), Lw = M ell, Lv = (Lv/sigma) sigma',
    'observation': 'y = x + sigma Z, Z ~ N(0, I_2); p = rho * N(0, sigma^2 I); s = grad log p (in v, w); s_p = (s_v - s_w)/sqrt2',
    'gap': '1 - R = (2 sigma^2 J_ww + N) / (sigma^2 (J_vv + J_ww)),  N = E[s_p D] = -E[d_p D],  D(y) = E[z - sin z | y];  gap/sigma = (1 - R)/sigma',
    'other': 'a = sigma^2 J_vv (= E[sigma^2 H_vv]); j = E[min(max(sigma^2 H_vv, 0), 1)]; y/sigma = sigma J_ww / j (constraint y/sigma <= 1/2)',
    'blurred_density_closed_form': 'p(y) = N(y_v; 0, Tv) N(y_w; 0, Tw) [1 + c e^{-u^2/2} A(y_w) cos(k0 Kv y_v / sigma)], Tv = Lv^2 + sigma^2, Kv = Lv^2/Tv, '
                                   's_v^2 = Lv^2 sigma^2 / Tv, u = k0 s_v / sigma, A(y_w) = ell/sqrt(ell^2 + s_w^2) exp(-(Kw y_w)^2 / (2 (ell^2 + s_w^2))) (same notation for w)',
    'numerics_note': 'many ripple periods fit in the strip (Lv >> 2 pi sigma/k0): averaging over the phase theta = k0 Kv y_v/sigma at fixed slow y_v is accurate to '
                     '(period/Lv)^2; positions v0 + (fraction of a period) lose the phase in double precision once Lv/sigma >~ 1e12 (stage W)'}, 'states': []}
from w_tools import terms_of, w_nodes
from w_full import full_sums
from v3_tools import ripple_terms_k, grid_sums
for st in states:
    sg = st['sigma']; s = np.sqrt(sg); ell = st['lam'] * s; Lw = st['M'] * ell; Lv = st['Lv_over_sigma'] * sg; eta = Lv / ell
    A = multi_gap(sg, st['k0'], [(st['c'], 0.0, st['lam'], 1, 0.0)], st['M'], eta, n_theta=96, n_slow=32, n_gh=64, n_gl=16, panels_per_ell=5, bg_panels=96)
    if sg >= 1e-6:      # full 2-d grid (w_full has an O(period/Lv) aliasing error, ~3e-3 at sigma = 1e-6: stage X finding)
        T = ripple_terms_k(sg, Lv, Lw, st['c'], st['k0'], 1e30, ell); Tv, Tw = np.sqrt(Lv ** 2 + sg ** 2), np.sqrt(Lw ** 2 + sg ** 2)
        hv = min(0.25 * sg, 2 * np.pi * sg / st['k0'] / 16); vn = np.arange(-6.5 * Tv, 6.5 * Tv + hv / 2, hv)
        wn = np.unique(np.concatenate([np.linspace(-6.5 * Tw, 6.5 * Tw, 801), np.arange(-min(8 * ell, 6.5 * Tw), min(8 * ell, 6.5 * Tw), ell / 60)]))
        Bo = grid_sums(T, sg, vn, wn, rows=16); Bo['N'] = Bo['N_ibp']; Bo['gap_over_sigma'] = Bo['gap_over_sigma_ibp']; methB = 'v3 grid_sums, full 2-d grid'
    else:
        bg = [(1.0, 0.0, st['lam'] * st['M'])]; win = [(st['c'], 0.0, st['lam'], 1)]; Lvs = st['Lv_over_sigma']
        Bo = full_sums(terms_of(bg, win, st['k0'], sg, Lvs), sg, st['k0'], np.sqrt(Lv ** 2 + sg ** 2), w_nodes(bg, win, sg, 40, 400), nv=32, n0=31, ext=6.5); methB = 'w_full (true V-envelope)'
    Bo['a'] = sg * sg * Bo['J_vv']
    rowA = {k: float(A[k]) for k in KEYS}; rowB = {k: float(Bo[k]) for k in KEYS}
    for row, src in ((rowA, A), (rowB, Bo)):
        row['terms'] = {'2 sigma^2 J_ww': 2 * sg * sg * row['J_ww'], 'N = E[s_p D]': row['N'], 'numerator': 2 * sg * sg * row['J_ww'] + row['N'],
                        'sigma^2 J_vv': sg * sg * row['J_vv'], 'sigma^2 J_ww': sg * sg * row['J_ww'], 'denominator': sg * sg * (row['J_vv'] + row['J_ww']),
                        '1 - R': (2 * sg * sg * row['J_ww'] + row['N']) / (sg * sg * (row['J_vv'] + row['J_ww']))}
    rel = {k: abs(rowA[k] / rowB[k] - 1) for k in KEYS}
    out['states'].append({'parameters': {**st, 'ell': ell, 'Lw': Lw, 'Lv': Lv, 'periods_in_Lv': Lv * st['k0'] / (2 * np.pi * sg)}, 'A_x_indep': rowA, 'B': rowB, 'B_method': methB, 'rel_A_vs_B': rel})
    print('sigma %g: gap A %.6f B %.6f; max rel over terms %.1e' % (sg, rowA['gap_over_sigma'], rowB['gap_over_sigma'], max(rel.values())), flush=True)
json.dump(out, open('results/x4_states_for_main.json', 'w'), indent=1)
