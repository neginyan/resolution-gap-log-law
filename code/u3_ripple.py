"""Stage U3 (ripple side): the two correction factors of R4 on the ripple states, where y/sigma is small.
  delta = E[(sigma^2 H_vv)_-] / a,   T2/T1 = (sigma^2 E[u^2 s_v^2] - sigma^2 E[u^2 H_vv]) / (sigma^2 E[u^2 s_v^2]),   P = Z J_ww / j^2.
(i) periodic lattice (u1_lattice.lattice_fields; u is a placeholder there, so only delta and Dbar), amplitudes c = 1 ... 0.01;
(ii) the localised ripple maximisers of u1_ripple_local (exact fields, u_cgm).  -> results/u3_ripple.json.  Python 3.11 compatible."""
import json, glob, os
import numpy as np
from u1_lattice import lattice_fields
from u_tools import u_terms
from u_cgm import evaluate, ripple_terms, nodes


def corr(f):
    s2 = f['sig'] ** 2; P = f['pw']; sv, Hvv, u = f['sv'], f['Hvv'], f['u']; E = lambda q: float(np.sum(P * q))
    a = s2 * E(sv * sv); T1 = s2 * E(u * u * sv * sv); T2 = T1 - s2 * E(u * u * Hvv)
    return {'delta': E(np.maximum(-s2 * Hvv, 0)) / a, 'T2_over_T1': T2 / T1, 'a': a}


if __name__ == '__main__':
    out = {'lattice': [], 'ripple_local': []}
    for k0 in (0.5, 1.0, 1.5, 2.0, 3.0):
        for c in (1.0, 0.3, 0.1, 0.03, 0.01):
            f, A = lattice_fields(k0, 0.3, c, n=128); q = corr(f); u = u_terms(f)
            P = f['pw']; j = float(np.sum(P * np.clip(np.maximum(f['Hvv'], 0), 0, 1))); y = float(np.sum(P * f['sw'] ** 2)) / j
            out['lattice'].append({'k0': k0, 'c': c, 'A': A, 'delta': q['delta'], 'Dbar': u['Dbar'], 'aIpt_over_Jww': u['aIpt_over_Jww'], 'y_over_sigma_at_sigma1': y})
    for fn in sorted(glob.glob('results/u1_ripple_local_s*_Y*_seed*.json')):
        o = json.load(open(fn)); b = o.get('best')
        if not b: continue
        sig = o['sigma']; Y0 = o['Y0']
        def conv(par):      # converged grid: W spacing halved w.r.t. the search grid (omega has narrow minima along W where the ripple strength crosses 0;
                            # factors 1.5 / 1 / 0.7 / 0.5 / 0.35 of ell/40 give 3.932 / 3.879 / 3.845 / 3.8806 / 3.8799 on the sigma = 1e-2 maximiser;
                            # sigma = 1e-3: 0.5 / 0.35 / 0.25 give 4.3751 / 4.3946 / 4.3945, so 0.35 is used)
            T = ripple_terms(sig, par['Lv'], par['Lw'], par['c'], par['k0'], par['Rv'], par['ell'], par['q0'])
            hv = min(0.2 * sig, 2 * np.pi * sig / par['k0'] / 20)
            hw = 0.35 * min(par['ell'] / 40, (2 * np.pi * sig / par['q0']) / 24 if par['q0'] > 0 else 1e9, 0.5 * np.sqrt(sig))
            vn = nodes(6.5 * par['Lv'], hv, 6.5 * par['Lv'], hv); wn = nodes(6.5 * par['Lw'], hw, 6.5 * par['Lw'], hw)
            while len(vn) * len(wn) > 8.0e6:                 # ~1.3 GB
                hw *= 1.1; wn = nodes(6.5 * par['Lw'], hw, 6.5 * par['Lw'], hw)
            return evaluate(T, sig, vn, wn)
        par = dict(b['par']); f, dg = conv(par); shrink = 1.0
        if dg['y_over_sigma'] > Y0 * (1 - 1e-3):          # constrained search: rescale q0 (the 4 k0^2 mechanism does not depend on q0) until feasible
            lo, hi = 0.5, 1.0
            for _ in range(12):
                m = (lo + hi) / 2; p2 = dict(par, q0=par['q0'] * m); f2, d2 = conv(p2)
                if d2['y_over_sigma'] <= Y0 * (1 - 1e-3): lo = m
                else: hi = m
            shrink = lo; par = dict(par, q0=par['q0'] * lo); f, dg = conv(par)
        q = corr(f); u = u_terms(f); from t_tools import t_terms; t = t_terms(f)
        units = dict(b['par_units']); units['q0_over_sqrt_sigma'] = par['q0'] / np.sqrt(sig)
        out['ripple_local'].append({'file': os.path.basename(fn), 'sigma': sig, 'Y0': Y0, 'q0_shrink_for_feasibility': shrink, 'opt_aIpt_search_grid': b['opt_aIpt_over_Jww'],
                                    'delta': q['delta'], 'T2_over_T1': q['T2_over_T1'], 'Dbar': u['Dbar'],
                                    'aIpt_over_Jww': u['aIpt_over_Jww'], 'term_Wd': u['term_Wd'], 'term_cross': u['term_cross'], 'term_omega_sW2': u['term_omega_sW2'],
                                    'bound_D': u['bound_D'], 'Lambda': u['Lambda'], 'h_Hvw': u['h_Hvw'], 'Lambda0': u['Lambda0'], 'P_sharp': t['P_sharp'], 'bound_ptilde': t['bound_ptilde'],
                                    'x_minus_1': t['x_minus_1'], 'P': dg['P'], 'F': dg['F'], 'gap_over_sigma': dg['gap_over_sigma'], 'y_over_sigma': dg['y_over_sigma'],
                                    'grid_mass': dg['grid_mass'], 'n_points': dg['n_points'], 'par_units': units})
    json.dump(out, open('results/u3_ripple.json', 'w'), indent=1, default=float)
    for r in out['ripple_local']: print(r['file'], {k: round(r[k], 4) for k in ('aIpt_over_Jww', 'Dbar', 'bound_D', 'delta', 'T2_over_T1', 'P_sharp', 'P', 'F', 'gap_over_sigma', 'y_over_sigma', 'q0_shrink_for_feasibility')})
