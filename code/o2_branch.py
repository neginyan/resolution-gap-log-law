"""Stage O, task O2(b),(c): the large-y branch as an exact inequality, and the two proof routes on all states.

Exact rewriting (stage J/M notation; H = -grad^2 log p, w = sigma^2 (H_vv)_+ in [0, 1] -- sigma^2 H_vv <= 1 always, because
H = (I - Cov(X|y)/sigma^2)/sigma^2 --, g = 1 - abar = E[sin^2(delta/2)|y] in [0, 1], delta = X_q - pi, J_tot = E[H_vv + H_ww] > 0):
    1 - f = E[g H_vv + (2 - g) H_ww]/J_tot = (E[g w]/sigma^2 - N + kappa2 J_ww)/J_tot                       (identity)
    N = E[g (H_vv)_-],  kappa2 = 2 - E[g H_ww]/J_ww,  1 - R = 1 - f - D,  D = R - f.
(B1) large-y branch (drop E[g w] >= 0; J_tot = j/sigma^2 + J_ww - N', N' = E[(H_vv)_-] >= 0, so J_ww/J_tot >= y/(1 + y)):
    (1 - R)/sigma >= [ kappa2 J_ww/J_tot - N/J_tot - D ]/sigma >= [ kappa2 y/(1 + y) - N/J_tot - D ]/sigma     (kappa2 J_ww >= 0)
    with y = sigma^2 J_ww/j.  The "~" of stage M used three approximations: E[g w] ~ Z/8 (kept in B2), J_tot ~ j/sigma^2 + J_ww
    (exact iff H_vv >= 0 everywhere; otherwise the inequality goes the safe way), and the dropped corrections N/J_tot and D.
(B2) focus-distance branch (Z-bound of stage M, rigorous):  E[g w] >= Z/8 - corr  (sin^2(x/2) >= x^2/4 - x^4/48),
    (1 - R)/sigma >= [ ((Z/8 - corr)/sigma^2 - N + kappa2 J_ww)/J_tot - D ]/sigma.
(B3) the state-wise theorem of paper B (Cramer-Rao route):  Theta' sigma/(1 + Theta' sigma/2) - eps_tilde - D   (l_tools.diag).
For every state: the gap, B1 (both forms), B2, B3, and max(B1, B2, B3) = the best proved state-wise bound; d, y/sigma, rho, kappa2,
N/J_tot, D.  States: stored states of stages A-L (m4 list), the N2 P-minimisers, the O2(a) rho-maximisers and the O2 route
minimisers (o2_route.py).   -> results/o2_branch.json"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from j2_smooth import arrays
from l_tools import diag


def bounds(ws, mus, Ss, sig, per_sd=6):
    A = arrays(ws, mus, Ss, sig, per_sd=per_sd, max_points=int(min(2_500_000, 2.4e7 / len(ws)))); p = A['pw']; s2 = A['s2']
    w, u, g = A['w'], A['u'], A['g']; j = np.sum(p * w); Z = np.sum(p * w * u * u); Hvv, Hww = A['Hvv'], A['Hww']
    Jww = np.sum(p * Hww); Jtot = np.sum(p * (Hvv + Hww)); N = np.sum(p * g * np.maximum(-Hvv, 0)); Np = np.sum(p * np.maximum(-Hvv, 0))
    k2 = 2 - np.sum(p * g * Hww) / Jww; corr = np.sum(p * w * (A['Ed4'] / 48 - A['varq'] / 4)); D = A['R'] - A['f']; y = s2 * Jww / j
    Egw = np.sum(p * g * w)
    return {'gap': (1 - A['R']) / sig, 'identity_check': ((Egw / s2 - N + k2 * Jww) / Jtot - (1 - A['f'])), 'B1_exact': ((k2 * Jww - N) / Jtot - D) / sig,
            'B1': (k2 * y / (1 + y) - N / Jtot - D) / sig, 'B2': (((Z / 8 - corr) / s2 - N + k2 * Jww) / Jtot - D) / sig, 'kappa2': k2, 'N_over_Jtot_sigma': N / Jtot / sig,
            'Nprime_sigma2_over_j': Np * s2 / j, 'D_over_sigma': D / sig, 'y_over_sigma': y / sig, 'd': np.sqrt(Z / (j * sig)), 'Egw_term': Egw / s2 / Jtot / sig}


def run(st):
    tag, ws, mus, Ss, sig = st
    try:
        b = bounds(ws, mus, Ss, sig); dg = diag(ws, mus, Ss, sig)
    except (MemoryError, ValueError, np.linalg.LinAlgError) as e:
        return {'tag': tag, 'sigma': sig, 'error': str(e)}
    b.update({'tag': tag, 'sigma': sig, 'B3': dg['bound_over_sigma'], 'rho': dg['rho'], 'geo': dg['geo'], 'ThetaP': dg['ThetaP']})
    b['best'] = max(b['B1'], b['B2'], b['B3']); return {k: float(v) if not isinstance(v, str) else v for k, v in b.items()}


def all_states():
    from i2_bound import states
    from l1_ratios import new_states
    from h2_adversarial import unpack
    S = [s for s in states() + new_states() if s[4] <= 1e-2]
    for fn, key in [('n2_pmin', None), ('o2_rhomax', None), ('o2_route', None)]:
        if os.path.exists(f'results/{fn}.json'):
            for o in json.load(open(f'results/{fn}.json')):
                ws, mus, Ss = unpack(np.array(o['x']), o['K'], o['sigma']); S.append((f"{fn}_K{o['K']}_s{o['sigma']:g}_{o.get('Y', o.get('ymax'))}", ws, mus, Ss, o['sigma']))
    return S


if __name__ == '__main__':
    S = all_states(); print(len(S), 'states', flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as pool: out = pool.map(run, S, chunksize=4)
    json.dump(out, open('results/o2_branch.json', 'w'), indent=1, default=float)
    ok = [o for o in out if 'error' not in o]
    print('errors', len(out) - len(ok), ' identity max', max(abs(o['identity_check']) for o in ok))
    print('B1 violations', sum(o['B1'] > o['gap'] + 1e-9 for o in ok), ' B1_exact >= B1 always', all(o['B1_exact'] >= o['B1'] - 1e-12 for o in ok))
    print('min best bound', min(o['best'] for o in ok), ' min gap', min(o['gap'] for o in ok))
