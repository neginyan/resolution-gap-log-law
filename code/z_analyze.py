"""Stage Z: summary of the step checks of Theorem Z and the figure (g_min of the strips with the theorem's lower bounds).
-> results/summary_Z.json, figures/stageZ_bound.png (+ .pdf).  Python 3.11."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
STEPS = ('S0', 'S1', 'S2', 'S3', 'S5', 'S6', 'S8', 'S56')
TWO_PI2E = 2 * (np.pi ** 2 + 1) / np.e                     # 2 (pi^2 + sigma^2)/e with sigma <= 1


def consts(K1, K2):
    """Theorem Z: gap/sigma >= (a/den)(1 - eps)(1 - eps_m) c / sqrt(ln(1/sigma) + C),  c = 1/(2 pi sqrt K1),  C = ln(2 (pi^2+1) K2 / e)/2."""
    return 1 / (2 * np.pi * np.sqrt(K1)), 0.5 * np.log(TWO_PI2E * K2)


def family(key, r):
    if 'family' in r: return r['family']
    if key.startswith('strip'): return 'strip (g_min optimum)'
    if key.startswith('stored'): return 'mixture (stored)'
    if key.startswith('t4_grating'): return 'grating mixture (adversarial)'
    return 'mixture (adversarial)'


def step_table(D):
    fam = {}
    for k, r in D.items():
        if 'error' in r: continue
        f = family(k, r); g = fam.setdefault(f, {'n': 0, 'n_y_le_half': 0})
        g['n'] += 1; g['n_y_le_half'] += int(r['y_over_sigma'] <= 0.5)
        for s in STEPS:
            if s not in r: continue
            lhs, rhs = r[s]
            if rhs is None or not np.isfinite(rhs) or not np.isfinite(lhs): continue
            q = lhs / rhs if rhs > 0 else np.inf; e = g.setdefault(s, {'min_ratio': np.inf, 'violations': 0})
            e['min_ratio'] = min(e['min_ratio'], q); e['violations'] += int(lhs < rhs * (1 - 1e-9))
        for q in ('K_Lambda', 'K_J', 'K_rho', 'eps_m', 'Xi', 'xbar_w', 'tail_mass_mz_gt_pi'):
            if q in r and np.isfinite(r[q]): g.setdefault(q + '_range', [np.inf, -np.inf]); g[q + '_range'] = [min(g[q + '_range'][0], r[q]), max(g[q + '_range'][1], r[q])]
        if 'bound_state' in r:
            g.setdefault('bound_le_gap', 0); g['bound_le_gap'] += int(r['bound_state'] <= r['gap_over_sigma']); g.setdefault('min_gap_over_bound', np.inf)
            g['min_gap_over_bound'] = min(g['min_gap_over_bound'], r['gap_over_sigma'] / r['bound_state'])
        if 'sqrtPw_bound_state' in r:
            g.setdefault('sqrtPw_over_bound_min', np.inf); g['sqrtPw_over_bound_min'] = min(g['sqrtPw_over_bound_min'], np.sqrt(r['P_w']) / r['sqrtPw_bound_state'])
    return fam


if __name__ == '__main__':
    X1 = json.load(open('results/z_chain_x1.json')); ST = json.load(open('results/z_chain_stored.json')); SP = json.load(open('results/z_chain_strips.json'))
    LA = json.load(open('results/z_chain_lattice.json'))['rows']; ID = json.load(open('results/z_identity.json'))
    T1 = {t['key']: t for t in json.load(open('results/t1_chain.json'))}
    out = {'steps_x1': step_table(X1), 'steps_stored': step_table(ST), 'steps_strips': step_table(SP)}
    # cross-checks against code-disjoint stored values
    cx = []
    for k, r in ST.items():
        if 'error' in r or k not in T1: continue
        cx.append((abs(r['Lambda'] / T1[k]['Lambda'] - 1), abs(r['P_sharp'] / T1[k]['P_sharp'] - 1), k))
    out['vs_t1_chain'] = {'n': len(cx), 'max_rel_Lambda': max(c[0] for c in cx), 'max_rel_Psharp': max(c[1] for c in cx), 'median_rel_Lambda': float(np.median([c[0] for c in cx])),
                          'worst': sorted(cx)[-3:]}
    out['vs_x1_P_w'] = max(abs(r['P_w'] / r['P_w_x1'] - 1) for r in X1.values() if 'P_w_x1' in r)
    out['grid_closed_L2'] = max(r['L2_rel_grid_vs_closed'] for D in (X1, ST) for r in D.values() if 'L2_rel_grid_vs_closed' in r and np.isfinite(r['L2_rel_grid_vs_closed']))
    out['grid_mass_dev'] = max(abs(r['grid_mass'] - 1) for D in (X1, ST) for r in D.values() if 'grid_mass' in r)
    out['strip_grid_checks'] = {k: SP[k]['grid_check'] for k in SP if 'grid_check' in SP[k]}
    out['lattice'] = {'K_Lambda_range': [min(r['K_Lambda'] for r in LA), max(r['K_Lambda'] for r in LA)], 'max_rel_vs_stageU': max(abs(r['Lambda'] / r['Lambda_stageU'] - 1) for r in LA)}
    out['identity'] = {'max_rel_Lambda_4xbar_sqrtp': max(r['rel_identity'] for r in ID), 'heis_ok': all(r['heis_lhs'] >= r['heis_rhs'] for r in ID),
                       'CR_ok': all(r['Jww'] >= r['CR_full'] >= r['CR_z'] for r in ID), 'jensen_ok': all(r['jensen_lhs'] <= r['jensen_rhs'] for r in ID), 'n': len(ID)}
    # strips: losses step by step and the theorem constants
    keys = sorted(SP, key=lambda k: -SP[k]['sigma']); rows = []
    K1 = max(SP[k]['K_Lambda'] for k in keys); K2 = max(SP[k]['K_J'] * SP[k]['K_rho'] for k in keys); c, C = consts(K1, K2)
    for k in keys:
        r = SP[k]; L = np.log(1 / r['sigma']); unif = c / np.sqrt(L + C)
        rows.append({'sigma': r['sigma'], 'gap_over_sigma': r['gap_over_sigma'], 'k0': r['k0'], 'xbar_w': r['xbar_w'], 'Xi': r['Xi'], 'K_Lambda': r['K_Lambda'], 'K_J': r['K_J'], 'K_rho': r['K_rho'],
                     'bound_state': r['bound_state'], 'bound_uniform_state_consts': r['bound_uniform'], 'theorem_curve': unif, 'gap_over_theorem': r['gap_over_sigma'] / unif,
                     'loss_S0': r['gap_over_sigma'] / (r['den_factor'] * np.sqrt(r['P_w'])), 'loss_S1': np.sqrt(r['P_w'] / (2 / np.pi ** 2 * r['P_sharp'])),
                     'loss_S2': np.sqrt(r['P_sharp'] / r['P_y']) / (1 - r['eps_m']), 'loss_S3': np.sqrt(r['P_y'] * r['Lambda']), 'loss_S4': np.sqrt(max(r['K_Lambda'], 1e-300) / r['K_Lambda']) if False else np.sqrt(1.0),
                     'loss_S56': np.sqrt(r['Xi'] / r['xbar_w']), 'loss_uniform_log': np.sqrt((2 * L + np.log(2 * (np.pi ** 2 + 1) * K2 / np.e)) / r['Xi']),
                     'eps': r['eps'], 'eps_eff': r['eps_eff'], 'eps_m': r['eps_m'], 'y_over_sigma': r['y_over_sigma'], 'den_factor': r['den_factor']})
    out['strips'] = rows; out['theorem_constants_strips'] = {'K1_max_K_Lambda': K1, 'K2_max_KJ_Krho': K2, 'c': c, 'C': C, 'A_loglaw_fit': 1.0241, 'A_over_c': 1.0241 / c}
    # constants for the other families (Z2)
    fams = {}
    for D in (X1, ST):
        for k, r in D.items():
            if 'error' in r or 'K_Lambda' not in r: continue
            f = family(k, r); e = fams.setdefault(f, {'K_Lambda': [], 'K_J': [], 'K_rho': [], 'y_le_half': []})
            e['K_Lambda'].append(r['K_Lambda']); e['K_J'].append(r['K_J']); e['K_rho'].append(r['K_rho']); e['y_le_half'].append(r['y_over_sigma'] <= 0.5)
    out['K_by_family'] = {f: {q: [float(np.min(v[q])), float(np.max(v[q]))] for q in ('K_Lambda', 'K_J', 'K_rho')} | {'n': len(v['K_Lambda']),
                          'K_Lambda_max_y_le_half': float(max([a for a, b in zip(v['K_Lambda'], v['y_le_half']) if b], default=np.nan)),
                          'c_y_le_half': float(1 / (2 * np.pi * np.sqrt(max([a for a, b in zip(v['K_Lambda'], v['y_le_half']) if b], default=np.nan)))),
                          'C_y_le_half': float(0.5 * np.log(TWO_PI2E * max([a * b for a, b, y in zip(v['K_J'], v['K_rho'], v['y_le_half']) if y], default=np.nan)))} for f, v in fams.items()}
    json.dump(out, open('results/summary_Z.json', 'w'), indent=1, default=float)
    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    a = ax[0]; s = np.array([r['sigma'] for r in rows]); Lx = np.log(1 / s)
    a.plot(Lx, [r['gap_over_sigma'] for r in rows], 'o', ms=6, color=OI[0], label='strips: $g_{min}(\\sigma)$ (numerical)')
    ll = np.geomspace(4, 260, 200); a.plot(ll[ll > 26], 1.0241 / np.sqrt(ll[ll > 26] - 23.40), '-', color=OI[0], lw=0.8, alpha=0.5, label='fit $1.02/\\sqrt{\\ln(1/\\sigma) - 23.4}$')
    a.plot(Lx, [r['bound_state'] for r in rows], 's', ms=5, mfc='none', mew=1.4, color=OI[5], label='Theorem Z, state constants ($\\Xi$, $K_\\Lambda$)')
    a.plot(Lx, [r['bound_uniform_state_consts'] for r in rows], '^', ms=5, mfc='none', mew=1.4, color=OI[3], label='Theorem Z, uniform form, state $K$')
    a.plot(ll, c / np.sqrt(ll + C), '-', color=OI[6], lw=1.6, label='Theorem Z: $c/\\sqrt{\\ln(1/\\sigma)+C}$, c = %.3f, C = %.2f' % (c, C))
    a.axhline(0.884, color=OI[1], lw=1, ls='--', label='mixtures 0.884')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('ln(1/$\\sigma$)'); a.set_ylabel('gap/$\\sigma$'); a.set_ylim(0.01, 14)
    a.set_title('strips: numerical minimum above the proved lower bounds'); a.legend(frameon=False, fontsize=6.5, loc='upper right')
    b = ax[1]
    for q, lab, col in (('loss_S1', 'step 1: $\\phi \\geq 2m^2/\\pi^2$, $1-\\phi/4 \\geq 1/2$', OI[6]), ('loss_S3', 'step 3: Heisenberg $P_y \\geq 1/\\Lambda$', OI[5]),
                        ('loss_S56', 'steps 5-7: Jensen + Cramer-Rao ($\\Xi$ vs $\\bar x_w$)', OI[3]), ('loss_uniform_log', 'uniform: $2\\ln(1/\\sigma)$ + C vs $\\Xi$', OI[2]),
                        ('loss_S0', 'step 0: Theorem W$\'$ (AM-GM)', OI[1])):
        b.plot(Lx, [r[q] for r in rows], 'o-', ms=4, color=col, label=lab)
    b.axhline(1, color=OI[0], lw=0.8); b.set_xscale('log'); b.set_xlabel('ln(1/$\\sigma$)'); b.set_ylabel('factor lost in gap/$\\sigma$ (1 = tight)')
    b.set_title('where the bound is loose (strip optima)'); b.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig('figures/stageZ_bound.png', dpi=160); fig.savefig('figures/stageZ_bound.pdf'); plt.close(fig)
    print(json.dumps({k: out[k] for k in ('vs_t1_chain', 'vs_x1_P_w', 'grid_closed_L2', 'grid_mass_dev', 'lattice', 'identity', 'theorem_constants_strips')}, indent=1, default=float))
    for r in rows: print('sigma %.0e gap %.4f bound_state %.4f unif(state K) %.4f curve %.4f ratio %.1f | losses S0 %.3f S1 %.3f S3 %.3f S56 %.3f unif %.3f K_Lam %.3f' % (
        r['sigma'], r['gap_over_sigma'], r['bound_state'], r['bound_uniform_state_consts'], r['theorem_curve'], r['gap_over_theorem'], r['loss_S0'], r['loss_S1'], r['loss_S3'], r['loss_S56'], r['loss_uniform_log'], r['K_Lambda']))
