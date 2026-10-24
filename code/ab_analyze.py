"""Stage AB: summary (results/summary_AB.json) and figure (figures/stageAB.png/.pdf).  Python 3.11."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

if __name__ == '__main__':
    TP = json.load(open('results/ab_twopoint.json')); LA = json.load(open('results/ab_lattice.json')); FI = json.load(open('results/ab_finite.json'))
    T1 = json.load(open('results/ab_tail_x1.json')); TS = json.load(open('results/ab_tail_stored.json')); ST = json.load(open('results/ab_strip_tail.json'))
    A = json.load(open('results/aa_chain.json'))
    lat = {k: v for k, v in LA.items() if k.startswith(('single', 'double'))}; lbest = max(lat.items(), key=lambda kv: kv[1]['F'])
    adv = {k: v for k, v in FI.items() if k.startswith('adv')}; abest = max(adv.items(), key=lambda kv: kv[1]['fine']['F'])
    Zs = [v for f in ('x1', 'stored', 'strips') for v in json.load(open('results/z_chain_%s.json' % f)).values()]
    Fz = max(np.log(v['sigma'] ** 2 * v['J_ww']) + v['Lambda'] / 4 for v in Zs)
    A0 = lbest[1]['F']
    out = {'two_point_main_line': TP['main_line_example'], 'two_point_sup': TP['sup_two_points'],
           'diamond_finite': {k: FI[k]['F'] for k in FI if k.startswith('diamond')}, 'chain12': FI['chain12']['F'],
           'diamond_infinite': LA['diamond_main_line']['128']['F'] if '128' in LA['diamond_main_line'] else LA['diamond_main_line'][128]['F'],
           'lattice_sup': {'key': lbest[0], 'F': A0, 'basis': lbest[1]['basis'], 'N256': lbest[1]['check_N256']['F'], 'J_ww': lbest[1]['J_ww'], 'Lambda': lbest[1]['Lambda']},
           'lattice_runs': {k: v['F'] for k, v in lat.items()}, 'adversarial_finite_best': {'key': abest[0], 'F': abest[1]['fine']['F']},
           'adversarial_finite_all': {k: v['fine']['F'] for k, v in adv.items()}, 'F_max_stage_Z_states': Fz}
    # diamond 1/n extrapolation
    ns = np.array([int(k[9:]) for k in out['diamond_finite']]); Fs = np.array(list(out['diamond_finite'].values())); o = np.argsort(ns); ns, Fs = ns[o], Fs[o]
    cf = np.polyfit(1 / ns[-3:], Fs[-3:], 1); out['diamond_extrapolated_1_over_n'] = float(cf[1])
    # Theorem Z'' with the conjecture Z2-F' (A0): c = 1/(2 pi), C = ln(2 (pi^2 + sigma^2))/2 + A0/2
    out['conjecture_constants'] = {'A0_estimate': A0, 'c': 1 / (2 * np.pi), 'C_sigma1': 0.5 * np.log(2 * (np.pi ** 2 + 1)) + A0 / 2, 'C_sigma0': 0.5 * np.log(2 * np.pi ** 2) + A0 / 2}
    out['t_half_C'] = {'sigma1': 0.5 * np.log(32 * (np.pi ** 2 + 0.5) / np.e), 'sigma0': 0.5 * np.log(32 * np.pi ** 2 / np.e)}
    # H0 removal
    mix = {**{'x1:' + k: v for k, v in T1.items()}, **{'st:' + k: v for k, v in TS.items()}}
    eq = {}
    for k, v in A.items():
        key = k.rsplit(':', 1)[0]; eq[key] = v['E_Q_yz2']
    rows = []
    for k, v in mix.items():
        if 'error' in v: continue
        rows.append({'key': k, 'eps_pi': v['eps_pi'], 'tail_mass': v['tail_mass'], 'Var_X_z': v['Var_rho_z'], 'E_Q': eq.get(k, np.nan)})
    out['H0_mixtures'] = {'n': len(rows), 'max_eps_pi': max(r['eps_pi'] for r in rows), 'n_eps_pi_pos': sum(r['eps_pi'] > 0 for r in rows),
                          'max_Var_X_z': max(r['Var_X_z'] for r in rows), 'n_Var_gt_pi2': sum(r['Var_X_z'] > np.pi ** 2 for r in rows),
                          'max_E_Q': max(r['E_Q'] for r in rows if np.isfinite(r['E_Q'])), 'worst': sorted(rows, key=lambda r: -r['eps_pi'])[:5]}
    out['H0_strips'] = ST
    LS = json.load(open('results/ab_lattice_scan.json')); out['lattice_scan'] = {'n_runs': len(LS['runs']), 'best_F': LS['best']['F_N192'],
        'best_by_points_per_cell': {k: max(r['F_N192'] for r in LS['runs'] if r['points_per_cell'] == k and np.isfinite(r['F_N192'])) for k in (1, 2, 3)}}
    # step 1 without H0: P_w >= (2/pi^2) P# (1 - eps_pi^2) on every mixture state; Theorem Z'' (state constants) below the gap
    ZZ = {**{'x1:' + k: v for k, v in json.load(open('results/z_chain_x1.json')).items()}, **{'st:' + k: v for k, v in json.load(open('results/z_chain_stored.json')).items()}}
    s1 = [(ZZ[r['key']]['P_w'] / (2 / np.pi ** 2 * ZZ[r['key']]['P_sharp'] * (1 - r['eps_pi'] ** 2)), r['key']) for r in rows]
    out['step1_noH0'] = {'n': len(s1), 'min_ratio': min(s1)[0], 'at': min(s1)[1]}
    epi = {r['key']: r['eps_pi'] for r in rows}
    g = []
    for k, v in A.items():
        if 'bound_state' not in v: continue
        key = k.rsplit(':', 1)[0]
        e = epi.get(key, 10 ** min(ST[key[3:]]['log10_eps_pi_bound'], 0) if key.startswith('sp:') else 0.0)
        g.append((v['gap_over_sigma'] / (v['bound_state'] * np.sqrt(1 - e * e)), key))
    out['theoremZpp_gap_over_bound_min'] = min(g)
    json.dump(out, open('results/summary_AB.json', 'w'), indent=1, default=float)
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2)); a = ax[0]
    sc = TP['scan_theta45']; a.plot([r['d'] for r in sc], [r['F'] for r in sc], '-', color=OI[5], label='two points, $\\Delta$ at 45°: F(|$\\Delta$|)')
    a.plot([TP['sup_two_points']['d']], [TP['sup_two_points']['F']], '*', ms=11, color=OI[5], label='two points, sup = %.4f (main line 0.2909)' % TP['sup_two_points']['F'])
    a.axhline(0.25, color=OI[0], lw=0.8, ls=':', label='single Gaussian: 1/4')
    a.axhline(A0, color=OI[6], lw=1.4, label='infinite lattice, sup over bases = %.4f' % A0)
    a.plot([5.5] * 0, [])
    for n, F in zip(ns, Fs): a.plot([3.5 * np.sqrt(2) * 0 + 6 + n * 0.3], [F], 's', ms=5, mfc='none', color=OI[3])
    a.plot([], [], 's', mfc='none', color=OI[3], label='diamond n x n, n = 3 ... 11 (placed at x = 6 + 0.3 n)')
    a.axhline(out['diamond_infinite'], color=OI[3], lw=0.8, ls='--', label='diamond, infinite lattice: %.4f' % out['diamond_infinite'])
    for k, F in out['adversarial_finite_all'].items(): a.plot([9.5], [F], 'o', ms=4, mfc='none', color=OI[1])
    a.plot([], [], 'o', mfc='none', color=OI[1], label='adversarial finite mixtures (K = 3, 6, 9)')
    a.set_xlabel('|$\\Delta$| (spacing, units of $\\sigma$)'); a.set_ylabel('F = ln($\\sigma^2 J_{ww}$) + $\\Lambda$/4'); a.set_ylim(-0.6, 0.6)
    a.set_title('conjecture Z2-F: sup F is finite, about 0.47'); a.legend(frameon=False, fontsize=6.2, loc='lower right')
    b = ax[1]
    sg = [r for r in rows]; ys = [max(r['eps_pi'], 1e-17) for r in sg]
    b.plot([np.sqrt(r['Var_X_z']) for r in sg], ys, 'o', ms=3, mfc='none', color=OI[5], alpha=0.7, label='mixtures (measured $\\varepsilon_\\pi$; 0 drawn at 1e-17)')
    b.plot([np.sqrt(v['Var_X_z']) for v in ST.values()], [10 ** max(v['log10_eps_pi_bound'], -17) for v in ST.values()], '^', ms=6, color=OI[6], label='strips (rigorous bound; < 1e-17 drawn at 1e-17)')
    b.axvline(np.pi, color=OI[0], lw=0.8, ls=':', label='sd = $\\pi$')
    b.set_xscale('log'); b.set_yscale('log'); b.set_xlabel('sd($X_z$)'); b.set_ylabel('$\\varepsilon_\\pi$'); b.set_title('H0 removed: the tail correction $\\varepsilon_\\pi$')
    b.legend(frameon=False, fontsize=6.5, loc='upper left')
    fig.tight_layout(); fig.savefig('figures/stageAB.png', dpi=160); fig.savefig('figures/stageAB.pdf'); plt.close(fig)
    print(json.dumps({k: out[k] for k in ('lattice_scan', 'step1_noH0', 'theoremZpp_gap_over_bound_min', 'two_point_sup', 'diamond_finite', 'diamond_extrapolated_1_over_n', 'diamond_infinite', 'lattice_sup', 'adversarial_finite_best', 'F_max_stage_Z_states', 'conjecture_constants', 't_half_C', 'H0_mixtures')}, indent=1, default=float))
