"""Stage AC: summary (results/summary_AC.json) and figure (figures/stageAC.png/.pdf).  Python 3.11."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
A0 = 0.47097204431532214


def deb_log(r):
    """de Bruijn check from the stored rows with the central difference of ln J."""
    out = {}
    for m in (0.5, 1.0):
        a, b, c = r['rows']['%.3f' % (m * 0.99)], r['rows']['%.3f' % (m * 1.01)], r['rows']['%.3f' % m]
        d = c['J'] * np.log(b['J'] / a['J']) / (b['t'] - a['t']); out['%.1f' % m] = abs(d / (-(c['Hvw2'] + c['Hww2'])) - 1)
    return max(out.values())


if __name__ == '__main__':
    H = {}
    for mode in ('lattice', 'points', 'mix', 'strip'):
        for k, v in json.load(open('results/ac_heat_%s.json' % mode)).items(): H[mode + ':' + k] = v
    out = {'n_states': len(H)}
    L1 = [(min(r['L1'][0] / r['L1'][1] for r in v['rows'].values()), k) for k, v in H.items()]
    out['L1_min_ratio'] = min(L1)
    out['IBP_identity_max_rel'] = max(max([r['IBP_i'] for r in v['rows'].values() if np.log(1 / (r['t'] * r['J'])) < 30], default=0.0) for v in H.values())   # deep combs (J < e^-30): round-off
    out['tHvv_max'] = max(max(r['t_Hvv_max'] for r in v['rows'].values()) for v in H.values())
    dd = [(deb_log(v), k) for k, v in H.items() if min(v['rows']['%.3f' % m]['R'] * v['rows']['%.3f' % m]['t'] for m in (0.5, 1.0)) >= 0.05]
    dd = [(deb_log(v), k) for k, v in H.items() if k.split(':')[0] in ('lattice', 'strip')]
    DB = json.load(open('results/ac_debruijn.json')); dd += [(v['rel'], 'fixedgrid:' + k) for k, v in DB.items()]
    out['deBruijn_max_rel'] = max(dd); out['deBruijn_n'] = len(dd)                # lattices, strips: profile rows; mixtures, points: one fixed fine grid (ac_debruijn)
    out['LC_all_monotone'] = all(v['LC_monotone'] for v in H.values()); out['LC_min_ratio'] = min((v['LC_min_ratio'], k) for k, v in H.items())
    out['q2_dimensionless_max'] = max(max(r['q2'] * r['t'] ** 2 for r in v['rows'].values()) for v in H.values())
    L3 = []
    for k, v in H.items():
        for tau in ('0.25', '0.50'):
            e = v['L3_tau%s' % tau]; L3.append((e['bound_LC'] / e['Lambda'], k, tau, e['avg_condition']))
    out['L3_min_ratio'] = min(L3); out['L3_avg_condition_all'] = all(x[3] for x in L3)
    # all stage-Z states and lattices: Lambda vs ln(1/(sigma^2 J_ww))
    pts = []
    for f in ('x1', 'stored', 'strips'):
        for k, v in json.load(open('results/z_chain_%s.json' % f)).items(): pts.append((np.log(1 / (v['sigma'] ** 2 * v['J_ww'])), v['Lambda'], f))
    for k, v in H.items():
        c = v['rows']['1.000']; pts.append((np.log(1 / (c['t'] * c['J'])), c['Lambda'], k.split(':')[0] + (':comb' if 'comb' in k else '')))
    out['max_Lambda_minus_4lnJ'] = max(L - 4 * x for x, L, _ in pts)
    combs = sorted([(np.log(1 / v['rows']['1.000']['J']), v['rows']['1.000']['Lambda'], k) for k, v in H.items() if 'comb' in k and 'ang85' in k])
    out['comb85_slopes'] = [{'ln1J': x, 'Lambda': L, 'ratio': L / x, 'Lambda_minus_4ln': L - 4 * x, 'key': k} for x, L, k in combs]
    out['B_min'] = 4.0; out['A_for_B4'] = 4 * A0
    out['constants'] = {'conj_B4': {'c': 1 / (2 * np.pi)}, 'LC_tau0.5': {'B': 16.0, 'A': 2 + 16 * np.log(2), 'c': 0.5 / (np.pi * np.sqrt(2)) * np.sqrt(0.5) * 2 / 2},
                        'LC_tau': {tau: {'B': 8 / (1 - tau), 'A': 2 + 8 / (1 - tau) * np.log(1 / tau), 'c': np.sqrt(1 - tau) / (2 * np.sqrt(2) * np.pi),
                                         'C_sigma1': 0.5 * np.log(2 * (np.pi ** 2 + 1) / tau) + (1 - tau) / 8} for tau in (0.1, 0.25, 0.5)}}
    out['constants']['LC_tau0.5']['c'] = out['constants']['LC_tau'][0.5]['c']
    Kq = {}
    for tau in ('0.25', '0.50'):
        Kq[tau] = max((v['L3_tau%s' % tau]['R_at_1'] / v['L3_tau%s' % tau]['avg_R'], k) for k, v in H.items())
    out['quasiLC_heat_states'] = {tau: {'K_max': Kq[tau][0], 'at': Kq[tau][1]} for tau in Kq}
    out['LC_violations_heat_states'] = [{'key': k, 'min_ratio': v['LC_min_ratio'], 'tR_at_1': v['rows']['1.000']['R'] * v['rows']['1.000']['t']} for k, v in H.items() if not v['LC_monotone']]
    LQ = json.load(open('results/ac_lc_quasi.json')); out['quasiLC_lattice_search'] = {'n_runs': len(LQ['runs']), 'K_max': LQ['max_K_avg']}
    K0 = max(max(v for v in LQ['max_K_avg'].values()), max(Kq[t][0] for t in Kq)); out['K0_estimate'] = K0
    out['constants']['quasiLC'] = {tau: {'B': 8 * K0 / (1 - tau), 'c': 1 / (np.pi * np.sqrt(8 * K0 / (1 - tau))), 'A': 2 + 8 * K0 / (1 - tau) * np.log(1 / tau)} for tau in (0.25, 0.5)}
    LS = json.load(open('results/ac_lc_search.json')); out['LC_search'] = {'lattice_min': min(q['min_ratio'] for q in LS['lattice']), 'finite_min': min(q['min_ratio'] for q in LS['finite']),
                                                                         'n_lattice': len(LS['lattice']), 'n_finite': len(LS['finite'])}
    json.dump(out, open('results/summary_AC.json', 'w'), indent=1, default=float)
    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2)); a = ax[0]
    for lab, col, mk, sel in (('stage-Z states (418)', OI[5], 'o', lambda g: g in ('x1', 'stored', 'strips')), ('lattices, points, mixtures (heat-flow runs)', OI[3], 's', lambda g: not g.endswith('comb') and g not in ('x1', 'stored', 'strips')),
                              ('tilted combs (lattices of lines)', OI[6], '^', lambda g: g.endswith('comb'))):
        P = [(x, L) for x, L, g in pts if sel(g)]; a.plot([p[0] for p in P], [p[1] for p in P], mk, ms=4, mfc='none', color=col, label=lab)
    xx = np.linspace(0, 400, 800)
    a.plot(xx, 4 * xx + 4 * A0, '-', color=OI[0], lw=1.2, label='conjecture: $\\Lambda \\leq 4\\ln(1/\\sigma^2 J_{ww}) + 4A_0$ (B = 4, sharp)')
    a.plot(xx, 2 + 16 * (xx + np.log(2)), '--', color=OI[7], lw=1.2, label='proved if LC: $\\Lambda \\leq 2 + 16\\ln(2/\\sigma^2 J_{ww})$ ($\\tau$ = 1/2)')
    a.set_xscale('symlog', linthresh=1); a.set_yscale('log'); a.set_xlabel('ln(1/($\\sigma^2 J_{ww}$))'); a.set_ylabel('$\\Lambda$'); a.set_title('(Z2-F$_w$): $\\Lambda$ against ln(1/$\\sigma^2J_{ww}$)')
    a.legend(frameon=False, fontsize=6.3, loc='upper left')
    b = ax[1]; ms = [0.25, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    for k, col in (('lattice:lattice_sup', OI[6]), ('lattice:comb_D0.70_ang85', OI[1]), ('points:two_d5.04_th40', OI[5]), ('points:diamond5', OI[3]),
                   ('strip:strip_gmin_1e-24', OI[0]), ('mix:mix_t4_K32_s0.01.json', OI[7]), ('mix:twoclusters_V2_s0.001', OI[2])):
        if k not in H: continue
        R = H[k]['R_profile']; b.plot(ms, [r / R[7] for r in R], 'o-', ms=3, color=col, label=k.split(':')[1] + (' (LC fails; t R = %.1e)' % (R[7] * H[k]['s2']) if 'twoclusters' in k else ''))
    b.axhline(1, color=OI[0], lw=0.6, ls=':'); b.set_xlabel('t/$\\sigma^2$ (blur variance)'); b.set_ylabel('R(t)/R($\\sigma^2$),  R = $-$d ln $J_{ww}$/dt')
    b.set_title('LC (R decreasing): holds except tiny-R and lattice cases'); b.legend(frameon=False, fontsize=6.0)
    fig.tight_layout(); fig.savefig('figures/stageAC.png', dpi=160); fig.savefig('figures/stageAC.pdf'); plt.close(fig)
    print(json.dumps({k: out[k] for k in out if k not in ('comb85_slopes',)}, indent=1, default=float)); print(out['comb85_slopes'])
