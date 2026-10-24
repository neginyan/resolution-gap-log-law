"""Stage AE: summary (results/summary_AE.json) and figure (figures/stageAE.png/.pdf).  Python 3.11."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ae_const import qlcw

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']


def qlcw2(eta, kap, xi, tau):
    """2-d: t^2 R' <= 3 eta tR + kappa_eta + xi  (xi: bound of the deep cross term X_eta)."""
    return qlcw(3 * eta, kap + xi, tau)


if __name__ == '__main__':
    C = json.load(open('results/ae_const.json')); O = json.load(open('results/ae_oned.json')); D = json.load(open('results/ae_decomp.json'))
    F = json.load(open('results/ae_finite.json')); A = json.load(open('results/ae_adv.json'))
    out = {'kappa_eta': {k: v['kappa'] for k, v in C['kappa_eta'].items()}, 'kappa_eta_grid_check': {k: abs(v['kappa'] - v['kappa_finer_grid']) / v['kappa'] for k, v in C['kappa_eta'].items()}}
    kap = out['kappa_eta']['0.1']
    # 1-d
    out['oned'] = {'lemma1_min_margin': O['lemmas']['lemma1_min_margin'], 'star_min_ratio': O['lemmas']['star_min_ratio'], 'n_lemma_states': O['lemmas']['n'],
                   'max_N_eta_over_tJ': max(O['N_eta_over_tJ_K%d' % K]['value'] for K in (2, 4, 8, 16, 32)),
                   'max_kappa_H': max(O['kappa_H_K%d' % K]['value'] for K in (2, 4, 8, 16, 32)),
                   'max_t2Rp': max(O['t2Rp_K%d' % K]['value'] for K in (2, 4, 8, 16, 32)),
                   'lattice_1d': {'max_t2Rp': max(r['t2Rp'] for r in O['lattice_1d']), 'max_kappa_H': max(r['kappa_H'] for r in O['lattice_1d']), 'max_N_eta_over_tJ': max(r['N_eta_over_tJ'] for r in O['lattice_1d'])},
                   'min_theorem_slack': min(O[k]['theorem_1d_slack'] for k in O if k.endswith(tuple('K%d' % K for K in (2, 4, 8, 16, 32))))}
    out['oned']['by_K'] = {k: {q: O[k][q] for q in ('value', 'tR', 'kappa_H', 'N_eta_over_tJ', 't2Rp')} for k in O if '_K' in k}
    # 2-d decomposition
    rows = [(k, r) for k, v in D['lattice'].items() for r in v] + [(k, r) for k, v in D['mix'].items() for r in v] + [(k, r) for k, r in D['adv'].items()]
    for q in ('kappa_H', 'N_eta', 'X_eta', 'cross', 'diag'):
        m = max(rows, key=lambda kr: kr[1][q]); out['decomp_max_%s' % q] = [m[1][q], m[0], m[1]['t']]
    out['decomp_n_rows'] = len(rows)
    out['decomp_table'] = {k: {q: max(r[q] for r in v) for q in ('kappa_H', 'N_eta', 'X_eta', 'cross', 'diag')} for k, v in list(D['lattice'].items())}
    out['decomp_table'].update({'mix:' + k: {q: max(r[q] for r in v) for q in ('kappa_H', 'N_eta', 'X_eta', 'cross', 'diag')} for k, v in D['mix'].items()})
    out['mix_mass_dev_max'] = max(abs(r['mass'] - 1) for v in D['mix'].values() for r in v)
    # check of the proved 2-d inequality: kappa_H <= 3 eta tR + N_eta + X_eta  (pointwise splitting, exact algebra)
    out['split_identity_min_slack'] = min(0.3 * r['tR'] + r['N_eta'] + r['X_eta'] - r['kappa_H'] for _, r in rows)
    # finite lattices
    out['finite'] = {k: {'max_t2R1': v['max_t2R1'], 'max_kappa_H': v['max_kappa_H'], 'max_X_eta': v['max_X_eta'], 'max_N_eta': v['max_N_eta']} for k, v in F.items()}
    out['finite_max_t2R1'] = max((v['max_t2R1'], k) for k, v in F.items())
    for k, v in F.items():                                                        # the blur window where the periodic lattices have t^2 R' > 0
        out['finite'][k]['max_t2R1_window'] = max(r['t2R1'] for r in v['rows'] if r and 0.6 <= r['t'] <= 7.0)
        out['finite'][k]['t_at_max'] = max((r['t2R1'], r['t']) for r in v['rows'] if r)[1]
    out['adv_many'] = {k: {q: v[q] for q in ('value', 'fine_t2R1', 'fine_kappa_H', 'fine_X_eta', 'fine_N_eta', 'tR', 'nfev')} for k, v in A.items()}
    out['adv_many_max_t2R1'] = max((v['fine_t2R1'], k) for k, v in A.items() if k.startswith('t2R1'))
    xs = [v['fine_X_eta'] for v in A.values()]
    out['X_eta_max_all'] = max(out['decomp_max_X_eta'][0], max(xs), max(v['max_X_eta'] for v in F.values()))
    out['N_eta_max_all'] = max(out['decomp_max_N_eta'][0], max(v['fine_N_eta'] for v in A.values()), max(v['max_N_eta'] for v in F.values()))
    # constants
    cons = {}
    for eta in ('0.05', '0.1', '0.2'):
        for tau in (0.25, 0.5):
            cons['1d_eta%s_tau%.2f' % (eta, tau)] = qlcw(float(eta), out['kappa_eta'][eta], tau)
            cons['2d_eta%s_tau%.2f_xi%.2f' % (eta, tau, out['X_eta_max_all'])] = qlcw2(float(eta), out['kappa_eta'][eta], out['X_eta_max_all'], tau)
    out['constants'] = cons
    json.dump(out, open('results/summary_AE.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('decomp_table', 'constants', 'finite', 'adv_many', 'oned')}, indent=1, default=float))
    print(json.dumps(out['oned'], default=float)[:1500])
    for k, v in cons.items(): print(k, {a: round(float(b), 4) for a, b in v.items()})
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2)); a = ax[0]
    for nm, col in (('adv_best', OI[6]), ('ACadv5', OI[5]), ('diamond', OI[3])):
        ns = sorted(int(k.split('_n')[1]) for k in F if k.startswith(nm + '_n'))
        if ns: a.plot(ns, [out['finite']['%s_n%d' % (nm, n)]['max_t2R1_window'] for n in ns], 'o-', color=col, label=nm + ' (finite n x n)')
    per = {'adv_best': 0.3495, 'ACadv5': 0.3226}
    for nm, col in (('adv_best', OI[6]), ('ACadv5', OI[5])): a.axhline(per[nm], color=col, ls=':', lw=0.9, label=nm + ' (periodic)')
    a.axhline(0, color=OI[0], lw=0.5); a.set_xscale('log'); a.set_xticks([3, 5, 10, 20, 40, 80]); a.set_xticklabels(['3', '5', '10', '20', '40', '80']); a.minorticks_off(); a.set_xlabel('n (n x n points)'); a.set_ylabel("max of $t^2R'$ over $0.6 \\leq t \\leq 7$"); a.legend(frameon=False, fontsize=6.5)
    a.set_title("finite pieces of lattices: $t^2R' < 0$ (LC) up to n = 41")
    b = ax[1]; cats = [('lattices (periodic)', [r for v in D['lattice'].values() for r in v], OI[3]), ('mixtures, points', [r for v in D['mix'].values() for r in v], OI[5])]
    for lab, R, col in cats:
        b.plot([r['N_eta'] for r in R], [r['X_eta'] for r in R], 'o', ms=3, mfc='none', color=col, label=lab)
    b.set_xlabel('$N_\\eta$ (deep diagonal, proved $\\leq \\kappa_\\eta$ = %.0f)' % kap); b.set_ylabel('$X_\\eta$ (deep cross term, open)')
    b.set_title('(CV) parts, $\\eta$ = 0.1 (units $t^2/J_{ww}$)'); b.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig('figures/stageAE.png', dpi=160); fig.savefig('figures/stageAE.pdf'); plt.close(fig)
