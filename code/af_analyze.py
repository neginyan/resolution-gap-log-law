"""Stage AF: summary (results/summary_AF.json) and figure (figures/stageAF.png/.pdf).  Python 3.11."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ae_const import qlcw

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
SPL = ('A19_a0.1', 'A9_a0.2', 'A3_a0.5', 'A1_a0.55')


def theorem_af(A, a0, kc3, y, tau):
    """2-d: t^2 R' <= A tR + (2/a0^2) kappa_{c^3}(c1) + y  ->  (qLC_w) and the log-law constants (qlcw with eta' = A)."""
    return qlcw(A, 2 * kc3 / a0 ** 2 + y, tau)


if __name__ == '__main__':
    C = json.load(open('results/af_const.json')); S = json.load(open('results/af_strip.json')); D = json.load(open('results/af_allstates.json')); SH = json.load(open('results/af_shapes.json'))
    out = {'kappa_two_sided': {k: v['two_sided'] for k, v in C['cube_minus1'].items()}, 'kappa_one_sided': {k: v['one_sided'] for k, v in C['cube_minus1'].items()},
           'kappa_c3': {k: v['two_sided'] for k, v in C['cube'].items()}, 'kappa_grid_check': max(abs(v['two_sided'] / v['two_sided_finer'] - 1) for v in list(C['cube_minus1'].values()) + list(C['cube'].values()))}
    out['strips'] = {k: {q: v[q] for q in ('sigma', 'm', 'tJ', 'tR', 't2R1', 'kappa_H', 'diag', 'cross', 'N_eta', 'X_eta', 'lambda_max', 'deep_mass')} | {'Y3': v['A3_a0.5']['Y'], 'Y1': v['A1_a0.55']['Y'], 'vis3': v['A3_a0.5']['visible']} for k, v in S.items()}
    out['strips_max'] = {q: max(v[q] for v in out['strips'].values()) for q in ('t2R1', 'kappa_H', 'X_eta', 'N_eta', 'lambda_max', 'Y3', 'Y1')}
    rows = [(k, r) for k, v in D['lattice'].items() for r in v] + [('mix:' + k, r) for k, v in D['mix'].items() for r in v] + [('adv:' + k, r) for k, r in D['adv'].items()]
    shr = [('shape:' + k, r) for k, v in SH.items() if k.startswith('scan_') for r in v] + [('shape:' + k, v) for k, v in SH.items() if k.startswith('adv_')]
    out['n_rows'] = len(rows); out['n_shape_rows'] = len(shr)
    for s in SPL:
        m = max(rows, key=lambda kr: kr[1][s]['Y']); out['Y_max_%s' % s] = [m[1][s]['Y'], m[0], m[1].get('t')]
        out['vis_max_%s' % s] = max(r[s]['visible'] for _, r in rows); out['vis_aggregate_ok_%s' % s] = all(r[s]['visible'] <= r[s]['visible_bound_Kww3'] * (1 + 1e-9) + 1e-12 for _, r in rows)
        out['shallow_ok_%s' % s] = all(r[s]['shallow_over_AtR'] <= 1 + 1e-9 for _, r in rows)
    out['Y3_max_shapes'] = max((r['Y3'], k) for k, r in shr); out['X_max_shapes'] = max((r['X_eta'], k) for k, r in shr); out['t2R1_max_shapes'] = max((r['t2R1'], k) for k, r in shr)
    out['X_max_states'] = max((r['X_eta'], k) for k, r in rows); out['ratio_X_max'] = max((r['ratio_X_over_N_etatR_1'], k) for k, r in rows + shr)
    out['eig_identity_max'] = max(r['eig_identity_max'] for _, r in rows)
    Y3 = max(out['Y_max_A3_a0.5'][0], out['Y3_max_shapes'][0], out['strips_max']['Y3'])
    out['Y3_max_all'] = Y3
    cons = {}
    for tau in (0.25, 0.5):
        cons['AF_A3_a0.5_tau%.2f_Ynum' % tau] = theorem_af(3.0, 0.5, out['kappa_c3']['2'], Y3, tau)
        cons['AF_A3_a0.5_tau%.2f_Y0' % tau] = theorem_af(3.0, 0.5, out['kappa_c3']['2'], 0.0, tau)
        cons['AF_A1_a0.55_tau%.2f_Ynum' % tau] = theorem_af(1.0, 0.55, out['kappa_c3']['1.1'], out['Y_max_A1_a0.55'][0], tau)
        cons['1d_two_sided_eta0.1_tau%.2f' % tau] = qlcw(0.1, out['kappa_two_sided']['0.1'], tau)
    out['constants'] = cons
    json.dump(out, open('results/summary_AF.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('strips', 'constants')}, indent=1, default=float))
    for k, v in cons.items(): print(k, {a: round(float(b), 4) for a, b in v.items()})
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2)); a = ax[0]
    for m, mk in ((0.25, 's'), (0.5, '^'), (1.0, 'o')):
        V = [v for v in out['strips'].values() if v['m'] == m]; L = [np.log(1 / v['sigma']) for v in V]
        a.plot(L, [v['t2R1'] for v in V], mk + '-', color=OI[6], ms=4, label="$t^2R'$, t = %g$\\sigma^2$" % m)
        a.plot(L, [v['kappa_H'] for v in V], mk + ':', color=OI[5], ms=4, label='$\\kappa_H$, t = %g$\\sigma^2$' % m)
    a.axhline(0, color=OI[0], lw=0.5); a.set_xscale('log'); a.set_yscale('symlog', linthresh=0.01); a.set_xlabel('ln(1/$\\sigma$) (strip optima)'); a.legend(frameon=False, fontsize=6)
    a.set_title('strips: no deep valleys ($\\lambda_{max}$ <= %.1f), $\\kappa_H$ <= %.3f' % (out['strips_max']['lambda_max'], out['strips_max']['kappa_H']))
    b = ax[1]
    for lab, R, col in (('lattices (periodic)', [r for k, r in rows if not k.startswith(('mix', 'adv'))], OI[3]), ('mixtures, points', [r for k, r in rows if k.startswith(('mix', 'adv'))], OI[5]), ('fans, arcs, bent, two combs', [r for _, r in shr], OI[6])):
        b.plot([r['X_eta'] for r in R], [(r['A3_a0.5']['Y'] if 'A3_a0.5' in r else r.get('Y3', 0)) for r in R], 'o', ms=3, mfc='none', color=col, label=lab)
    b.set_xlabel('$X_\\eta$ (deep cross term, stage AE)'); b.set_ylabel('$Y$ (near-v deep valleys, A = 3, $a_0$ = 1/2)'); b.legend(frameon=False, fontsize=7)
    b.set_title('what remains after the eigen split'); fig.tight_layout(); fig.savefig('figures/stageAF.png', dpi=160); fig.savefig('figures/stageAF.pdf'); plt.close(fig)
