"""Stage AD: summary (results/summary_AD.json), stage figure (figures/stageAD.png/.pdf), theorem staircase on the strips,
dense staircase curves for the paper-candidate figures (drawn by paperE/make_figures.py).  Python 3.11."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ad_gamma2 import identity_check
from aa_tools import best_t

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
A0 = 0.47097204431532214


# ---------------- theorem staircase: every rung written as  Lambda <= A + B * ln(1/(sigma^2 J_ww)),  curve = (sqrt2/pi)/sqrt(Lambda_bound)
def lam_cr(sig, A, B):
    """Cramer-Rao (H0''): ln(1/(sigma^2 J_ww)) <= 2 ln(1/sigma) + ln(2 V_z), V_z = pi^2 + sigma^2."""
    return A + B * (2 * np.log(1 / sig) + np.log(2 * (np.pi ** 2 + sig * sig)))


def curve_from_lambda(L): return (np.sqrt(2) / np.pi) / np.sqrt(L)


def c_C(A, B, sig=1.0):
    return 1 / (np.pi * np.sqrt(B)), 0.5 * np.log(2 * (np.pi ** 2 + sig * sig)) + A / (2 * B)


def rung_qlcw(K, tau, c0):
    B = 8 * K / (1 - tau); return 2 + 8 * c0 + B * np.log(1 / tau), B


def c0_of_kappa(kappa, tau): return kappa * (np.log(1 / tau) / (1 - tau) - 1)


def curve_Zpp(sig, t=None, K1=1.0):
    EQ = np.pi ** 2 + sig * sig / 2; t = best_t(sig, EQ)[0] if t is None else t
    return np.sqrt(t) / (2 * np.pi * np.sqrt(K1)) / np.sqrt(np.log(1 / sig) + 0.5 * np.log(8 * EQ / (np.e * t * (1 - t)))), t


def curve_curv(sig, kappa, tau=None):
    """K = 1, c0 = kappa (ln(1/tau)/(1-tau) - 1); tau fixed or optimised per sigma."""
    taus = [tau] if tau is not None else np.geomspace(1e-4, 0.9, 400)
    best = max((curve_from_lambda(lam_cr(sig, *rung_qlcw(1.0, ta, c0_of_kappa(kappa, ta)))), ta) for ta in taus); return best


if __name__ == '__main__':
    out = {}
    # ---- Gamma_2 identities, recomputed here (lattices, finite differences in t)
    ids = {}
    for nm, B in (('sup', [[3.25, 3.25], [3.212, -3.212]]), ('rect', [[3.0, 0.0], [0.0, 2.0]]), ('oblique', [[2.5, 0.7], [0.3, 2.2]])):
        for t in (0.6, 1.0, 1.7): ids['%s_t%.1f' % (nm, t)] = identity_check(B, t)
    out['gamma2_identities'] = {k: max(v[k] for v in ids.values()) for k in ('G1', 'G2', 'G3', 'G4')}
    # ---- two points (closed form)
    TP = json.load(open('results/ad_twopoint.json')); out['two_points'] = {k: TP[k] for k in TP if k != 'curve'}
    # ---- stored stage-AC rows
    ST = json.load(open('results/ad_qlc_stored.json'))
    keys = ['tau%s_c0%.1f' % (tau, c0) for tau in ('0.25', '0.5') for c0 in (0.0, 0.1, 0.5)]
    out['stored'] = {'n': len(ST), 'K_needed': {k: max((v['K'][k], s) for s, v in ST.items()) for k in keys},
                     't2R1_max': max((max(v['t2R1'].values()), s) for s, v in ST.items())}
    # LC-violation table: the violators of exact LC, K needed with c0
    viol = {s: v for s, v in ST.items() if v['LC_min_ratio'] < 1 - 1e-9}
    out['LC_violators_with_c0'] = {s: {'LC_min_ratio': v['LC_min_ratio'], 'tR': v['tR'], **{k: v['K'][k] for k in keys}} for s, v in viol.items()}
    # ---- lattice ladders
    LT = json.load(open('results/ad_qlc_lattice.json'))
    out['lattice'] = {'n': len(LT), 'K_needed': {k: max((v['K'][k], s) for s, v in LT.items() if v['K'][k] is not None) for k in keys},
                      't2R1_max': max((v['max_t2R1'], s) for s, v in LT.items()), 'kappa_H_max': max((v['max_kappa_H'], s) for s, v in LT.items())}
    out['LC_violators_with_c0'].update({'lattice:' + s: {'LC_min_ratio': None, 'max_t2R1': v['max_t2R1'], **{k: v['K'][k] for k in keys}} for s, v in LT.items() if s.startswith('ACadv')})
    # ---- mixtures / points
    MX = json.load(open('results/ad_qlc_mix.json'))
    out['mix'] = {'n': len(MX), 't2R1_max': max((max(v[m]['t2R1'] for m in ('0.5', '1.0')), s) for s, v in MX.items()),
                  'kappa_H_max': max((max(v[m]['kappa_H'] for m in ('0.5', '1.0')), s) for s, v in MX.items()),
                  'mass_dev_max': max(abs(v[m]['mass'] - 1) for v in MX.values() for m in ('0.5', '1.0'))}
    # ---- adversarial
    AD = json.load(open('results/ad_qlc_adv.json'))
    out['adv'] = {f'{kind}_{what}': max((q['value'] for q in AD[kind] if q['what'] == what), default=None) for kind in ('lattice', 'finite') for what in ('t2R1', 'kappa_H')}
    out['adv']['n_lattice'] = len(AD['lattice']); out['adv']['n_finite'] = len(AD['finite'])
    kap = max(out['stored']['t2R1_max'][0], out['lattice']['t2R1_max'][0], out['mix']['t2R1_max'][0], out['adv']['lattice_t2R1'] or -9, out['adv']['finite_t2R1'] or -9)
    out['kappa_num'] = kap
    kapH = max(out['lattice']['kappa_H_max'][0], out['mix']['kappa_H_max'][0], out['adv']['lattice_kappa_H'] or -9, out['adv']['finite_kappa_H'] or -9, TP['kappa_H_sup'])
    out['kappa_H_num'] = kapH
    # ---- constants of the staircase (sigma = 1 for C)
    cons = {}
    for tau in (0.25, 0.5):
        for nm, kv in (('LC_exact', 0.0), ('kappa_num', kap), ('kappa_H_num', kapH)):
            c0 = c0_of_kappa(kv, tau); A, B = rung_qlcw(1.0, tau, c0); c, C = c_C(A, B)
            cons['curv_%s_tau%.2f' % (nm, tau)] = {'kappa': kv, 'c0': c0, 'A': A, 'B': B, 'c': c, 'C_sigma1': C}
        for c0 in (0.1, 0.5):
            Kn = max(out['stored']['K_needed']['tau%s_c0%.1f' % (('0.25' if tau == 0.25 else '0.5'), c0)][0], out['lattice']['K_needed']['tau%s_c0%.1f' % (('0.25' if tau == 0.25 else '0.5'), c0)][0])
            A, B = rung_qlcw(Kn, tau, c0); c, C = c_C(A, B); cons['qlcw_c0%.1f_tau%.2f' % (c0, tau)] = {'K_needed': Kn, 'c0': c0, 'A': A, 'B': B, 'c': c, 'C_sigma1': C}
    c, C = c_C(4 * A0, 4.0); cons['conj_Z2Fprime'] = {'A': 4 * A0, 'B': 4.0, 'c': c, 'C_sigma1': C}
    cons['Zpp_t_half'] = {'c_times_sqrtK1': 1 / (2 * np.pi * np.sqrt(2)), 'C_sigma1': 0.5 * np.log(8 * (np.pi ** 2 + 0.5) / (np.e * 0.25))}
    cons['curv_tau_to_0'] = {'B': 8.0, 'c': 1 / (np.pi * np.sqrt(8))}
    out['constants'] = cons
    # ---- strips: staircase curves and ratios
    SZ = json.load(open('results/summary_Z.json'))['strips']; rows = []
    for r in SZ:
        s = r['sigma']; g = r['gap_over_sigma']
        z_half = curve_Zpp(s, 0.5)[0]; z_opt, tst = curve_Zpp(s)
        cu, tau_opt = curve_curv(s, kap); cu4 = curve_curv(s, kap, 0.25)[0]
        cj = curve_from_lambda(lam_cr(s, 4 * A0, 4.0))
        rows.append({'sigma': s, 'gap': g, 'K_Lambda_strip': r['K_Lambda'], 'Zpp_t_half_K1': z_half, 'Zpp_topt_K1': z_opt, 't_opt': tst,
                     'Zpp_topt_Kstrip': z_opt / np.sqrt(r['K_Lambda']), 'curv_tau_quarter': cu4, 'curv_tau_opt': cu, 'tau_opt': tau_opt, 'conj': cj,
                     'ratio_Zpp_topt_K1': g / z_opt, 'ratio_curv_tau_opt': g / cu, 'ratio_conj': g / cj})
    out['strip_staircase'] = rows
    xx = np.geomspace(4, 230, 120); ss = np.exp(-xx)
    out['staircase_curves'] = {'ln1_sigma': list(xx), 'conj': [curve_from_lambda(lam_cr(s, 4 * A0, 4.0)) for s in ss],
                               'curv_tau_opt': [curve_curv(s, kap)[0] for s in ss], 'Zpp_topt_K1': [curve_Zpp(s)[0] for s in ss]}
    json.dump(out, open('results/summary_AD.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('strip_staircase', 'constants')}, indent=1, default=float))
    for k, v in cons.items(): print(k, {a: round(float(b), 4) for a, b in v.items()})
    for r in rows: print('%.0e gap %.3f  Z''(K1=1,t*) %.4f  curv(tau*) %.4f (tau %.3f)  conj %.4f  ratios %.1f %.1f %.1f' % (r['sigma'], r['gap'], r['Zpp_topt_K1'], r['curv_tau_opt'], r['tau_opt'], r['conj'], r['ratio_Zpp_topt_K1'], r['ratio_curv_tau_opt'], r['ratio_conj']))

    # ---------------- stage figure
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2)); a = ax[0]; C_ = TP['curve']
    a.plot(C_['a'], C_['kappa_H'], '-', color=OI[5], label='$\\kappa_H = -t^2E[(H^3)_{ww}]/J_{ww}$ (upper bound)')
    a.plot(C_['a'], C_['t2Rprime'], '-', color=OI[6], label="$t^2R'$ (actual)")
    a.axhline(0, color=OI[0], lw=0.6, ls=':'); a.set_xscale('log'); a.set_xlabel('$a = d^2/(4t)$ (two point masses, separation along w)'); a.set_ylabel('dimensionless')
    a.set_title("two points: $t^2R' < 0$ (exact LC), $\\kappa_H \\leq$ %.3f" % TP['kappa_H_sup']); a.legend(frameon=False, fontsize=7)
    b = ax[1]; P = []
    for v in LT.values():
        for r in v['rows']:
            if r is not None: P.append((r['tR'], r['t2R1'], r['kappa_H'], 'l'))
    for v in MX.values():
        if isinstance(v, dict):
            for m in ('0.5', '1.0'): P.append((v[m]['tR'], v[m]['t2R1'], v[m]['kappa_H'], 'm'))
    for g, col, lab in (('l', OI[3], 'lattices, combs (blur ladder)'), ('m', OI[5], 'mixtures, points')):
        Q = [p for p in P if p[3] == g]
        b.plot([p[0] for p in Q], [p[1] for p in Q], 'o', ms=3, mfc='none', color=col, label="$t^2R'$: " + lab)
        b.plot([p[0] for p in Q], [p[2] for p in Q], '.', ms=2, color=col, alpha=0.4, label='$\\kappa_H$: ' + lab)
    b.axhline(kap, color=OI[6], lw=0.8, ls='--', label="sup $t^2R'$ found = %.3f" % kap)
    b.set_xscale('log'); b.set_xlabel('$tR$'); b.set_ylabel("$t^2R'$ and $\\kappa_H$"); b.set_ylim(-1.2, 3.0); b.set_title("curvature: $t^2R' \\leq \\kappa_H$ (proved); both bounded")
    b.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig('figures/stageAD.png', dpi=160); fig.savefig('figures/stageAD.pdf'); plt.close(fig)

