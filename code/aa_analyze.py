"""Stage AA: summary (results/summary_AA.json) and figure (figures/stageAA.png/.pdf).  Python 3.11."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aa_tools import best_t

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
KAPPA_U = lambda s: 1 / (1 + np.sqrt(2) / np.pi * np.sqrt(12 * (np.pi ** 2 + s * s)))


def zprime_curve(sig, K1=1.0, t=None, EQ=None):
    """Theorem Z': c(t)/sqrt(ln(1/sigma) + C_t),  c(t) = sqrt(t)/(2 pi sqrt K1),  C_t = ln(8 (pi^2 + sigma^2/2)/(e t (1-t)))/2."""
    EQ = np.pi ** 2 + sig * sig / 2 if EQ is None else EQ; t = best_t(sig, EQ)[0] if t is None else t
    return np.sqrt(t) / (2 * np.pi * np.sqrt(K1)) / np.sqrt(np.log(1 / sig) + 0.5 * np.log(8 * EQ / (np.e * t * (1 - t))))


if __name__ == '__main__':
    A = json.load(open('results/aa_chain.json')); out = {'aa1_steps': {}}
    for s in ('A_jensen', 'B_cut', 'C_kernel', 'D_heis_L2', 'final', 'S_lambda', 'AA3_r', 'AA3_r_uniform', 'AA3_GY'):
        q = [(v[s][0] / max(v[s][1], 1e-300), k) for k, v in A.items() if s in v]
        out['aa1_steps'][s] = {'n': len(q), 'min_ratio': min(q)[0], 'at': min(q)[1], 'violations': sum(1 for r, _ in q if r < 1 - 1e-9)}
    out['E_Q_max'] = max(v['E_Q_yz2'] for v in A.values())
    g = [(v['gap_over_sigma'] / v['bound_state'], k) for k, v in A.items() if 'bound_state' in v]; out['gap_over_bound_state_min'] = min(g)
    g = [(v['gap_over_sigma'] / v['bound_uniform'], k) for k, v in A.items() if 'bound_uniform' in v]; out['gap_over_bound_uniform_min'] = min(g)
    q = [(v['sqrtPw'] / v['sqrtPw_bound_state'], k) for k, v in A.items()]; out['sqrtPw_over_bound_min'] = min(q)
    # constants of Theorem Z'
    out['constants'] = {'t_half': {'c_over_invsqrtK1': 1 / (2 * np.pi * np.sqrt(2)), 'C': 0.5 * np.log(32 * (np.pi ** 2 + 0.5) / np.e)},
                        'kappa_uniform_sigma_to_0': KAPPA_U(0.0), 'kappa_uniform_sigma_1': KAPPA_U(1.0),
                        'conjecture_F': {'c': 1 / (2 * np.pi)}}
    S = json.load(open('results/summary_Z.json'))['strips']
    rows = []
    for r in S:
        s = r['sigma']; th, f = best_t(s)
        rows.append({'sigma': s, 'gap_over_sigma': r['gap_over_sigma'], 'Z_curve_K2strip': r['theorem_curve'], 'Zprime_t_half_K1_1': zprime_curve(s, 1.0, 0.5),
                     'Zprime_tstar_K1_1': zprime_curve(s, 1.0), 't_star': th, 'Zprime_tstar_K1_1_uniform_kappa': KAPPA_U(s) * zprime_curve(s, 1.0)})
    out['strip_curves'] = rows
    # AA2: adversarial K_Lambda / F and the product family
    adv = []
    for fn in sorted(glob.glob('results/aa_klam_s*.json')):
        for k, v in json.load(open(fn)).items():
            if v.get('fine'): adv.append({'key': k, 'sigma': v['sigma'], **{q: v['fine'][q] for q in ('K_Lambda', 'F_log', 'P_w', 'y_over_sigma', 'Lambda', 'J_ww', 'grid_mass')}, 'kind': v['kind']})
    out['adversarial'] = adv
    def best(kind, cond=lambda r: True):
        c = [r for r in adv if r['kind'] == kind and r['y_over_sigma'] <= 0.5 + 1e-6 and abs(r['grid_mass'] - 1) < 1e-3 and cond(r)]
        key = 'F_log' if kind == 'F' else 'K_Lambda'; return max(c, key=lambda r: r[key]) if c else None
    out['adv_best'] = {'KL': best('KL'), 'KL_gap': best('KL_gap', lambda r: np.sqrt(r['P_w']) <= 1.0 + 1e-6), 'F': best('F')}
    Z = [v for f in ('x1', 'stored', 'strips') for v in json.load(open('results/z_chain_%s.json' % f)).values()]
    small = [v for v in Z if np.sqrt(v['P_w']) <= 1 and v['y_over_sigma'] <= 0.5]
    out['K_Lambda_max_small_gap_stored'] = max(v['K_Lambda'] for v in small); out['n_small_gap_stored'] = len(small)
    Fz = [np.log(v['sigma'] ** 2 * v['J_ww']) + v['Lambda'] / 4 for v in Z]; out['F_log_max_stored'] = max(Fz)
    P = json.load(open('results/aa_product.json')); out['product_family'] = P
    json.dump(out, open('results/summary_AA.json', 'w'), indent=1, default=float)
    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2)); a = ax[0]
    Zy = [v for v in Z if v['y_over_sigma'] <= 0.5]; a.plot([np.sqrt(v['P_w']) for v in Zy], [v['K_Lambda'] for v in Zy], 'o', ms=3, mfc='none', color=OI[5], alpha=0.6, label='stage-Z states with y/$\\sigma \\leq$ 1/2 (%d)' % len(Zy))
    ad = [r for r in adv if r['y_over_sigma'] <= 0.5 + 1e-6 and abs(r['grid_mass'] - 1) < 1e-3]
    a.plot([np.sqrt(r['P_w']) for r in ad], [r['K_Lambda'] for r in ad], 's', ms=5, mfc='none', mew=1.3, color=OI[6], label='adversarial searches (best of each run)')
    a.plot([r['sqrtPw_lower'] for r in P], [r['K_Lambda'] for r in P], '^-', ms=6, color=OI[3], label='product family (wide part + low-density bump)')
    a.axvline(1, color=OI[0], lw=0.8, ls=':'); a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('$\\sqrt{P_w}$ (≈ gap/$\\sigma$)'); a.set_ylabel('$K_\\Lambda$')
    a.set_title('$K_\\Lambda$ is large only where the gap is large'); a.legend(frameon=False, fontsize=6.5)
    b = ax[1]; s = np.array([r['sigma'] for r in rows]); Lx = np.log(1 / s); ll = np.geomspace(4, 260, 200); sg = np.exp(-ll)
    b.plot(Lx, [r['gap_over_sigma'] for r in rows], 'o', ms=6, color=OI[0], label='strips: $g_{min}(\\sigma)$')
    b.plot(ll, [zprime_curve(x, 1.0) for x in sg], '-', color=OI[6], lw=1.6, label="Theorem Z' ($K_1 = 1$, optimal t; no $K_J$, $K_\\rho$)")
    b.plot(ll, [zprime_curve(x, 1.0, 0.5) for x in sg], '--', color=OI[6], lw=1.0, label="Theorem Z', t = 1/2")
    b.plot(ll, [KAPPA_U(x) * zprime_curve(x, 1.0) for x in sg], ':', color=OI[2], lw=1.4, label="Theorem Z' with the $\\varepsilon_m$-free factor $\\kappa_u$")
    b.plot(Lx, [r['Z_curve_K2strip'] for r in rows], 's', ms=4, mfc='none', color=OI[3], label='stage Z curve (with $K_2$ of the strips)')
    b.axhline(0.884, color=OI[1], lw=1, ls='--', label='mixtures 0.884')
    b.set_xscale('log'); b.set_yscale('log'); b.set_ylim(1e-3, 60); b.set_xlabel('ln(1/$\\sigma$)'); b.set_ylabel('gap/$\\sigma$')
    b.set_title("Theorem Z' below the strips"); b.legend(frameon=False, fontsize=6.5, loc='upper right')
    fig.tight_layout(); fig.savefig('figures/stageAA.png', dpi=160); fig.savefig('figures/stageAA.pdf'); plt.close(fig)
    print(json.dumps({k: out[k] for k in ('aa1_steps', 'E_Q_max', 'gap_over_bound_state_min', 'gap_over_bound_uniform_min', 'constants', 'adv_best', 'K_Lambda_max_small_gap_stored', 'n_small_gap_stored', 'F_log_max_stored')}, indent=1, default=float))
    for r in rows: print(r)
