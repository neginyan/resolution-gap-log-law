"""Stage W2: line-resolved sums for the direct gap bound (the tug-of-war between the window and the background).
line_sums: for every W-node w (a 'line' = fixed w), the v-integrals of p and of p*field for the point fields of v3_tools (exact, with the
stage-W cancellation-free fixes).  From them:
  exact gap:        gap/sigma = (2 sigma^2 J_ww + N)/(sigma^2 (J_vv + J_ww)) / sigma,   N = -E[d_p D]
  second order N:   N2 = E[ sigma^2 (H_vv - H_ww) (m_z^2 + Var_z) / 4 ]   (from N = -sigma^-2 E[Cov(x_p, x_z - sin x_z | y)] and x - sin x = x^3/6 + ...)
  line focus:       omega(w) = sigma^2 E[s_v^2 | w];  window lines = {omega >= theta * a},  a = sigma^2 J_vv
  split:            gap = G_win + G_bg (numerator split over lines, common denominator)
  tug-of-war terms: sqrt(P#_win) (P# restricted to the window lines),  T_bg = sqrt(mu_bg a_bg)/a  (bg: Cramer-Rao J_ww,bg >= mu_bg^2/E_bg[w^2]
                    and N_bg >= a_bg E_bg[w^2]/(8 mu_bg)  =>  (2 s^2 J_ww,bg + N_bg)/(s a) >= sqrt(mu_bg a_bg)/a in units of sigma);
                    this mean-focus form FAILS on strips (the background focus sits next to the window, where w^2 is small), so the
                    floor form T_bg_floor = mu_bg sqrt(omega_floor)/a (omega_floor = 5% mass quantile of the background line focus) is also reported.
Python 3.11."""
import numpy as np
from v3_tools import _pre, point_fields
from u_cgm import trap_weights

KEYS = ('svv', 'sww', 'Nibp', 'u2svv', 'aHvv', 'aHww', 'mz2', 'Varz', 'Napx', 'Napx_vv', 'Napx_ww', 'j', 'Z', 'N1', 'phi_svv', 'phi_sww', 'ibp_rest')


def line_sums(terms, sig, vn, wv, wn, rows=16, pre=None):
    pre = _pre(terms, sig) if pre is None else pre; out = {k: np.zeros(len(wn)) for k in KEYS + ('p',)}
    for a in range(0, len(vn), rows):
        V, W = np.meshgrid(vn[a:a + rows], wn, indexing='ij'); x = np.stack([V.ravel(), W.ravel()], 1)
        p, q = point_fields(pre, sig, x); pw = (p * np.repeat(wv[a:a + rows], len(wn))).reshape(V.shape)
        out['p'] += pw.sum(0)
        for k in KEYS: out[k] += (pw * q[k].reshape(V.shape)).sum(0)
    return out


def line_sums_envelope(terms, sig, kb, Lv, wn, nv=32, n0=15, ext=6.5):
    """true V-envelope: one-period averages at v0 nodes, trapezoid over v0 (as w_full)."""
    pre = _pre(terms, sig); per = 2 * np.pi * sig / kb; v0s = np.linspace(-ext * Lv, ext * Lv, n0); c0 = trap_weights(v0s)
    tot = None
    for v0, c in zip(v0s, c0):
        vn = v0 + (np.arange(nv) + 0.5) * per / nv - per / 2; o = line_sums(None, sig, vn, np.full(nv, c / nv), wn, pre=pre)
        tot = o if tot is None else {k: tot[k] + o[k] for k in tot}
    return tot


def analyse(L, wn, sig, theta=1.0, wgt=None):
    ww = trap_weights(wn) if wgt is None else wgt; s2 = sig * sig
    tot = {k: float(np.sum(L[k] * ww)) for k in L}; m = tot['p']; E = {k: v / m for k, v in tot.items()}
    a = s2 * E['svv']; den = s2 * (E['svv'] + E['sww'])
    gap = (2 * s2 * E['sww'] + E['Nibp']) / den / sig; gap2 = (2 * s2 * E['sww'] + E['Napx']) / den / sig
    om = s2 * L['svv'] / np.maximum(L['p'], 1e-300); win = om >= theta * a
    def part(mask):
        q = {k: float(np.sum((L[k] * ww)[mask])) / m for k in L}; return q
    W, B = part(win), part(~win)
    Gw = (2 * s2 * W['sww'] + W['Nibp']) / den / sig; Gb = (2 * s2 * B['sww'] + B['Nibp']) / den / sig
    # window-restricted P# (normalised inside the window lines) and the background tug-of-war term
    if W['p'] > 0:
        aw = s2 * W['svv'] / W['p']; Psw = s2 * (W['u2svv'] / W['p']) * (W['sww'] / W['p']) / aw ** 2
    else: aw, Psw = 0.0, float('nan')
    mub = B['p']; ab = s2 * B['svv']; wb2 = (B['mz2'] * 2) / max(mub, 1e-300)          # E_bg[w^2] ~ 2 E_bg[m_z^2] (z = (v+w)/sqrt2)
    Tbg = np.sqrt(max(mub * ab, 0)) / a
    # floor version: N_bg >= omega_floor E_bg[w^2] / 8 with omega_floor = the 5% (mass) quantile of the line focus on background lines
    bm = ~win; wts = (L['p'] * ww)[bm]; oms = om[bm]
    if wts.sum() > 0:
        o_ = np.argsort(oms); cw = np.cumsum(wts[o_]) / wts.sum(); omf = float(max(oms[o_][np.searchsorted(cw, 0.05)], 0.0))
    else: omf = 0.0
    Tbg_floor = mub * np.sqrt(omf) / a
    # AM-GM decomposition (exact up to the posterior third-order remainder R3 = N - N1):
    #   sigma a gap ~ 2 s2 J_ww + N  = [2 s2 J_ww + (s2/2) E[s_v^2 phi]] + rest,   rest = -(s2/2) E[s_w^2 phi] + (s2/2) E[ibp_rest] + R3
    #   [..] >= 2 s2 sqrt(J_ww E[s_v^2 phi]) = sigma a sqrt(P_phi),  P_phi := 4 s2 J_ww E[s_v^2 phi] / a^2   (= P# when phi ~ u^2/4)
    Pphi = 4 * s2 * E['sww'] * E['phi_svv'] / a ** 2; ibp_lhs = E['N1']; ibp_rhs = s2 / 2 * (E['phi_svv'] - E['phi_sww']) + s2 / 2 * E['ibp_rest']
    rest = -s2 / 2 * E['phi_sww'] + s2 / 2 * E['ibp_rest'] + (E['Nibp'] - E['N1'])
    slack = (2 * s2 * E['sww'] + s2 / 2 * E['phi_svv']) - sig * a * np.sqrt(Pphi)
    amgm = {'P_phi': Pphi, 'sqrtP_phi': float(np.sqrt(Pphi)), 'ibp_identity_rel': abs(ibp_rhs / ibp_lhs - 1), 'rest_over_sa': rest / (sig * a),
            'R3_over_sa': (E['Nibp'] - E['N1']) / (sig * a), 'amgm_slack_over_sa': slack / (sig * a), 'den_factor': a / den,
            'recon_gap': (sig * a * np.sqrt(Pphi) + slack + rest) / den / sig}
    return {**amgm, 'gap_over_sigma': gap, 'gap_N2_over_sigma': gap2, 'N': E['Nibp'], 'N2': E['Napx'], 'N2_vv': E['Napx_vv'], 'N2_ww': E['Napx_ww'],
            'N2_over_N': E['Napx'] / E['Nibp'], 'a': a, 'J_ww': E['sww'], 'theta': theta,
            'mass_win': W['p'], 'mass_bg': mub, 'a_win_share': s2 * W['svv'] / a, 'G_win': Gw, 'G_bg': Gb, 'sqrtPsharp_win': float(np.sqrt(Psw)) if Psw == Psw else float('nan'),
            'a_win_per_mass': aw, 'a_bg': ab, 'T_bg': Tbg, 'bound_win_plus_bg': (float(np.sqrt(Psw)) if Psw == Psw else 0.0) + Tbg, 'omega_floor_bg': omf, 'T_bg_floor': Tbg_floor,
            'bound_floor': (float(np.sqrt(Psw)) if Psw == Psw else 0.0) + Tbg_floor, 'E_bg_w2': wb2,
            'Gbg_Jww': 2 * s2 * B['sww'] / den / sig, 'Gbg_N': B['Nibp'] / den / sig}
