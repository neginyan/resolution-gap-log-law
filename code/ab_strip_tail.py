"""Stage AB2: the tail correction eps_pi on the ripple strips (g_min optima, sigma = 1e-2 ... 1e-96) -- a rigorous upper bound.
  eps_pi^2 = E[s_v^2 m_z^2 1{|m_z| > pi}] / E[s_v^2 m_z^2]  <=  sqrt(E[s_v^4 m_z^4] P(|m_z| > pi)) / E[s_v^2 m_z^2]        (Cauchy-Schwarz)
  m = (Kv y_v, Kw y_w) + sigma^2 (window part of the score);  |window part| <= c_s := sigma^2 (q G + G max|A'|)/(1 - G)  (pointwise),
  p <= (1 + G) N(y_v; Tv) N(y_w; Tw)  =>  P(|m_z| > pi) <= (1 + G) erfc( (pi - c_s) sqrt2 / sqrt(2 (Kv^2 Tv + Kw^2 Tw)) ).
E[s_v^2 m_z^2], E[s_v^4 m_z^4] by the two-scale quadrature of z_strip (same closed-form fields).  Also Var(X_z) and E_Q[y_z^2] (hypotheses H0'', H0').
-> results/ab_strip_tail.json.  Python 3.11."""
import json
import numpy as np
from math import erfc
from scipy.special import log_ndtr
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from z_strip import strip_params

SQ2 = np.sqrt(2.0)


def moments(P, n_theta=64, n_slow=24, n_gl=12):
    sig, k0, c, ell, Lw, Lv = P['sig'], P['k0'], P['c'], P['ell'], P['Lw'], P['Lv']; s2 = sig * sig; kap = k0 / sig
    Tv, Tw = Lv * Lv + s2, Lw * Lw + s2; Kv, Kw = Lv * Lv / Tv, Lw * Lw / Tw; sv2, sw2 = Lv * Lv * s2 / Tv, Lw * Lw * s2 / Tw
    G = c * np.exp(-0.5 * kap * kap * sv2); e2 = ell * ell + sw2; sc = np.sqrt(e2) / Kw; Rw = 9.0 * np.sqrt(Tw)
    edges = np.unique(np.concatenate([np.linspace(-Rw, Rw, 65), np.arange(-14 * sc, 14 * sc * (1 + 1e-9), sc / 3)])); edges = edges[(edges >= -Rw) & (edges <= Rw)]
    xg, wg = leggauss(n_gl); yw = ((edges[:-1, None] + edges[1:, None]) / 2 + (edges[1:, None] - edges[:-1, None]) / 2 * xg[None]).ravel()
    wyw = ((edges[1:, None] - edges[:-1, None]) / 2 * wg[None]).ravel() * np.exp(-yw * yw / (2 * Tw)) / np.sqrt(2 * np.pi * Tw)
    A = ell / np.sqrt(e2) * np.exp(-Kw * Kw * yw * yw / (2 * e2)); dA = -A * Kw * Kw * yw / e2
    ts, wts = hermegauss(n_slow); wts = wts / wts.sum(); yvs = ts * np.sqrt(Tv); th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta; q = kap * Kv
    S = {'m': 0.0, 'a2': 0.0, 'a4': 0.0}
    for yv, wv in zip(yvs, wts):
        for t in th:
            F = G * A * np.cos(t); den = 1 + F; s_v = -yv / Tv + (-G * A * q * np.sin(t)) / den; s_w = -yw / Tw + G * dA * np.cos(t) / den
            mz = (yv + yw + s2 * (s_v + s_w)) / SQ2; W = wv / n_theta * wyw * den
            S['m'] += W.sum(); tv = sig * s_v; S['a2'] += np.sum(W * tv * tv * mz * mz); S['a4'] += np.sum(W * tv ** 4 * mz ** 4)   # sigma s_v: the ratio a4/a2^2 is scale-free
    a2, a4 = S['a2'] / S['m'], S['a4'] / S['m']
    dAmax = float(np.max(np.abs(dA))) if len(dA) else 0.0
    cs = s2 * (q * G + G * dAmax) / (1 - G); var_m = Kv * Kv * Tv + Kw * Kw * Tw
    x = (np.pi - cs) * SQ2 / np.sqrt(var_m) if np.pi > cs else 0.0          # P(|N(0, var_m/2)| > pi - cs) = 2 Phi(-x)
    logP = float(np.log1p(G) + np.log(2.0) + log_ndtr(-x))
    log_eps2 = 0.5 * (np.log(a4 / a2 ** 2) + logP)                           # eps_pi^2 <= sqrt(a4 P)/a2
    return {'ratio_a4_over_a2sq': a4 / a2 ** 2, 'c_s': cs, 'sd_mz_background': float(np.sqrt(var_m / 2)), 'log10_P_tail_bound': logP / np.log(10),
            'log10_eps_pi_bound': float(0.5 * log_eps2 / np.log(10)), 'eps_pi_bound': float(np.exp(0.5 * log_eps2)),
            'Var_X_z': float((Lv * Lv + Lw * Lw) / 2)}


if __name__ == '__main__':
    Z = json.load(open('results/z_chain_strips.json')); A = json.load(open('results/aa_chain.json')); out = {}
    for k, z in Z.items():
        P = strip_params(z['sigma'], z['k0'], z['lam'], z['M'], z['eta'], z['c']); r = moments(P)
        r['E_Q_yz2'] = [v['E_Q_yz2'] for kk, v in A.items() if kk.startswith('sp:%s:' % k)][0]; r['sigma'] = z['sigma']
        r['H0p_ok'] = r['E_Q_yz2'] <= np.pi ** 2 + z['sigma'] ** 2 / 2; r['H0pp_ok'] = r['Var_X_z'] <= np.pi ** 2; out[k] = r
        print('%-20s log10 eps_pi <= %.1f  log10 P_tail <= %.1f  sd(m_z) %.3g  Var(X_z) %.3g  E_Q %.3g' % (k, r['log10_eps_pi_bound'], r['log10_P_tail_bound'], r['sd_mz_background'], r['Var_X_z'], r['E_Q_yz2']))
    json.dump(out, open('results/ab_strip_tail.json', 'w'), indent=1, default=float)
