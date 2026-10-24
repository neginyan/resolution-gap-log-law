"""Stage AF: the eight strip optima (stage W/Z), at blur t = sigma^2: kappa_H, its stage-AE split (N_eta, X_eta), the eigen split
(shallow / w-visible / Y) and t^2 R' (central difference of R from ac_heat.sums_strip at t(1 +- 1e-3)).
Fields from the two-scale closed form of ac_heat.sums_strip (same quadrature), collected over all points and passed to af_tools.decompose.
Blurs t = m sigma^2, m = 1/4, 1/2, 1 (the window [tau sigma^2, sigma^2] of the integration lemma).  -> results/af_strip.json.  Python 3.11."""
import json
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from ac_heat import sums_strip
from z_strip import strip_params
from af_tools import decompose, eig2


def strip_fields(P, t, n_theta=64, n_slow=24, n_gl=12):
    kap = P['k0'] / P['sig']; c, ell, Lw, Lv = P['c'], P['ell'], P['Lw'], P['Lv']
    Tv, Tw = Lv * Lv + t, Lw * Lw + t; Kv, Kw = Lv * Lv / Tv, Lw * Lw / Tw; sv2, sw2 = Lv * Lv * t / Tv, Lw * Lw * t / Tw
    G = c * np.exp(-0.5 * kap * kap * sv2); e2 = ell * ell + sw2; sc = np.sqrt(e2) / Kw; Rw = 9.0 * np.sqrt(Tw)
    edges = np.unique(np.concatenate([np.linspace(-Rw, Rw, 65), np.arange(-14 * sc, 14 * sc * (1 + 1e-9), sc / 3)])); edges = edges[(edges >= -Rw) & (edges <= Rw)]
    xg, wg = leggauss(n_gl); yw = ((edges[:-1, None] + edges[1:, None]) / 2 + (edges[1:, None] - edges[:-1, None]) / 2 * xg[None]).ravel()
    wyw = ((edges[1:, None] - edges[:-1, None]) / 2 * wg[None]).ravel() * np.exp(-yw * yw / (2 * Tw)) / np.sqrt(2 * np.pi * Tw)
    A = ell / np.sqrt(e2) * np.exp(-Kw * Kw * yw * yw / (2 * e2)); dA = -A * Kw * Kw * yw / e2; ddA = A * (Kw ** 4 * yw * yw / e2 ** 2 - Kw * Kw / e2)
    ts, wts = hermegauss(n_slow); wts = wts / wts.sum(); yvs = ts * np.sqrt(Tv); th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta; q0 = kap * Kv
    out = {k: [] for k in ('W', 'Hvv', 'Hvw', 'Hww', 'sw')}
    for yv, wv in zip(yvs, wts):
        for tt in th:
            ct, st = np.cos(tt), np.sin(tt); F = G * A * ct; den = 1 + F
            Fv = -G * A * q0 * st; Fw = G * dA * ct; Fvv = -G * A * q0 * q0 * ct; Fww = G * ddA * ct; Fvw = -G * dA * q0 * st
            s_w = -yw / Tw + Fw / den; s_v = -yv / Tv + Fv / den
            out['Hvv'].append(1 / Tv - (Fvv * den - Fv * Fv) / den ** 2); out['Hww'].append(1 / Tw - (Fww * den - Fw * Fw) / den ** 2)
            out['Hvw'].append(-(Fvw * den - Fv * Fw) / den ** 2); out['W'].append(wv / n_theta * wyw * den); out['sw'].append(s_w)
    return {k: np.concatenate(v) for k, v in out.items()}


if __name__ == '__main__':
    Z = json.load(open('results/z_chain_strips.json')); out = {}
    for k, z in Z.items():
        P = strip_params(z['sigma'], z['k0'], z['lam'], z['M'], z['eta'], z['c'])
        for m in (0.25, 0.5, 1.0):
          t = m * z['sigma'] ** 2
          with np.errstate(over='ignore'):
            f = strip_fields(P, t); W = f['W'] / f['W'].sum(); J = np.sum(W * f['sw'] ** 2)
            D = decompose(f['W'], f['Hvv'], f['Hvw'], f['Hww'], t, J)
            Em, Ep = sums_strip(P, t * (1 - 1e-3)), sums_strip(P, t * (1 + 1e-3)); E0 = sums_strip(P, t)
          R = lambda E: (E['Hvw2'] + E['Hww2']) / E['J']
          D['t2R1'] = (t * R(Ep) - t * R(Em)) / 2e-3                                     # = t^2 R' (no t^2: underflow at sigma = 1e-96)
          D['J_check'] = abs(J / E0['J'] - 1); D['tR_check'] = abs(D['tR'] / (t * R(E0)) - 1)
          D['sigma'] = z['sigma']; D['tJ'] = t * J; D['m'] = m
          # where the deep valleys point: weighted mean of a over points with lambda_1 > 3
          m1, m2, a1, a2 = eig2(1 - t * f['Hvv'], -t * f['Hvw'], 1 - t * f['Hww']); deep = (m1 - 1) > 3
          D['deep_mass'] = float(np.sum(W * deep)); D['deep_a_mean'] = float(np.sum(W * deep * a1) / max(np.sum(W * deep), 1e-300)); D['deep_a_max'] = float(np.max(a1[deep])) if deep.any() else 0.0
          D['lambda_max'] = float(np.max(m1 - 1)); out['%s_m%.2f' % (k, m)] = D
          print('%-18s m %.2f lam_max %.2f tJ %.2e tR %.3f t2R1 %.4f kappa_H %.4f diag %.4f cross %.4f N %.4f X %.4f Y(19,0.1) %.4f vis %.4f deep a %.2e' % (k, m, D['lambda_max'], D['tJ'], D['tR'], D['t2R1'], D['kappa_H'], D['diag'], D['cross'], D['N_eta'], D['X_eta'],
              D['A19_a0.1']['Y'], D['A19_a0.1']['visible'], D['deep_a_mean']), flush=True)
    json.dump(out, open('results/af_strip.json', 'w'), indent=1, default=float)
