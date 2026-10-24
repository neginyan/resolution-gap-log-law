"""Stage AG: largest posterior excess lambda_max = max eigenvalue of (-tH) = K - I on the strip optima, over the whole window
t in [sigma^2/4, sigma^2] (25 blurs), on a dense (theta, y_w) grid (the curvature does not depend on the slow coordinate y_v),
restricted to points with density p > thr * max p (thr = 1e-8 as in the main line, also 1e-12 and no restriction).
Two-scale closed form as ac_heat.sums_strip.  -> results/ag_strip_lambda.json.  Python 3.11."""
import json
import numpy as np
from z_strip import strip_params


def lam_field(P, t, n_theta=720, n_w=2501):          # grid <= 8e6 points
    kap = P['k0'] / P['sig']; c, ell, Lw, Lv = P['c'], P['ell'], P['Lw'], P['Lv']
    Tv, Tw = Lv * Lv + t, Lw * Lw + t; Kv, Kw = Lv * Lv / Tv, Lw * Lw / Tw; sv2, sw2 = Lv * Lv * t / Tv, Lw * Lw * t / Tw
    G = c * np.exp(-0.5 * kap * kap * sv2); e2 = ell * ell + sw2; q0 = kap * Kv
    yw = np.linspace(-9 * np.sqrt(Tw), 9 * np.sqrt(Tw), n_w); sc = np.sqrt(e2) / Kw
    yw = np.unique(np.concatenate([yw, np.linspace(-14 * sc, 14 * sc, n_w)]))
    th = 2 * np.pi * np.arange(n_theta) / n_theta; TH, YW = np.meshgrid(th, yw, indexing='ij')
    A = ell / np.sqrt(e2) * np.exp(-Kw * Kw * YW ** 2 / (2 * e2)); dA = -A * Kw * Kw * YW / e2; ddA = A * (Kw ** 4 * YW ** 2 / e2 ** 2 - Kw * Kw / e2)
    ct, st = np.cos(TH), np.sin(TH); F = G * A * ct; den = 1 + F
    Fv = -G * A * q0 * st; Fw = G * dA * ct; Fvv = -G * A * q0 * q0 * ct; Fww = G * ddA * ct; Fvw = -G * dA * q0 * st
    Hvv = 1 / Tv - (Fvv * den - Fv * Fv) / den ** 2; Hww = 1 / Tw - (Fww * den - Fw * Fw) / den ** 2; Hvw = -(Fvw * den - Fv * Fw) / den ** 2
    a, b, d = -t * Hvv, -t * Hvw, -t * Hww; lam = (a + d) / 2 + np.sqrt((a - d) ** 2 / 4 + b * b)
    dens = np.exp(-YW ** 2 / (2 * Tw)) * den                      # density at y_v = 0 (the slow factor only lowers it)
    return lam, dens / dens.max()


if __name__ == '__main__':
    Z = json.load(open('results/z_chain_strips.json')); out = {}
    for k, z in Z.items():
        P = strip_params(z['sigma'], z['k0'], z['lam'], z['M'], z['eta'], z['c']); rows = []
        for m in np.linspace(0.25, 1.0, 25):
            with np.errstate(over='ignore', invalid='ignore'):
                lam, dn = lam_field(P, m * z['sigma'] ** 2)
            rows.append({'m': float(m), 'lam_max_1e-8': float(lam[dn > 1e-8].max()), 'lam_max_1e-12': float(lam[dn > 1e-12].max()), 'lam_max_all': float(np.nanmax(lam))})
        out[k] = {'rows': rows, 'max_1e-8': max(r['lam_max_1e-8'] for r in rows), 'max_1e-12': max(r['lam_max_1e-12'] for r in rows), 'max_all': max(r['lam_max_all'] for r in rows),
                  'at_quarter': rows[0]}
        print('%-18s max over [s^2/4, s^2]: %.4f (p > 1e-8 max), %.4f (1e-12), %.4f (all);  at s^2/4: %.4f, s^2/2: %.4f, s^2: %.4f' % (k, out[k]['max_1e-8'], out[k]['max_1e-12'], out[k]['max_all'],
              rows[0]['lam_max_1e-8'], rows[8]['lam_max_1e-8'], rows[-1]['lam_max_1e-8']), flush=True)
    json.dump(out, open('results/ag_strip_lambda.json', 'w'), indent=1)
