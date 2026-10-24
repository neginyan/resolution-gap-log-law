"""Stage U1: the constant of lemma (p~) on an exactly periodic 2-D lattice (checkerboard grating), evaluated on one period cell.
rho(v, w) = 1 + c cos(k v) cos(q w)  (c <= 1, rho >= 0)  =>  p = rho * N(0, sigma^2 I) = 1 + A cos(k v) cos(q w),  A = c exp(-(k0^2 + q0^2)/2),
k0 = k sigma, q0 = q sigma.  All fields are closed form; integrals over one period cell are exact on a uniform grid (trapezoid = spectral).
Small-A prediction: a I(p~) / J_ww = Lambda = 4 k0^2 (independent of A and q0), y/sigma -> 0.
Output results/u1_lattice.json.  Python 3.11 compatible."""
import json, sys
import numpy as np
from t_tools import t_terms


def lattice_fields(k0, q0, c, sig=1.0, n=256):
    k, q = k0 / sig, q0 / sig; A = c * np.exp(-(k0 ** 2 + q0 ** 2) / 2)
    v = np.arange(n) * (2 * np.pi / k) / n; w = np.arange(n) * (2 * np.pi / q) / n
    Vv, Ww = np.meshgrid(v, w, indexing='ij'); cv, sv_, cw, sw_ = np.cos(k * Vv), np.sin(k * Vv), np.cos(q * Ww), np.sin(q * Ww)
    p = 1 + A * cv * cw; pv = -A * k * sv_ * cw; pw_ = -A * q * cv * sw_
    pvv = -A * k * k * cv * cw; pww = -A * q * q * cv * cw; pvw = A * k * q * sv_ * sw_
    s_v, s_w = pv / p, pw_ / p
    Hvv = -(pvv / p - s_v * s_v); Hww = -(pww / p - s_w * s_w); Hvw = -(pvw / p - s_v * s_w)
    P = p / p.sum()
    return {'n': (n, n), 'h': np.array([v[1] - v[0], w[1] - w[0]]), 'sig': sig, 'mass': 1.0, 'pw': P, 'sv': s_v, 'sw': s_w,
            'Hvv': Hvv, 'Hvw': Hvw, 'Hww': Hww, 'u': Vv + Ww, 'wcol': w}, A


def one(k0, q0, c, n=256):
    f, A = lattice_fields(k0, q0, c, n=n); t = t_terms(f)
    return {'k0': k0, 'q0': q0, 'c': c, 'A': A, 'aIpt_over_Jww': t['aIpt_over_Jww'], 'Lambda': t['Lambda'], 'a': t['a'],
            'y_over_sigma': t['y_over_sigma'], 'pred_4k0sq': 4 * k0 ** 2, 'I_W': t['I_W'], 'Jww': t['Jww']}


if __name__ == '__main__':
    rows = []
    for k0 in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        for q0 in (0.3, 1.0):
            for c in (1.0, 0.5, 0.1):
                rows.append(one(k0, q0, c)); r = rows[-1]
                print('k0 %.1f q0 %.1f c %.1f  aIpt/Jww %.4g  Lambda %.4g  pred %.3g  y/s %.3g  a %.3g' % (k0, q0, c, r['aIpt_over_Jww'], r['Lambda'], r['pred_4k0sq'], r['y_over_sigma'], r['a']))
    conv = [one(2.0, 1.0, 1.0, n=nn)['aIpt_over_Jww'] for nn in (64, 128, 256, 512)]
    json.dump({'rows': rows, 'grid_convergence_k2_q1': conv}, open('results/u1_lattice.json', 'w'), indent=1, default=float)
    print('convergence', conv)
