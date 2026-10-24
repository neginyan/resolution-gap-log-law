"""Stage V1: the V-periodic limit of the ripple strip (exact, closed form).
rho(v, w) = N(w; 0, Lw^2) [1 + c exp(-w^2/(2 ell^2)) cos(k v)]   (periodic in v; the limit 'strip width -> infinity' of v2_strip, in which the
background focus sigma^2/Lv^2 vanishes).  Blur: p = N(w; 0, Lw^2 + s^2) + c sqrt(S2/Lw^2) e^{-k0^2/2} N(w; 0, S2 + s^2) cos(k v),
S2 = (1/Lw^2 + 1/ell^2)^{-1}.  One V-period, symmetric about v = 0 (so E_r[v | w] = 0); u = v + w + sigma^2 (s_v + s_w).
Computes B1, B2 = J_ww E_p~[ubar^2]/a (the weight-free moment without the within-line variance; within a strip of width Lv << ell
P# -> B2), a, a I(p~)/J_ww, and compares B2 with the small-amplitude, no-background prediction 1/(4 k0^2).
-> results/v1_vperiodic.json.  Python 3.11 compatible."""
import json
import numpy as np
from v_tools import v_terms
from u_cgm import nodes, trap_weights


def vper_fields(k0, lam, M, c=1.0, sig=1.0, nv=128, nw_win=60):
    k = k0 / sig; ell = lam * sig; Lw = M * ell; S2 = 1 / (1 / Lw ** 2 + 1 / ell ** 2); A = c * np.sqrt(S2 / Lw ** 2) * np.exp(-k0 ** 2 / 2)
    T0 = Lw ** 2 + sig ** 2; T2 = S2 + sig ** 2
    v = (np.arange(nv) - nv / 2 + 0.5) * (2 * np.pi / k) / nv
    w = nodes(7 * np.sqrt(T2), np.sqrt(T2) / nw_win, 7 * np.sqrt(T0), np.sqrt(T0) / 60)      # fine across the window, coarse on the background
    V, W = np.meshgrid(v, w, indexing='ij')
    g0 = np.exp(-W ** 2 / (2 * T0)) / np.sqrt(2 * np.pi * T0); g2 = np.exp(-W ** 2 / (2 * T2)) / np.sqrt(2 * np.pi * T2)
    cv, sv_ = np.cos(k * V), np.sin(k * V)
    p = g0 + A * g2 * cv
    pv = -A * k * g2 * sv_; pw = -W / T0 * g0 - A * W / T2 * g2 * cv
    pvv = -A * k * k * g2 * cv; pww = (W ** 2 / T0 ** 2 - 1 / T0) * g0 + A * (W ** 2 / T2 ** 2 - 1 / T2) * g2 * cv; pvw = A * k * W / T2 * g2 * sv_
    s_v, s_w = pv / p, pw / p
    Hvv = -(pvv / p - s_v ** 2); Hww = -(pww / p - s_w ** 2); Hvw = -(pvw / p - s_v * s_w)
    P = p * trap_weights(w)[None, :]; P = P / P.sum()
    f = {'n': p.shape, 'h': [v[1] - v[0], w], 'sig': sig, 'mass': 1.0, 'pw': P, 'sv': s_v, 'sw': s_w, 'Hvv': Hvv, 'Hvw': Hvw, 'Hww': Hww,
         'u': V + W + sig ** 2 * (s_v + s_w), 'wcol': w}
    return f, A


if __name__ == '__main__':
    rows = []
    for k0 in (1.0, 1.5, 2.0, 2.5):
        for lam in (1.0, 3.0, 10.0):
            for M in (10, 30, 100, 300, 1000, 3000):
                f, A = vper_fields(k0, lam, M); v = v_terms(f)
                rows.append({'k0': k0, 'lam': lam, 'M': M, 'A': A, 'B1': v['B1'], 'B2': v['B2'], 'a': v['a'], 'aIpt_over_Jww': v['aIpt_over_Jww'], 'x': v['x'],
                             'pred_no_background': 1 / (4 * k0 ** 2)})
                print('k0 %.1f lam %4.1f M %5d: B2 %.4f (pred %.4f) B1 %.4f a %.2e c0 %.2f x %.4f' % (k0, lam, M, v['B2'], 1 / (4 * k0 ** 2), v['B1'], v['a'], v['aIpt_over_Jww'], v['x']), flush=True) if M >= 100 else None
    f1, _ = vper_fields(1.5, 3.0, 1000, nv=64, nw_win=40); f2, _ = vper_fields(1.5, 3.0, 1000, nv=256, nw_win=120)
    conv = abs(v_terms(f1)['B2'] / v_terms(f2)['B2'] - 1)
    json.dump({'rows': rows, 'grid_convergence_rel': conv}, open('results/v1_vperiodic.json', 'w'), indent=1, default=float); print('grid conv', conv)
