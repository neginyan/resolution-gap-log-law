"""Stage U1: a NORMALISABLE counterexample family to lemma (p~): a Gaussian envelope times a checkerboard grating.
rho(x) = N(x; 0, Sigma_E) (1 + c cos(k v) cos(q w)),  c <= 1  (so rho >= 0, total mass 1), centred at the top (pi, 0); (v, w) as in s_tools.
Gaussian x plane wave is a Gaussian with complex mean:  N(x; 0, S) e^{i kap.x} = e^{-kap'S kap/2} N(x; i S kap, S),  and the sigma-blur of
N(.; m, S) is N(.; m, S + sigma^2 I) (also for complex m).  So p, grad p, Hess p are closed form (5 complex terms); s, H follow exactly.
Fields are put on a uniform (v, w) grid and passed to t_terms / chain (the same code as for all other states).
Output results/u1_patch.json (restartable by key).  Python 3.11 compatible."""
import json, os, sys
import numpy as np
from t_tools import t_terms
from s_tools import chain

OUT = 'results/u1_patch.json'


def patch_fields(k0, q0, c, Lv, Lw, sig, h_over_sig=0.4, span=4.5, max_points=2.0e6):
    """lengths k0 = k sig, q0 = q sig; Lv, Lw envelope sd in units of sigma.  Grid in (v, w) relative to the top."""
    k, q = k0 / sig, q0 / sig; S = np.diag([(Lv * sig) ** 2, (Lw * sig) ** 2]); T = S + sig * sig * np.eye(2); Ti = np.linalg.inv(T)
    h = h_over_sig * sig; hw = h * max(1.0, 0.5 / q0)          # w-structure lives on the scale sigma / q0 (anisotropic grid)
    nv = int(2 * span * Lv * sig / h) + 1; nw = int(2 * span * Lw * sig / hw) + 1
    while nv * nw > max_points:
        h *= 1.1; hw *= 1.1; nv = int(2 * span * Lv * sig / h) + 1; nw = int(2 * span * Lw * sig / hw) + 1
    v = (np.arange(nv) - (nv - 1) / 2) * h; w = (np.arange(nw) - (nw - 1) / 2) * hw
    Vv, Ww = np.meshgrid(v, w, indexing='ij'); X = np.stack([Vv, Ww], -1)
    terms = [(1.0, np.zeros(2))] + [(c / 4, np.array([a * k, b * q])) for a in (1, -1) for b in (1, -1)]
    p = np.zeros(Vv.shape); g = np.zeros(Vv.shape + (2,)); Hs = np.zeros(Vv.shape + (2, 2)); detT = np.linalg.det(T)
    for cf, kap in terms:
        m = 1j * (S @ kap); pref = cf / (2 * np.pi * np.sqrt(detT))
        d = X - m; Tid = d @ Ti.T; N = pref * np.exp(-0.5 * kap @ S @ kap - 0.5 * np.einsum('...i,...i->...', d, Tid))   # one exponent (no overflow)
        p = p + N.real; g = g + (-Tid * N[..., None]).real
        Hs = Hs + ((np.einsum('...i,...j->...ij', Tid, Tid) - Ti) * N[..., None, None]).real
    s = g / p[..., None]; H = -(Hs / p[..., None, None] - np.einsum('...i,...j->...ij', s, s))
    dA = h * hw; mass = float(p.sum() * dA); P = p / p.sum(); s2 = sig * sig
    u = Vv + Ww + s2 * (s[..., 0] + s[..., 1])          # sqrt2 (m_q - pi) with m = y + sigma^2 s, y_q - pi = (v + w)/sqrt2
    f = {'n': Vv.shape, 'h': np.array([h, hw]), 'sig': sig, 'mass': mass, 'pw': P, 'sv': s[..., 0], 'sw': s[..., 1],
         'Hvv': H[..., 0, 0], 'Hvw': H[..., 0, 1], 'Hww': H[..., 1, 1], 'u': u, 'wcol': w}
    return f, float(p.min())


def one(k0, q0, c, Lv, Lw, sig=1.0, **kw):
    f, pmin = patch_fields(k0, q0, c, Lv, Lw, sig, **kw); t = t_terms(f); ch = chain(f)
    o = {'k0': k0, 'q0': q0, 'c': c, 'Lv': Lv, 'Lw': Lw, 'sigma': sig, 'grid_mass': f['mass'], 'p_min': pmin, 'n': list(f['n'])}
    for kk in ('aIpt_over_Jww', 'Lambda', 'a', 'y_over_sigma', 'P_sharp', 'bound_ptilde', 'bound_Lambda', 'x_minus_1', 'I_W', 'Jww',
               'sd_ptilde_mu_over_sigma', 'x_bound_ptilde', 'K_V'):
        o[kk] = float(t[kk])
    for kk in ('P', 'd', 'Z'):
        if kk in ch: o[kk] = float(ch[kk])
    o['pred_4k0sq'] = 4 * k0 * k0
    return o


if __name__ == '__main__':
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    jobs = []
    for k0 in (1.0, 1.5, 2.0):
        for L in (10, 20, 40, 80):
            jobs.append((k0, 1.0, 1.0, L, L, 1.0))
    jobs += [(1.5, 1.0, 1.0, 40, 40, 0.01), (1.5, 1.0, 1.0, 40, 40, 0.001), (1.5, 0.5, 1.0, 40, 80, 1.0), (2.0, 0.5, 1.0, 40, 80, 1.0)]
    # physical scaling: W-structure on the sqrt(sigma) scale (q0 = sqrt(sigma)), envelope Lw = 60 / q0 (in sigma units)
    for sg in (0.01, 0.001):
        q0 = float(np.sqrt(sg))
        for k0 in (1.0, 1.5, 2.0):
            jobs.append((k0, q0, 1.0, 40, round(60 / q0), sg))
    for jb in jobs:
        key = 'k%g_q%g_c%g_Lv%g_Lw%g_s%g' % jb
        if key in R: continue
        R[key] = one(*jb); json.dump(R, open(OUT, 'w'), indent=1)
        r = R[key]; print('%-34s aIpt/Jww %.3f Lam %.3f pred %.0f y/s %.4f a %.4f P# %.3g bnd %.3g mass %.6f pmin %.2e' % (
            key, r['aIpt_over_Jww'], r['Lambda'], r['pred_4k0sq'], r['y_over_sigma'], r['a'], r['P_sharp'], r['bound_ptilde'], r['grid_mass'], r['p_min']), flush=True)
