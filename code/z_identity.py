"""Stage Z: numerical checks of the exact identities and inequalities of Theorem Z by a spectral (FFT) method on a uniform periodic grid
(independent of the expectation sums of z_tools.grid_sums):
  (a) Lambda = sigma^2 E[(s_v s_w - 2 H_vw)^2] / J_ww = 4 xbar_w(sqrt p)     (h = sqrt p: s_w sqrt p = 2 d_w h, (s_v s_w - 2 H_vw) sqrt p = 4 d_v d_w h)
  (b) slice-wise Heisenberg:  E[s_v^2 y_z^2] >= sigma^2 J_vv^2 / (2 Lambda J_ww)   (i.e. P_y >= 1/Lambda)
  (c) 2-d Jensen:  |d_w p|^2 <= exp(-xbar_w(p)) |d_w p^(w)|^2
  (d) Cramer-Rao:  J_ww >= e_w^T Cov(y)^-1 e_w >= 1/(2 Var(y_z))
on two-cluster states and random tilted Gaussian mixtures.  -> results/z_identity.json.  Python 3.11."""
import json
import numpy as np
from z_tools import fields, _prep

RM = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)


def spectral(terms, sig, n=1024, ext=7.0):
    s2 = sig * sig; T = [np.asarray(S, float) + s2 * np.eye(2) for _, _, S in terms]
    cv = [abs(m[0].real) + ext * np.sqrt(t[0, 0]) for (_, m, _), t in zip(terms, T)]; cw = [abs(m[1].real) + ext * np.sqrt(t[1, 1]) for (_, m, _), t in zip(terms, T)]
    Lv, Lw = max(cv), max(cw); v = np.linspace(-Lv, Lv, n, endpoint=False); w = np.linspace(-Lw, Lw, n, endpoint=False); hv, hw = v[1] - v[0], w[1] - w[0]
    V, W = np.meshgrid(v, w, indexing='ij'); x = np.stack([V.ravel(), W.ravel()], 1)
    p, g, Hs = fields(_prep(terms, sig), x); p = p.reshape(n, n); dA = hv * hw; good = p > 1e-250; p = np.where(good, p, 1e-250)
    g = g * good.ravel()[:, None]; Hs = Hs * good.ravel()[:, None, None]
    sv, sw = g[:, 0].reshape(n, n) / p, g[:, 1].reshape(n, n) / p; Hvw = -(Hs[:, 0, 1].reshape(n, n) / p - sv * sw)
    mass = p.sum() * dA; Jvv = np.sum(p * sv * sv) * dA / mass; Jww = np.sum(p * sw * sw) * dA / mass
    Lam = s2 * np.sum(p * (sv * sw - 2 * Hvw) ** 2) * dA / mass / Jww
    xiv = 2 * np.pi * np.fft.fftfreq(n, d=hv); xiw = 2 * np.pi * np.fft.fftfreq(n, d=hw); XV, XW = np.meshgrid(xiv, xiw, indexing='ij')
    def xbar(f):
        F = np.abs(np.fft.fft2(f)) ** 2; return s2 * np.sum(XV ** 2 * XW ** 2 * F) / np.sum(XW ** 2 * F)
    xh, xp = xbar(np.sqrt(np.maximum(p, 0))), xbar(p)
    yz = (V + W) / np.sqrt(2); Ey = np.sum(p * sv * sv * yz * yz) * dA / mass
    # Cramer-Rao
    m1 = np.array([np.sum(p * V), np.sum(p * W)]) * dA / mass; C = np.array([[np.sum(p * (V - m1[0]) ** 2), np.sum(p * (V - m1[0]) * (W - m1[1]))],
                                                                              [0, np.sum(p * (W - m1[1]) ** 2)]]) * dA / mass; C[1, 0] = C[0, 1]
    Ci = np.linalg.inv(C); Vz = np.array([1, 1]) @ C @ np.array([1, 1]) / 2
    # Jensen: |d_w p|^2 vs exp(-xbar_w) |d_w p^(w)|^2, p^(w) by removing the v-blur in Fourier space (exact for Gaussian blur)
    P = np.fft.fft2(p * good); Fw = XW ** 2 * np.abs(P) ** 2; ok = s2 * XV ** 2 < 18.0     # de-blur only where numerically meaningful
    Fw_unblur_v = np.where(ok, Fw * np.exp(np.minimum(s2 * XV ** 2, 18.0)), 0.0)          # truncation only lowers the right side
    return {'Lambda': float(Lam), 'four_xbar_sqrtp': float(4 * xh), 'rel_identity': float(abs(Lam / (4 * xh) - 1)), 'xbar_p': float(xp), 'K_Lambda': float(Lam / (4 * xp)),
            'heis_lhs': float(Ey), 'heis_rhs': float(s2 * Jvv ** 2 / (2 * Lam * Jww)), 'Jww': float(Jww), 'CR_full': float(Ci[1, 1]), 'CR_z': float(1 / (2 * (Vz))),
            'jensen_lhs': float(np.sum(Fw)), 'jensen_rhs': float(np.exp(-xp) * np.sum(Fw_unblur_v)), 'mass': float(mass)}


if __name__ == '__main__':
    from u2_chain import two_clusters
    out = []; rng = np.random.default_rng(5); TOP = np.array([np.pi, 0.0])
    for V0 in (2, 20):
        ws, mus, Ss = two_clusters(V0, 0.01); terms = [(np.log(a), RM @ (np.asarray(m) - TOP), RM @ np.asarray(S) @ RM.T) for a, m, S in zip(ws, mus, Ss)]
        r = spectral(terms, 0.01); r['state'] = 'two clusters V0=%g' % V0; out.append(r)
    for t in range(6):
        sig = 0.01; K = 3 + t % 3; terms = []
        for k in range(K):
            th = rng.uniform(0, np.pi); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); d = np.diag([rng.uniform(0.3, 6) * sig, rng.uniform(1, 8) * np.sqrt(sig)]) ** 2
            terms.append((np.log(1.0 / K), np.array([rng.uniform(-3, 3) * sig, rng.uniform(-4, 4) * np.sqrt(sig)]), R @ d @ R.T))
        r = spectral(terms, sig); r['state'] = 'random tilted mixture K%d #%d' % (K, t); out.append(r)
    json.dump(out, open('results/z_identity.json', 'w'), indent=1)
    for r in out: print('%-32s Lambda %.5f 4xbar(sqrt p) %.5f rel %.1e | heis %.4g >= %.4g | Jww %.4g >= CR %.4g >= %.4g | jensen %.4g <= %.4g | K_Lam %.3f' % (
        r['state'], r['Lambda'], r['four_xbar_sqrtp'], r['rel_identity'], r['heis_lhs'], r['heis_rhs'], r['Jww'], r['CR_full'], r['CR_z'], r['jensen_lhs'], r['jensen_rhs'], r['K_Lambda']))
