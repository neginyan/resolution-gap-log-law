"""Stage AD: Gamma_2 identity for R = -d ln J_ww/dt along the heat flow (blur variance t).
With f = ln p, s = grad f, M = Hess f = -H, u = s_w, L = Delta + s.grad (symmetric in L^2(p)):
  (G1) J' = -E|grad u|^2 = -E[(H^2)_ww]
  (G2) J'' = E||Hess u||^2 + 2 E[(H^3)_ww]                       (heat equation + integration by parts)
  (G3) Bochner: E[(L u)^2] = E||Hess u||^2 + E[(H^3)_ww]
  (G4) J R' = -E[(L u + R u)^2] - E[(H^3)_ww]                    (so LC holds where E[(H^3)_ww] >= 0)
  (G5) curvature constant kappa_H := -t^2 E[(H^3)_ww]/J_ww  >=  t^2 R'.
Periodic lattices (exact cell averages, Poisson / FFT, derivatives up to order 3).  -> functions used by ad_qlc.py, run_all_AD.py.  Python 3.11."""
import numpy as np


def lattice_fields(B, t, N=128, pts=None, wts=None):
    """cell average of all quantities for the lattice B (columns = basis) with points pts (fractional coords) and weights wts per cell."""
    B = np.asarray(B, float); Bi = np.linalg.inv(B); mm = np.fft.fftfreq(N, 1.0 / N); M1, M2 = np.meshgrid(mm, mm, indexing='ij')
    K = 2 * np.pi * np.einsum('ij,jab->iab', Bi.T, np.stack([M1, M2])); C = np.exp(-t * (K[0] ** 2 + K[1] ** 2) / 2) / abs(np.linalg.det(B))
    if pts is not None:
        ph = 0
        for f_, w_ in zip(pts, wts):
            o = B @ np.asarray(f_, float); ph = ph + w_ * np.exp(-1j * (K[0] * o[0] + K[1] * o[1]))
        C = C * ph / np.sum(wts)
    ifft = lambda X: np.real(np.fft.ifft2(X)) * N * N
    I = 1j; k = (K[0], K[1])
    p = ifft(C); d1 = [ifft(I * k[a] * C) for a in range(2)]
    d2 = [[ifft(-k[a] * k[b] * C) for b in range(2)] for a in range(2)]
    d3 = [[[ifft(-I * k[a] * k[b] * k[c] * C) for c in range(2)] for b in range(2)] for a in range(2)]
    s = [d1[a] / p for a in range(2)]
    M = [[d2[a][b] / p - s[a] * s[b] for b in range(2)] for a in range(2)]
    T = [[[d3[a][b][c] / p - M[a][b] * s[c] - M[a][c] * s[b] - M[b][c] * s[a] - s[a] * s[b] * s[c] for c in range(2)] for b in range(2)] for a in range(2)]
    return p, s, M, T


def gamma2_sums(B, t, N=128, pts=None, wts=None):
    p, s, M, T = lattice_fields(B, t, N, pts, wts); w = p / p.sum(); V, W = 0, 1
    u = s[W]; gu = [M[a][W] for a in range(2)]; Hu2 = sum(gu[a] ** 2 for a in range(2))
    H = [[-M[a][b] for b in range(2)] for a in range(2)]
    H3ww = sum(H[W][a] * H[a][b] * H[b][W] for a in range(2) for b in range(2))
    HessU2 = sum(T[W][a][b] ** 2 for a in range(2) for b in range(2))
    Lu = sum(T[W][a][a] for a in range(2)) + sum(s[a] * M[a][W] for a in range(2))
    J = np.sum(w * u * u); G = np.sum(w * Hu2); R = G / J
    E = {'J': J, 'G': G, 'R': R, 'H3ww': np.sum(w * H3ww), 'HessU2': np.sum(w * HessU2), 'Lu2': np.sum(w * Lu ** 2), 'uLu': np.sum(w * u * Lu),
         'LuRu2': np.sum(w * (Lu + R * u) ** 2), 'Hvw2': np.sum(w * H[V][W] ** 2), 'Hww2': np.sum(w * H[W][W] ** 2), 'p_min_over_max': p.min() / p.max()}
    E = {k: float(v) for k, v in E.items()}; E['kappa_H'] = -t * t * E['H3ww'] / E['J']; E['tR'] = t * E['R']; return E


def identity_check(B, t, h=0.004, N=128, pts=None, wts=None):
    """finite differences in t against (G1), (G2), (G4)."""
    E = gamma2_sums(B, t, N, pts, wts); Em = gamma2_sums(B, t * (1 - h), N, pts, wts); Ep = gamma2_sums(B, t * (1 + h), N, pts, wts); dt = t * h
    J1 = (Ep['J'] - Em['J']) / (2 * dt); J2 = (Ep['J'] - 2 * E['J'] + Em['J']) / dt ** 2; R1 = (Ep['R'] - Em['R']) / (2 * dt)
    return {'G1': abs(J1 / (-E['G']) - 1), 'G2': abs(J2 / (E['HessU2'] + 2 * E['H3ww']) - 1), 'G3_uLu': abs(E['uLu'] / (-E['G']) - 1),
            'G3': abs(E['Lu2'] / (E['HessU2'] + E['H3ww']) - 1), 'G4': abs(E['J'] * R1 / (-E['LuRu2'] - E['H3ww']) - 1) if abs(E['J'] * R1) > 1e-12 else 0.0,
            't2R1': t * t * R1, 'kappa_H': E['kappa_H'], 'tR': E['tR']}


if __name__ == '__main__':
    for nm, B in (('sup', [[3.25, 3.25], [3.212, -3.212]]), ('square', [[3.0, 0.0], [0.0, 2.0]]), ('oblique', [[2.5, 0.7], [0.3, 2.2]])):
        for t in (0.6, 1.0, 1.7):
            print(nm, t, {k: '%.2e' % v for k, v in identity_check(B, t).items()})
