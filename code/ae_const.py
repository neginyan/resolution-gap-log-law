"""Stage AE: the constant kappa_eta of the valley lemma (1-d, and every w-line in 2-d), and the resulting (qLC_w) constants.
Units t = 1 (scale).  c(y) := Var(X | Y = y) (the posterior variance along the line), h = 1 - c = -(ln p)''.
  Lemma AE1 (Laplace second difference):  ln p(y+u) + ln p(y-u) - 2 ln p(y) >= D(u, c(y)) := ln cosh(u sqrt(2 c(y))) - u^2.
  Lemma AE2 (score energy):  if p(b) >= p(a) e^Delta on an interval of length L, then int_a^b p s^2 >= 4 p(a) (e^{Delta/2} - 1)^2 / L.
  => p(y0) <= u / (4 (e^{D/4} - 1)^2) * int_{|y - y0| <= u} p s^2   (one side rises by D/2).
  Theorem AE3:  E[(c - 1)_+^3 1{c >= 1 + eta}] <= kappa_eta * J,   kappa_eta = sum_k 2 u_k Q_k  over bands c in [c_k, c_{k+1}),
                Q_k = (c_{k+1} - 1)^3 u_k / (4 (e^{D(u_k, c_k)/4} - 1)^2),  u_k optimised per band.
  (Tonelli: swap the y0 and y integrals; in each band the set of y0 within u_k of y has length <= 2 u_k.)
Tail c >= C_TAIL: bound with u = sqrt(c_k/2) per band; the band sum is continued until the terms are < 1e-30 of the total.
-> results/ae_const.json.  Python 3.11."""
import json
import numpy as np


def D(u, c): return np.log(np.cosh(u * np.sqrt(2 * c))) - u * u if u * np.sqrt(2 * c) < 700 else u * np.sqrt(2 * c) - np.log(2) - u * u


def Dv(u, c):
    x = u * np.sqrt(2 * c); return np.where(x < 300, np.log(np.cosh(np.minimum(x, 300))), x - np.log(2)) - u * u


def kappa_eta(eta, n=240, c_max=2.0e4, nu=160, sub=6):
    """optimal partition of [1 + eta, c_max] into bands (dynamic programming over a geometric grid in c - 1);
    band cost = min_u 2u * sup_{c in band} (c-1)^3 u / (4 (e^{D(u,c)/4} - 1)^2)  (sup taken on a sub-grid of the band
    plus both ends; D increases and (c-1)^3 increases in c, so between sub-grid points the expression is bounded by
    (c_next - 1)^3 / ... (D at c_prev): we use that conservative pairing between consecutive sub-grid points)."""
    cg = 1 + np.geomspace(eta, c_max - 1, n * sub + 1); U = np.geomspace(1e-3, 60.0, nu)
    lo, hi = cg[:-1], cg[1:]                                       # sub-intervals: conservative value g(hi) / (e^{D(u, lo)/4} - 1)^2
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        Dm = Dv(U[:, None], lo[None, :]); Fm = np.where(Dm > 0, (hi[None, :] - 1) ** 3 * U[:, None] / (4 * np.expm1(np.minimum(Dm, 1400) / 4) ** 2), np.inf)
    nb = n; best = np.full(nb + 1, np.inf); best[0] = 0.0; arg = [None] * (nb + 1)
    for i in range(nb):
        if not np.isfinite(best[i]): continue
        run = np.full(nu, 0.0)
        for j in range(i + 1, nb + 1):
            run = np.maximum(run, Fm[:, (j - 1) * sub:j * sub].max(axis=1)); cost = np.min(2 * U * run); k = int(np.argmin(2 * U * run))
            if best[i] + cost < best[j]: best[j] = best[i] + cost; arg[j] = (i, float(U[k]), float(cost))
    # tail [c_max, infinity): one band with a single u; (c-1)^3 e^{-D/2} decreases fast -> sup over c in [c_max, 1e7]
    ct = np.geomspace(c_max, 1e7, 4000)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        tails = [(2 * u * np.max((ct[1:] - 1) ** 3 * u / (4 * np.expm1(np.minimum(Dv(u, ct[:-1]), 1400) / 4) ** 2)), u) for u in (20.0, 40.0, 80.0)]
    tail = min(tails); bands = []; j = nb
    while j > 0:
        i, u, cost = arg[j]; bands.append((float(cg[i * sub]), float(cg[j * sub]), u, cost)); j = i
    return float(best[nb] + tail[0]), bands[::-1] + [(c_max, np.inf, tail[1], float(tail[0]))]


def qlcw(eta3, kap, tau):
    """t^2 R' <= eta3 * tR + kap  =>  (qLC_w): K = (1+eta3)(1-tau)/(1-tau^(1+eta3)), c0 = kap [ln(1/tau)/(1 - tau^(1+eta3)) - 1/(1+eta3)]."""
    e = eta3; K = (1 + e) * (1 - tau) / (1 - tau ** (1 + e)); c0 = kap * (np.log(1 / tau) / (1 - tau ** (1 + e)) - 1 / (1 + e))
    B = 8 * K / (1 - tau); A = 2 + 8 * c0 + B * np.log(1 / tau); return {'K': K, 'c0': c0, 'B': B, 'A': A, 'c': 1 / (np.pi * np.sqrt(B)),
                                                                          'C_sigma1': 0.5 * np.log(2 * (np.pi ** 2 + 1)) + A / (2 * B)}


if __name__ == '__main__':
    out = {'kappa_eta': {}, 'bands_sample': {}, 'qlcw_1d': {}}
    for eta in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        k, b = kappa_eta(eta); k2, _ = kappa_eta(eta, n=360); out['kappa_eta']['%g' % eta] = {'kappa': k, 'kappa_finer_grid': k2, 'n_bands': len(b)}
        out['bands_sample']['%g' % eta] = [list(map(float, x)) for x in b]
        print('eta %.2f  kappa_eta %.4f  (finer grid %.4f)  bands %d' % (eta, k, k2, len(b)), flush=True)
    # 1-d theorem: t^2 R' <= eta tR + kappa_eta (the shallow part needs no factor 3 in 1-d)
    for eta in ('0.05', '0.1', '0.2', '0.5'):
        for tau in (0.25, 0.5, 0.1):
            out['qlcw_1d']['eta%s_tau%.2f' % (eta, tau)] = qlcw(float(eta), out['kappa_eta'][eta]['kappa'], tau)
    json.dump(out, open('results/ae_const.json', 'w'), indent=1, default=float)
    for k, v in out['qlcw_1d'].items(): print(k, {a: round(float(b), 4) for a, b in v.items()})
