"""Stage Z: the chain quantities on ripple strips (two-scale evaluator, written anew; numpy only).

State (as stages W/X):  rho = N(v; 0, Lv^2) N(w; 0, Lw^2) [1 + c exp(-w^2 / 2 ell^2) cos(kappa v)],  kappa = k0/sigma, ell = lam sqrt(sigma), Lw = M ell, Lv = eta ell.
Blurred density (posterior of the background is Gaussian):
    p = N(y_v; Tv) N(y_w; Tw) [1 + G A(y_w) cos theta],  theta = kappa Kv y_v,  G = c exp(-kappa^2 sv^2 / 2),
    A(y_w) = ell/sqrt(ell^2 + sw^2) exp(-Kw^2 y_w^2 / (2 (ell^2 + sw^2))),   Kx = Lx^2/Tx,  sx^2 = Lx^2 sigma^2 / Tx.
Fields from log p in closed form; expectations by two-scale quadrature (phase theta uniform, slow y_v Gauss-Hermite, y_w panels).
L2 integrals (|p|^2, |d_w p|^2, |d_v d_w p|^2, |d_w p^(w)|^2 with the blur in w only, |rho|^2) in closed form: the two ripple pieces sit at
xi_v = +-kappa Kv, separated from the background and from each other by kappa Lv >> 1 (no overlaps).
Python 3.11."""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss

SQ2 = np.sqrt(2.0)


def strip_params(sig, k0, lam, M, eta, c):
    ell = lam * np.sqrt(sig); Lw = M * ell; Lv = eta * ell; return dict(sig=sig, k0=k0, c=c, ell=ell, Lw=Lw, Lv=Lv)


def strip_sums(P, n_theta=64, n_slow=24, n_gl=12, panels_per_scale=3, bg_panels=64):
    sig, k0, c, ell, Lw, Lv = P['sig'], P['k0'], P['c'], P['ell'], P['Lw'], P['Lv']; s2 = sig * sig; kap = k0 / sig
    Tv, Tw = Lv * Lv + s2, Lw * Lw + s2; Kv, Kw = Lv * Lv / Tv, Lw * Lw / Tw; sv2, sw2 = Lv * Lv * s2 / Tv, Lw * Lw * s2 / Tw
    G = c * np.exp(-0.5 * kap * kap * sv2); e2 = ell * ell + sw2
    sc = np.sqrt(e2) / Kw                                   # y_w scale of the window
    Rw = 9.0 * np.sqrt(Tw); pts = [np.linspace(-Rw, Rw, bg_panels + 1), np.arange(-14 * sc, 14 * sc * (1 + 1e-9), sc / panels_per_scale)]
    edges = np.unique(np.concatenate(pts)); edges = edges[(edges >= -Rw) & (edges <= Rw)]
    xg, wg = leggauss(n_gl); yw = ((edges[:-1, None] + edges[1:, None]) / 2 + (edges[1:, None] - edges[:-1, None]) / 2 * xg[None]).ravel()
    wyw = ((edges[1:, None] - edges[:-1, None]) / 2 * wg[None]).ravel() * np.exp(-yw * yw / (2 * Tw)) / np.sqrt(2 * np.pi * Tw)
    A = ell / np.sqrt(e2) * np.exp(-Kw * Kw * yw * yw / (2 * e2)); dA = -A * Kw * Kw * yw / e2
    ts, wts = hermegauss(n_slow); wts = wts / wts.sum(); yvs = ts * np.sqrt(Tv)
    th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta; q = kap * Kv
    keys = ('m', 'svv', 'sww', 'y2svv', 'u2svv', 'd2svv', 'lam', 'sww_w', 'phi_svv', 'jpos')
    S = {k: 0.0 for k in keys}
    for yv, wv in zip(yvs, wts):
        for t in th:
            ct, st = np.cos(t), np.sin(t)
            F = G * A * ct; den = 1 + F; Fv = -G * A * q * st; Fw = G * dA * ct; Fvw = -G * dA * q * st
            s_v = -yv / Tv + Fv / den; s_w = -yw / Tw + Fw / den
            dvw = (Fvw * den - Fv * Fw) / den ** 2; Hvw = -dvw
            Fvv = -G * A * q * q * ct; Hvv = 1 / Tv - (Fvv * den - Fv * Fv) / den ** 2
            dz = s2 * (s_v + s_w); u = yv + yw + dz; phi = 2 * np.sin(u / (2 * SQ2)) ** 2
            W = wv / n_theta * wyw * den
            S['m'] += W.sum(); S['svv'] += np.sum(W * s_v * s_v); S['sww'] += np.sum(W * s_w * s_w)
            S['y2svv'] += np.sum(W * (yv + yw) ** 2 * s_v * s_v); S['u2svv'] += np.sum(W * u * u * s_v * s_v); S['d2svv'] += np.sum(W * dz * dz * s_v * s_v)
            S['lam'] += np.sum(W * (s_v * s_w - 2 * Hvw) ** 2); S['sww_w'] += np.sum(W * s_w * s_w * (1 - phi / 4)); S['phi_svv'] += np.sum(W * s_v * s_v * phi); S['jpos'] += np.sum(W * np.maximum(Hvv, 0))
    m = S.pop('m'); E = {k: v / m for k, v in S.items()}; E['tail'] = 0.0
    return E, m


def strip_closed(P):
    """(I0, Iww, Ivvww) for p (full blur), p^(w) (blur in w only), rho (no blur), and Var_rho(z)."""
    LD = np.longdouble                                                 # 80-bit: the products reach 1e390 at sigma = 1e-96
    sig, k0, c, ell, Lw, Lv = (LD(P[k]) for k in ('sig', 'k0', 'c', 'ell', 'Lw', 'Lv')); s2 = sig * sig; kap = k0 / sig
    n = lambda S: 1 / (2 * np.sqrt(np.pi) * S); d = lambda S: 1 / (4 * np.sqrt(np.pi) * S ** 3)
    Le = 1 / np.sqrt(1 / Lw ** 2 + 1 / ell ** 2); gam = Le / Lw; win = c * c / 2 * gam * gam
    out = {}
    for name, bv, bw in (('full', s2, s2), ('wonly', LD(0), s2), ('rho', LD(0), LD(0))):
        Sv, Sw, Se = np.sqrt(Lv * Lv + bv), np.sqrt(Lw * Lw + bw), np.sqrt(Le * Le + bw)
        damp = np.exp(-kap * kap * Lv * Lv * bv / (Sv * Sv))          # |exp(-b xi^2/2)|^2 at the ripple wavenumber (exact Gaussian algebra)
        xi0 = kap * Lv * Lv / (Sv * Sv); xi2 = xi0 * xi0 + 1 / (2 * Sv * Sv)
        I0 = n(Sv) * n(Sw) + win * damp * n(Sv) * n(Se)
        Iww = n(Sv) * d(Sw) + win * damp * n(Sv) * d(Se)
        Ivvww = d(Sv) * d(Sw) + win * damp * n(Sv) * xi2 * d(Se)
        out[name] = (I0, Iww, Ivvww)
    return out, float((Lv * Lv + Lw * Lw) / 2)


def strip_terms(P):
    """the same state as complex Gaussian terms (for the code-disjoint grid check with z_tools)."""
    sig, k0, c, ell, Lw, Lv = P['sig'], P['k0'], P['c'], P['ell'], P['Lw'], P['Lv']; kap = k0 / sig
    Le2 = 1 / (1 / Lw ** 2 + 1 / ell ** 2); gam = np.sqrt(Le2) / Lw; Sb = np.diag([Lv * Lv, Lw * Lw]); Sw = np.diag([Lv * Lv, Le2])
    T = [(0j, np.zeros(2, complex), Sb)]
    for sgn in (1, -1): T.append((np.log(c * gam / 2) - 0.5 * kap * kap * Lv * Lv + 0j, np.array([1j * sgn * kap * Lv * Lv, 0]), Sw))
    return T
