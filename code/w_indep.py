"""Stage W (main-line request 1): a COMPLETELY SEPARATE evaluator of the strip gap.  It shares nothing with v3_tools / w_tools / w_full / u_cgm:
not the state construction, not the point fields, not the integration.  Only numpy is imported.

State (ripple strip, ripple filling the strip):  rho(v, w) = N(v; 0, Lv^2) N(w; 0, Lw^2) [1 + c exp(-w^2/2 ell^2) cos(kappa v)],  kappa = k0/sigma.

1. Closed form of the blurred density (own derivation).  With the background posterior x_v|y ~ N(mu_v, s_v^2), x_w|y ~ N(mu_w, s_w^2),
   mu_v = Kv y_v, Kv = Lv^2/Tv, s_v^2 = Lv^2 sigma^2/Tv (same for w), Tv = Lv^2 + sigma^2:
     p(y) = N(y_v; 0, Tv) N(y_w; 0, Tw) [1 + g A(y_w) cos(theta)],  theta = kappa Kv y_v,  g = c exp(-kappa^2 s_v^2 / 2),
     A(y_w) = ell/sqrt(ell^2 + s_w^2) exp(-mu_w^2 / (2 (ell^2 + s_w^2))).
   Scores and H_vv analytically from this.
2. N by the POSTERIOR-COVARIANCE form N = -sigma^-2 E_y[ Cov(x_p, h(x_z) | y) ], h(z) = z - sin z (not the d_p D form used elsewhere).
   The posterior is the background Gaussian reweighted by W = 1 + c E(w) cos(kappa v); with x = mu + delta,
     Cov(x_p, h(x_z)) = sum_{n>=1} h^(n)(mu_z)/n! Cov(delta_p, delta_z^n)      (Taylor in delta ~ sigma; n <= 7),
   posterior moments of delta: the v-factor in closed form (E[d^a e^{i kappa d}] = s^a i^a He_a(u) e^{-u^2/2}), the w-factor by 1-d Gauss-Hermite
   quadrature (delta_v, delta_w independent under the background);
   the n = 1 term is formed as (Var delta_v - Var delta_w)/2 with s_v^2 - s_w^2 = sigma^4 (Lv^2 - Lw^2)/(Tv Tw) exactly and the ripple corrections
   integrated with (delta^2 - s^2) integrands (no cancellation).
3. Integration by two-scale averaging: theta (fast phase) by the periodic trapezoid rule, y_v (slow envelope) by Gauss-Hermite,
   y_w by composite Gauss-Legendre panels (fine on the window, coarse on the background).
4. gap/sigma from 1 - R = (2 sigma^2 J_ww + N) / (sigma^2 (J_vv + J_ww)); also y/sigma = sigma J_ww / j and P#.
Python 3.11."""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from math import factorial


def strip_gap(sig, k0, lam, M, eta, c, n_theta=48, n_slow=16, n_gh=48, n_gl=12, win_panels_per_ell=3, bg_panels=48):
    s = np.sqrt(sig); ell = lam * s; Lw = M * ell; Lv = eta * ell; kap = k0 / sig; s2 = sig * sig
    Tv, Tw = Lv * Lv + s2, Lw * Lw + s2; Kv, Kw = Lv * Lv / Tv, Lw * Lw / Tw
    sv2, sw2 = Lv * Lv * s2 / Tv, Lw * Lw * s2 / Tw; sv, sw = np.sqrt(sv2), np.sqrt(sw2)
    # posterior offsets are handled in units of sigma (delta = sigma * d), so that no sigma^4 or sigma^n products can underflow
    svh, swh = Lv / np.sqrt(Tv), Lw / np.sqrt(Tw)
    dvar0 = s2 * (Lv * Lv - Lw * Lw) / (Tv * Tw)                              # (s_v^2 - s_w^2)/sigma^2, exact
    g = c * np.exp(-0.5 * (kap * sv) ** 2); q = kap * Kv; e2 = ell * ell + sw2
    t, wt = hermegauss(n_gh); wt = wt / wt.sum()                              # E over N(0,1)
    # ---- y_w panels
    Rw = 8.5 * np.sqrt(Tw); le = np.sqrt(e2)
    ew = np.arange(-min(14 * le, Rw), min(14 * le, Rw) + 1e-300, le / win_panels_per_ell)
    edges = np.unique(np.concatenate([np.linspace(-Rw, Rw, bg_panels + 1), ew[np.abs(ew) < Rw]]))
    xg, wg = leggauss(n_gl); yw = ((edges[:-1, None] + edges[1:, None]) / 2 + (edges[1:, None] - edges[:-1, None]) / 2 * xg[None]).ravel()
    ww = ((edges[1:, None] - edges[:-1, None]) / 2 * wg[None]).ravel()
    # ---- slow y_v and fast theta (independent in the two-scale limit Lv >> sigma/k0)
    ts, ws_ = hermegauss(n_slow); ws_ = ws_ / ws_.sum(); yv = ts * np.sqrt(Tv)
    th = 2 * np.pi * np.arange(n_theta) / n_theta; wth = np.full(n_theta, 1.0 / n_theta)
    # w-factor moments (functions of y_w): Ew_b = E0[delta_w^b E(mu_w + delta_w)],  E(w) = exp(-w^2/2 ell^2)
    muw = Kw * yw; A = ell / np.sqrt(e2) * np.exp(-muw * muw / (2 * e2)); dA = -A * muw * Kw / e2
    Ew = [np.sum(wt[None] * (swh * t[None]) ** b * np.exp(-(muw[:, None] + sw * t[None]) ** 2 / (2 * ell * ell)), 1) for b in range(9)]
    Ew2c = np.sum(wt[None] * swh * swh * (t[None] ** 2 - 1) * np.exp(-(muw[:, None] + sw * t[None]) ** 2 / (2 * ell * ell)), 1)
    uu = kap * sv; eu = np.exp(-0.5 * uu * uu); He = [1.0, uu]
    for n_ in range(1, 8): He.append(uu * He[n_] - n_ * He[n_ - 1])
    acc = {k: 0.0 for k in ('m', 'svv', 'sww', 'N', 'j', 'u2svv')}
    for yv_, wv_ in zip(yv, ws_):
        muv = Kv * yv_
        for th_, wth_ in zip(th, wth):
            # density factor and fields on the y_w line
            cw = np.cos(th_); sn = np.sin(th_); den = 1 + g * A * cw
            pw_rel = den                                                           # p / (N(y_v) N(y_w)); N(y_v) is in the Hermite weight
            s_v = -yv_ / Tv - g * A * q * sn / den; s_w = -yw / Tw + g * dA * cw / den
            Hvv = 1 / Tv + q * q * g * A * (cw + g * A) / den ** 2
            # v-factor moments at this phase: Cv_a = E0[delta_v^a cos(kappa (mu_v + delta_v))], kappa mu_v = theta (mod 2 pi) + kappa (Kv yv - mu_v) = theta
            # closed form (no quadrature: the e^{-u^2/2} smallness must not come from a cancelling sum):
            #   E[delta^a e^{i(theta + kappa delta)}] = s^a i^a He_a(u) e^{-u^2/2} e^{i theta},  u = kappa s_v
            Cv = [float(np.real(svh ** a * (1j ** a) * He[a] * np.exp(1j * th_))) * eu for a in range(9)]
            Cv2c = -svh * svh * uu * uu * np.cos(th_) * eu
            Z = 1 + c * Cv[0] * Ew[0]
            mv = lambda a: (svh ** a * _gm(a) if a % 2 == 0 else 0.0)
            # posterior moments M_ab = E0[W dv^a dw^b] / Z
            def Mab(a, b): return (mv(a) * (swh ** b * _gm(b) if b % 2 == 0 else 0.0) + c * Cv[a] * Ew[b]) / Z
            Ev, Eww = Mab(1, 0), Mab(0, 1)
            dVv = c * Ew[0] * Cv2c / Z - Ev * Ev; dVw = c * Cv[0] * Ew2c / Z - Eww * Eww
            cov1 = 0.5 * (dvar0 + dVv - dVw)
            muz = (muv + muw) / np.sqrt(2)
            hn = [None, 2 * np.sin(muz / 2) ** 2, np.sin(muz), np.cos(muz), -np.sin(muz), -np.cos(muz), np.sin(muz), np.cos(muz)]
            cov = hn[1] * cov1
            for n in range(2, 8):                                                  # Cov(dp, dz^n), dp = (dv - dw)/r2, dz = (dv + dw)/r2
                Edpz = 0.0; Edz = 0.0
                for i in range(n + 1):
                    cb = factorial(n) / (factorial(i) * factorial(n - i))
                    Edz = Edz + cb * Mab(i, n - i)
                    Edpz = Edpz + cb * (Mab(i + 1, n - i) - Mab(i, n - i + 1))
                Edz = Edz / 2 ** (n / 2); Edpz = Edpz / 2 ** ((n + 1) / 2); Edp = (Ev - Eww) / np.sqrt(2)
                cov = cov + hn[n] / factorial(n) * sig ** (n - 1) * (Edpz - Edp * Edz)
            Npt = -cov                                                             # cov is Cov(x_p, h)/sigma^2 already
            u = yv_ + yw + s2 * (s_v + s_w); wf = np.clip(s2 * np.maximum(Hvv, 0), 0, 1)
            W = wv_ * wth_ * ww * np.exp(-yw * yw / (2 * Tw)) / np.sqrt(2 * np.pi * Tw) * pw_rel
            acc['m'] += np.sum(W); acc['svv'] += np.sum(W * s_v * s_v); acc['sww'] += np.sum(W * s_w * s_w); acc['N'] += np.sum(W * Npt)
            acc['j'] += np.sum(W * wf); acc['u2svv'] += np.sum(W * u * u * s_v * s_v)
    E = {k: v / acc['m'] for k, v in acc.items() if k != 'm'}
    a = s2 * E['svv']; omr = (2 * s2 * E['sww'] + E['N']) / (s2 * (E['svv'] + E['sww']))
    return {'gap_over_sigma': omr / sig, 'a': a, 'J_ww': E['sww'], 'N': E['N'], 'y_over_sigma': sig * E['sww'] / E['j'], 'P_sharp': s2 * E['u2svv'] * E['sww'] / a ** 2,
            'mass': acc['m']}


def _gm(a):
    """E[t^a] for t ~ N(0,1), a even."""
    r = 1.0
    for i in range(a - 1, 0, -2): r *= i
    return r
