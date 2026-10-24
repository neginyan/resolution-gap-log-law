"""Stage X: the separate (phase-based, two-scale) evaluator of stage W generalised to SEVERAL ripple windows.  numpy only; shares no code with
v3_tools / w_tools / w_full / u_cgm (nor with w_indep: written anew, checked against it on single windows).

State:  rho(v, w) = N(v; 0, Lv^2) N(w; 0, Lw^2) [1 + sum_r c_r exp(-(w - w_r)^2 / 2 ell_r^2) cos(m_r kappa v + phi_r)],   kappa = k_b/sigma,
        integers m_r (harmonics of the base wavenumber k_b: k and 2k superposed, or windows side by side with different k = W-varying k0),
        phases phi_r (windows that overlap in w can interfere), positivity sum_r c_r exp(...) <= 1 for all w.
Blurred density (closed form):  p(y) = N(y_v; 0, Tv) N(y_w; 0, Tw) [1 + sum_r g_r A_r(y_w) cos(m_r theta + phi_r)],  theta = kappa Kv y_v,
        g_r = c_r exp(-u_r^2/2), u_r = m_r kappa s_v,  A_r(y_w) = ell_r/sqrt(ell_r^2 + s_w^2) exp(-(mu_w - w_r)^2 / (2 (ell_r^2 + s_w^2))).
N by the posterior-covariance form N = -sigma^-2 E[Cov(x_p, x_z - sin x_z | y)], offsets in units of sigma, v-moments in closed form
(E[d^a e^{i k d}] = s^a i^a He_a(k s) e^{-(k s)^2/2}), w-moments by Gauss-Hermite; two-scale quadrature (phase theta / slow y_v / y_w panels).
Returns the gap and every term of 1 - R = (2 sigma^2 J_ww + N)/(sigma^2 (J_vv + J_ww)) (for stage X4).  Python 3.11."""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from math import factorial


def _gm(a):
    r = 1.0
    for i in range(a - 1, 0, -2): r *= i
    return r


def positivity(win, sig=1.0):
    """max_w sum_r c_r exp(-(w - w_r)^2 / 2 ell_r^2) (w_r, ell_r in sqrt(sigma) units)."""
    lo = min(wr - 9 * er for c, wr, er, m, ph in win); hi = max(wr + 9 * er for c, wr, er, m, ph in win)
    w = np.linspace(lo, hi, 8001); return float(sum(c * np.exp(-(w - wr) ** 2 / (2 * er * er)) for c, wr, er, m, ph in win).max())


def multi_gap(sig, kb, win, M, eta, n_theta=None, n_slow=16, n_gh=48, n_gl=12, panels_per_ell=3, bg_panels=48):
    """win: list of (c_r, w_r, ell_r, m_r, phi_r) with w_r, ell_r in units of sqrt(sigma); background Lw = M * ell_1, Lv = eta * ell_1."""
    s = np.sqrt(sig); s2 = sig * sig; ell1 = win[0][2] * s; Lw = M * ell1; Lv = eta * ell1; kap = kb / sig
    Tv, Tw = Lv * Lv + s2, Lw * Lw + s2; Kv, Kw = Lv * Lv / Tv, Lw * Lw / Tw
    sv, sw = Lv * sig / np.sqrt(Tv), Lw * sig / np.sqrt(Tw); svh, swh = Lv / np.sqrt(Tv), Lw / np.sqrt(Tw)
    dvar0 = s2 * (Lv * Lv - Lw * Lw) / (Tv * Tw)                    # (s_v^2 - s_w^2)/sigma^2
    q = kap * Kv; mmax = max(r[3] for r in win); n_theta = n_theta or max(48, 24 * mmax)
    t, wt = hermegauss(n_gh); wt = wt / wt.sum()
    Rw = 8.5 * np.sqrt(Tw); pts = [np.linspace(-Rw, Rw, bg_panels + 1)]
    R = []
    for c, wr, er, m, ph in win:
        e_abs = er * s; wa = wr * s; e2 = e_abs * e_abs + sw * sw; le = np.sqrt(e2)
        pts.append(np.arange(max(wa - 14 * le, -Rw), min(wa + 14 * le, Rw), le / panels_per_ell))
        ur = m * kap * sv; He = [1.0, ur]
        for n_ in range(1, 8): He.append(ur * He[n_] - n_ * He[n_ - 1])
        R.append(dict(c=c, wa=wa, e=e_abs, e2=e2, m=m, ph=ph, ur=ur, eu=np.exp(-0.5 * ur * ur), He=He, g=c * np.exp(-0.5 * ur * ur)))
    edges = np.unique(np.concatenate(pts)); xg, wg = leggauss(n_gl)
    yw = ((edges[:-1, None] + edges[1:, None]) / 2 + (edges[1:, None] - edges[:-1, None]) / 2 * xg[None]).ravel()
    ww = ((edges[1:, None] - edges[:-1, None]) / 2 * wg[None]).ravel(); muw = Kw * yw
    for r in R:
        r['A'] = r['e'] / np.sqrt(r['e2']) * np.exp(-(muw - r['wa']) ** 2 / (2 * r['e2'])); r['dA'] = -r['A'] * (muw - r['wa']) * Kw / r['e2']
        E_ = np.exp(-(muw[:, None] + sw * t[None] - r['wa']) ** 2 / (2 * r['e'] ** 2))
        r['Ew'] = [np.sum(wt[None] * (swh * t[None]) ** b * E_, 1) for b in range(9)]
        r['Ew2c'] = np.sum(wt[None] * swh * swh * (t[None] ** 2 - 1) * E_, 1)
    ts, ws_ = hermegauss(n_slow); ws_ = ws_ / ws_.sum(); yv = ts * np.sqrt(Tv)
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    keys = ('m', 'svv', 'sww', 'N', 'j', 'u2svv', 'Hvv'); acc = {k: 0.0 for k in keys}
    pw_w = ww * np.exp(-yw * yw / (2 * Tw)) / np.sqrt(2 * np.pi * Tw)
    mv = lambda a: (svh ** a * _gm(a) if a % 2 == 0 else 0.0); mw = lambda b: (swh ** b * _gm(b) if b % 2 == 0 else 0.0)
    for yv_, wv_ in zip(yv, ws_):
        muv = Kv * yv_; muz = (muv + muw) / np.sqrt(2)
        hn = [None, 2 * np.sin(muz / 2) ** 2, np.sin(muz), np.cos(muz), -np.sin(muz), -np.cos(muz), np.sin(muz), np.cos(muz)]
        for th_ in th:
            F = 0.0; F1 = 0.0; F2 = 0.0; Fw = 0.0
            Cv = []
            for r in R:
                arg = r['m'] * th_ + r['ph']; cs, sn = np.cos(arg), np.sin(arg)
                F = F + r['g'] * r['A'] * cs; F1 = F1 - r['g'] * r['A'] * r['m'] * sn; F2 = F2 - r['g'] * r['A'] * r['m'] ** 2 * cs; Fw = Fw + r['g'] * r['dA'] * cs
                ph = np.exp(1j * arg)
                Cv.append(([float(np.real(svh ** a * (1j ** a) * r['He'][a] * ph)) * r['eu'] for a in range(9)], -svh * svh * r['ur'] ** 2 * cs * r['eu']))
            den = 1 + F
            s_v = -yv_ / Tv + q * F1 / den; s_w = -yw / Tw + Fw / den
            Hvv = 1 / Tv - q * q * (F2 * den - F1 * F1) / den ** 2
            Z = 1 + sum(r['c'] * cv[0][0] * r['Ew'][0] for r, cv in zip(R, Cv))
            def Mab(a, b): return (mv(a) * mw(b) + sum(r['c'] * cv[0][a] * r['Ew'][b] for r, cv in zip(R, Cv))) / Z
            Ev, Eww = Mab(1, 0), Mab(0, 1)
            dVv = sum(r['c'] * r['Ew'][0] * cv[1] for r, cv in zip(R, Cv)) / Z - Ev * Ev
            dVw = sum(r['c'] * cv[0][0] * r['Ew2c'] for r, cv in zip(R, Cv)) / Z - Eww * Eww
            cov = hn[1] * 0.5 * (dvar0 + dVv - dVw); Edp = (Ev - Eww) / np.sqrt(2)
            for n in range(2, 8):
                Edpz = 0.0; Edz = 0.0
                for i in range(n + 1):
                    cb = factorial(n) / (factorial(i) * factorial(n - i)); Edz = Edz + cb * Mab(i, n - i); Edpz = Edpz + cb * (Mab(i + 1, n - i) - Mab(i, n - i + 1))
                cov = cov + hn[n] / factorial(n) * sig ** (n - 1) * (Edpz / 2 ** ((n + 1) / 2) - Edp * Edz / 2 ** (n / 2))
            W = wv_ / n_theta * pw_w * den
            u = yv_ + yw + s2 * (s_v + s_w); wf = np.clip(s2 * np.maximum(Hvv, 0), 0, 1)
            acc['m'] += np.sum(W); acc['svv'] += np.sum(W * s_v * s_v); acc['sww'] += np.sum(W * s_w * s_w); acc['N'] += np.sum(W * (-cov))
            acc['j'] += np.sum(W * wf); acc['u2svv'] += np.sum(W * u * u * s_v * s_v); acc['Hvv'] += np.sum(W * Hvv)
    E = {k: v / acc['m'] for k, v in acc.items() if k != 'm'}
    a = s2 * E['svv']; num = 2 * s2 * E['sww'] + E['N']; den_ = s2 * (E['svv'] + E['sww'])
    return {'gap_over_sigma': num / den_ / sig, 'a': a, 'J_vv': E['svv'], 'J_ww': E['sww'], 'N': E['N'], 'j': E['j'], 'y_over_sigma': sig * E['sww'] / E['j'],
            'P_sharp': s2 * E['u2svv'] * E['sww'] / a ** 2, 'two_sigma2_Jww': 2 * s2 * E['sww'], 'numerator': num, 'denominator': den_,
            'E_sigma2_Hvv': s2 * E['Hvv'], 'mass': acc['m'], 'Lv': Lv, 'Lw': Lw}
