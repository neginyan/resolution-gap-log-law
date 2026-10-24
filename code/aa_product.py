"""Stage AA2: K_Lambda is not bounded by a universal constant -- an explicit family (product states), and its gap is large.
For a product state p(v, w) = p_v(v) p_w(w):  xbar_w(sqrt p) = sigma^2 J_v / 4,  xbar_w(p) = sigma^2 |p_v'|^2 / |p_v|^2,  so
        K_Lambda = J_v |p_v|^2 / (4 |p_v'|^2)      (exact; the w-factor cancels).
Family: rho_v = (1 - delta) N(0, L^2) + delta N(d, s_n^2) (a wide smooth part and a narrow low-density bump), rho_w = N(0, Lw^2), Lw = 1;
delta = 4 sigma (net focus a ~ delta >= 2 sigma J_ww: y/sigma ~ 1/4), L = (sigma/160)^(1/3) (so delta^2 (L/sigma)^3 <= 0.1: the bump stays negligible in L2),
s_n = 0.1 sigma, d = 4 L  ->  K_Lambda ~ delta L^2/(2 sigma^2) = 2 L^2/sigma ~ 0.07 sigma^(-1/3).
Exact 1-d quadrature (fine nonuniform grid) for J_v, j = sigma^2 E[(H_vv)_+], E[s_v^2 phi]; closed forms for |p_v|^2, |p_v'|^2.
sqrt(P_w) (= the gap/sigma up to a/den and eps) from P_w = 4 sigma^2 E[s_w^2 (1 - phi/4)] E[s_v^2 phi]/a^2 with phi = 1 - cos m_z, m_z = (m_v + m_w)/sqrt2,
m_w = Kw y_w Gaussian (the w-average in closed form).  -> results/aa_product.json.  Python 3.11."""
import json
import numpy as np


def one(sig, Lw=1.0):
    L = (sig / 160) ** (1 / 3); sn = 0.1 * sig; d = 4 * L; dl = 4 * sig
    comps = [(1 - dl, 0.0, L * L + sig * sig, L * L), (dl, d, sn * sn + sig * sig, sn * sn)]          # (weight, mean, blurred var T, prior var S)
    # closed forms
    I0 = sum(a * b * np.exp(-(m1 - m2) ** 2 / (2 * (T1 + T2))) / np.sqrt(2 * np.pi * (T1 + T2)) for a, m1, T1, _ in comps for b, m2, T2, _ in comps)
    I1 = sum(a * b * np.exp(-(m1 - m2) ** 2 / (2 * (T1 + T2))) / np.sqrt(2 * np.pi * (T1 + T2)) * (1 / (T1 + T2) - (m1 - m2) ** 2 / (T1 + T2) ** 2) for a, m1, T1, _ in comps for b, m2, T2, _ in comps)
    # quadrature grid: fine around each component, coarse elsewhere
    pts = [np.linspace(m - 12 * np.sqrt(T), m + 12 * np.sqrt(T), 40001) for _, m, T, _ in comps]; y = np.unique(np.concatenate(pts))
    wq = np.zeros_like(y); dd = np.diff(y); wq[1:] += dd / 2; wq[:-1] += dd / 2
    P = np.zeros_like(y); P1 = np.zeros_like(y); P2 = np.zeros_like(y); Mv = np.zeros_like(y)
    for a, m, T, S in comps:
        g = a * np.exp(-(y - m) ** 2 / (2 * T)) / np.sqrt(2 * np.pi * T); P += g; P1 += -g * (y - m) / T; P2 += g * ((y - m) ** 2 / T ** 2 - 1 / T)
        Mv += g * (m + S / T * (y - m))                                       # posterior mean of x_v times p
    s = P1 / P; Hvv = -(P2 / P - s * s); mv = Mv / P
    Jv = np.sum(wq * P * s * s); jpos = sig * sig * np.sum(wq * P * np.maximum(Hvv, 0))
    Tw = Lw * Lw + sig * sig; Kw = Lw * Lw / Tw; varmw = Kw * Kw * Tw; Jww = 1 / Tw
    phi_mean = 1 - np.cos(mv / np.sqrt(2)) * np.exp(-varmw / 4)               # E_w[1 - cos((m_v + m_w)/sqrt2)]
    Esvphi = np.sum(wq * P * s * s * phi_mean); a = sig * sig * Jv
    Esw_w = Jww * 1.0                                                         # E[s_w^2 (1 - phi/4)] >= J_ww/2; use the exact lower form 1/2 below
    Pw_lower = 2 * sig * sig * Jww * Esvphi / a ** 2                          # P_w >= 2 sigma^2 J_ww E[s_v^2 phi]/a^2  (1 - phi/4 >= 1/2)
    return {'sigma': sig, 'L': L, 'delta': dl, 's_n': sn, 'd': d, 'Lw': Lw, 'K_Lambda': Jv * I0 / (4 * I1), 'xbar_w_sqrtp': sig * sig * Jv / 4,
            'xbar_w_p': sig * sig * I1 / I0, 'a': a, 'y_over_sigma': sig * Jww / jpos, 'sqrtPw_lower': float(np.sqrt(Pw_lower)), 'mass_check': float(np.sum(wq * P)),
            'Lambda': sig * sig * Jv, 'F_log': float(np.log(sig * sig * Jww) + sig * sig * Jv / 4)}


if __name__ == '__main__':
    out = [one(s) for s in (1e-2, 1e-4, 1e-6, 1e-9, 1e-12)]
    json.dump(out, open('results/aa_product.json', 'w'), indent=1)
    for r in out: print('sigma %.0e  K_Lambda %.3f  y/sigma %.3f  sqrt(P_w) >= %.3g  (mass %.6f)' % (r['sigma'], r['K_Lambda'], r['y_over_sigma'], r['sqrtPw_lower'], r['mass_check']))
