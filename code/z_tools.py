"""Stage Z: tools for the conditional log-law theorem (new code; numpy only; shares no code with v3_tools / u_cgm / w_tools / x_indep).

States are lists of Gaussian terms (lc, m, S) in (v, w) coordinates relative to the top:  rho = sum_k exp(lc_k) N(m_k, S_k)
(lc, m may be complex: ripple terms; the sum is real).  p = rho * N(0, sigma^2 I).

grid_sums(terms, sig, vn, wn): streamed 2-d trapezoid sums of the point fields built from p, grad p, Hess p only
    (s = grad p / p, H = -Hess p / p + s s^T, Tweedie m = y + sigma^2 s; no posterior-per-term code):
    Fisher-type expectations E[.] (weight p) and L2 integrals (weight 1).
pair_sums(terms, sig, blur): closed-form L2 integrals sum_kl from Gaussian overlaps (second, grid-free method):
    I0 = int f^2,  Iww = int (d_w f)^2,  Ivvww = int (d_v d_w f)^2  for f = rho blurred by N(0, blur).
moments(terms): Var_rho(z), z = (v + w)/sqrt 2.
Python 3.11."""
import numpy as np

SQ2 = np.sqrt(2.0)
EKEYS = ('svv', 'sww', 'y2svv', 'u2svv', 'd2svv', 'lam', 'sww_w', 'phi_svv', 'tail', 'jpos')
LKEYS = ('p2', 'pw2', 'pvw2', 'pv2')


def _prep(terms, sig):
    s2 = sig * sig; pre = []
    for lc, m, S in terms:
        T = np.asarray(S, float) + s2 * np.eye(2); Ti = np.linalg.inv(T)
        pre.append((complex(lc) - np.log(2 * np.pi * np.sqrt(np.linalg.det(T))), np.asarray(m, complex), Ti))
    return pre


def fields(pre, x):
    """p, grad p, Hess p at real points x (n, 2)."""
    P = np.zeros(len(x), complex); G = np.zeros((len(x), 2), complex); Hs = np.zeros((len(x), 2, 2), complex)
    for c0, m, Ti in pre:
        d = x - m; Td = d @ Ti.T; Nt = np.exp(c0 - 0.5 * np.einsum('ni,ni->n', d, Td))
        P += Nt; G += -Td * Nt[:, None]; Hs += (np.einsum('ni,nj->nij', Td, Td) - Ti[None]) * Nt[:, None, None]
    return P.real, G.real, Hs.real


def grid_sums(terms, sig, vn, wn, rows=64):
    pre = _prep(terms, sig); s2 = sig * sig
    wv = np.zeros_like(vn); d = np.diff(vn); wv[1:] += d / 2; wv[:-1] += d / 2
    ww = np.zeros_like(wn); d = np.diff(wn); ww[1:] += d / 2; ww[:-1] += d / 2
    E = {k: 0.0 for k in EKEYS + ('mass',)}; L = {k: 0.0 for k in LKEYS}
    for a in range(0, len(vn), rows):
        V, W = np.meshgrid(vn[a:a + rows], wn, indexing='ij'); x = np.stack([V.ravel(), W.ravel()], 1)
        dA = np.outer(wv[a:a + rows], ww).ravel()
        p, g, Hs = fields(pre, x)
        ok = p > 0; p = np.where(ok, p, 1e-300)
        sv, sw = g[:, 0] / p, g[:, 1] / p
        Hvw = -(Hs[:, 0, 1] / p - sv * sw); Hvv = -(Hs[:, 0, 0] / p - sv * sv)
        yv, yw = x[:, 0], x[:, 1]
        dz = s2 * (sv + sw)                                  # sqrt2 (m_z - y_z)
        u = yv + yw + dz                                     # sqrt2 m_z
        mz = u / SQ2; phi = 2 * np.sin(mz / 2) ** 2
        wt = p * dA * ok
        E['mass'] += wt.sum(); E['svv'] += np.sum(wt * sv * sv); E['sww'] += np.sum(wt * sw * sw)
        E['y2svv'] += np.sum(wt * (yv + yw) ** 2 * sv * sv); E['u2svv'] += np.sum(wt * u * u * sv * sv); E['d2svv'] += np.sum(wt * dz * dz * sv * sv)
        E['lam'] += np.sum(wt * (sv * sw - 2 * Hvw) ** 2); E['sww_w'] += np.sum(wt * sw * sw * (1 - phi / 4)); E['phi_svv'] += np.sum(wt * sv * sv * phi)
        E['tail'] += np.sum(wt * (np.abs(mz) > np.pi)); E['jpos'] += np.sum(wt * np.maximum(Hvv, 0))
        L['p2'] += np.sum(dA * p * p * ok); L['pw2'] += np.sum(dA * g[:, 1] ** 2); L['pv2'] += np.sum(dA * g[:, 0] ** 2)
        L['pvw2'] += np.sum(dA * Hs[:, 0, 1] ** 2)
    m = E.pop('mass'); E = {k: v / m for k, v in E.items()}
    return E, L, m


def pair_sums(terms, blur):
    """closed-form L2 integrals of f = rho * N(0, blur) (blur: 2x2 matrix):  I0 = |f|^2, Iww = |d_w f|^2, Ivvww = |d_v d_w f|^2."""
    A = [np.asarray(S, float) + blur for lc, m, S in terms]
    I0 = Iww = Ivvww = 0j
    for k, (lck, mk, Sk) in enumerate(terms):
        for l, (lcl, ml, Sl) in enumerate(terms):
            Sg = A[k] + A[l]; Q = np.linalg.inv(Sg); d = np.asarray(mk, complex) - np.asarray(ml, complex); g = Q @ d
            Phi = np.exp(complex(lck) + complex(lcl) - np.log(2 * np.pi * np.sqrt(np.linalg.det(Sg))) - 0.5 * d @ Q @ d)
            gv, gw = g; Qvv, Qww, Qvw = Q[0, 0], Q[1, 1], Q[0, 1]
            I0 += Phi; Iww += (Qww - gw * gw) * Phi
            Ivvww += (gv * gv * gw * gw - Qvv * gw * gw - Qww * gv * gv - 4 * Qvw * gv * gw + Qvv * Qww + 2 * Qvw * Qvw) * Phi
    return float(I0.real), float(Iww.real), float(Ivvww.real)


def moments(terms):
    ez = np.array([1.0, 1.0]) / SQ2; M0 = M1 = M2 = 0j
    for lc, m, S in terms:
        wk = np.exp(complex(lc)); mz = ez @ np.asarray(m, complex); M0 += wk; M1 += wk * mz; M2 += wk * (ez @ np.asarray(S, float) @ ez + mz * mz)
    M0, M1, M2 = M0.real, M1.real, M2.real
    return M2 / M0 - (M1 / M0) ** 2


def chain(sig, E, L, closed, Vz_rho, gap=None, den_factor=None, eps=None, P_w=None):
    """every quantity of the stage-Z chain from the sums.  closed = dict(full=(I0, Iww, Ivvww), wonly=(...), rho=(...))."""
    s2 = sig * sig; Jvv, Jww = E['svv'], E['sww']; a = s2 * Jvv
    Psharp = s2 * E['u2svv'] * Jww / a ** 2                         # = 2 s^2 E[s_v^2 m_z^2] J_ww / a^2
    Py = s2 * E['y2svv'] * Jww / a ** 2                             # = 2 s^2 E[s_v^2 y_z^2] J_ww / a^2
    eps_m = np.sqrt(E['d2svv'] / E['y2svv'])
    Lam = s2 * E['lam'] / Jww
    I0, Iww, Ivvww = closed['full']; I0w, Iwww, _ = closed['wonly']; I0r = closed['rho'][0]
    LD = np.longdouble
    xw = float(LD(s2) * LD(Ivvww) / LD(Iww))                       # spectral mean of sigma^2 xi_v^2 under xi_w^2 |p^|^2
    xw_grid = s2 * L['pvw2'] / L['pw2']
    K_Lam = Lam / (4 * xw); K_J = float(LD(Jww) * LD(I0) / LD(Iww)); R_w = float(LD(Iwww) / LD(I0)); K_rho = float(LD(I0r) / LD(I0)) if np.isfinite(I0r) else np.inf
    Vz = Vz_rho + s2
    jensen_rhs = float(np.log(LD(Iwww) / LD(Iww)))                                 # x_w <= ln(|d_w p^(w)|^2 / |d_w p|^2)
    Xi = np.log(2 * Vz * K_J * R_w)
    Pw = P_w if P_w is not None else 4 * s2 * E['sww_w'] * E['phi_svv'] / a ** 2
    j = s2 * E['jpos']; yos = sig * Jww / j; delta = j / a - 1
    out = {'sigma': sig, 'a': a, 'y_over_sigma': yos, 'delta': delta, 'den_lower_y': 1 / (1 + sig * (1 + delta) / 2), 'J_ww': Jww, 'P_w': Pw, 'P_sharp': Psharp, 'P_y': Py, 'eps_m': float(eps_m), 'Lambda': Lam, 'xbar_w': xw, 'xbar_w_grid': xw_grid,
           'K_Lambda': K_Lam, 'K_J': K_J, 'R_w': R_w, 'K_rho': K_rho, 'V_z': Vz, 'Xi': Xi, 'jensen_rhs': jensen_rhs,
           'L2_rel_grid_vs_closed': max(abs(L['p2'] / I0 - 1), abs(L['pw2'] / Iww - 1), abs(L['pvw2'] / Ivvww - 1)), 'tail_mass_mz_gt_pi': E['tail'],
           # the steps (left side >= right side)
           'S1': [Pw, 2 / np.pi ** 2 * Psharp], 'S2': [np.sqrt(Psharp), np.sqrt(Py) * (1 - eps_m)], 'S3': [Py, 1 / Lam], 'S4_KLam': K_Lam,
           'S5': [jensen_rhs, xw], 'S6': [Jww, 1 / (2 * Vz)], 'S8': [K_rho, np.e * s2 * R_w], 'S56': [Xi, xw]}
    sq = np.sqrt(2) / np.pi * max(1 - eps_m, 0) / np.sqrt(4 * K_Lam * Xi)
    out['sqrtPw_bound_state'] = sq                                  # sqrt(P_w) >= this (steps 1-8, state constants)
    Kx = max(K_Lam, K_J, K_rho); C = 0.5 * np.log(2 * Vz * K_J * K_rho / np.e)
    out['K_max'] = Kx; out['C_state'] = C
    out['sqrtPw_bound_uniform'] = max(1 - eps_m, 0) / (2 * np.pi * np.sqrt(K_Lam)) / np.sqrt(np.log(1 / sig) + C)
    if gap is not None:
        out['gap_over_sigma'] = gap; out['den_factor'] = den_factor; out['eps'] = eps
        f = den_factor * (1 - max(eps, 0.0) if eps is not None else 1.0)
        out['bound_state'] = f * sq; out['bound_uniform'] = f * out['sqrtPw_bound_uniform']
        out['S0'] = [gap, den_factor * np.sqrt(Pw) * (1 - (eps if eps is not None else 0.0))]
    return out
