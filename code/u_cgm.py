"""Stage U: exact evaluation for 'complex Gaussian mixtures' (Gaussians times plane waves), e.g. a faint V-ripple on a wide background.
A term is (log_coef, m, S): exp(log_coef) * N(x; m, S) with complex log_coef (kept as a log so that huge/tiny factors never overflow) and complex mean m (in (v, w) coordinates relative to the top (pi, 0)).
  Gaussian x plane wave:  N(x; c, S) e^{i kap.(x - c)} = e^{-kap'S kap/2} N(x; c + i S kap, S).
Blur:      p(y) = sum_t coef_t N(y; m_t, T_t),  T_t = S_t + sigma^2 I            (also for complex m; real part taken at the end)
Posterior: coef_t N(x; m_t, S_t) N(y - x; 0, sigma^2 I) = coef_t N(y; m_t, T_t) N(x; m'_t, C_t),
           m'_t = m_t + S_t T_t^{-1} (y - m_t),  C_t = S_t - S_t T_t^{-1} S_t,   so E[g(x) | y] = Re sum_t alpha_t(y) E_{N(m'_t, C_t)}[g] / p(y)
with E[x_p] = (m'_v - m'_w)/sqrt2 and E[-sin x_q] = E[sin z] = sin(E z) exp(-Var z / 2), z = x_q - pi = (v + w)/sqrt2 (analytic in E z).
R = E[s . E[f | y]] / (sigma^2 E|s|^2),  f = (p, -sin q)  (pendulum, kappa = 1),  exactly as in l_tools.diag / j2_smooth.arrays;
the reported R / gap use the integration-by-parts-stable form (see evaluate); the direct form is kept as R_direct.
Grid: tensor grid in (v, w), possibly NON-uniform (fine where the structure is, coarse on the smooth background), trapezoid weights.
Fields are returned in the s_tools.fields format (with 'h'[1] = the w nodes, accepted by np.gradient) so that t_terms / chain apply.
Python 3.11 compatible."""
import numpy as np

SQ2 = np.sqrt(2.0)


def nodes(fine_half, h_fine, coarse_half, h_coarse):
    """symmetric nodes: spacing h_fine on [-fine_half, fine_half], then h_coarse out to +-coarse_half."""
    f = np.arange(0, fine_half + 1e-12 * h_fine, h_fine)
    c = np.arange(f[-1] + h_coarse, coarse_half + 1e-12 * h_coarse, h_coarse) if coarse_half > f[-1] else np.zeros(0)
    pos = np.concatenate([f, c]); return np.concatenate([-pos[:0:-1], pos])


def trap_weights(x):
    w = np.zeros_like(x); d = np.diff(x); w[1:] += d / 2; w[:-1] += d / 2; return w


def evaluate(terms, sig, vn, wn, chunk=200_000):
    s2 = sig * sig; Vv, Ww = np.meshgrid(vn, wn, indexing='ij'); n = Vv.shape
    X = np.stack([Vv.ravel(), Ww.ravel()], 1); N = len(X)
    p = np.zeros(N); g = np.zeros((N, 2)); Hs = np.zeros((N, 2, 2)); Exp = np.zeros(N); Esz = np.zeros(N)
    pre = []
    for lc, m, S in terms:
        T = S + s2 * np.eye(2); Ti = np.linalg.inv(T); K = np.eye(2) - s2 * Ti; C = s2 * (np.eye(2) - s2 * Ti)   # = S T^-1 and S - S T^-1 S, cancellation-free (S >> sigma^2)
        varz = (C[0, 0] + 2 * C[0, 1] + C[1, 1]) / 2
        pre.append((lc - np.log(2 * np.pi * np.sqrt(np.linalg.det(T))), np.asarray(m, complex), Ti, K, varz))
    for a in range(0, N, chunk):
        x = X[a:a + chunk]
        P_ = np.zeros(len(x), complex); G_ = np.zeros((len(x), 2), complex); H_ = np.zeros((len(x), 2, 2), complex)
        Ep = np.zeros(len(x), complex); Ez = np.zeros(len(x), complex)
        for c0, m, Ti, K, varz in pre:
            d = x - m; Tid = d @ Ti.T; Nt = np.exp(c0 - 0.5 * np.einsum('ni,ni->n', d, Tid))
            P_ += Nt; G_ += -Tid * Nt[:, None]; H_ += (np.einsum('ni,nj->nij', Tid, Tid) - Ti[None]) * Nt[:, None, None]
            mp = m + d @ K.T; Ep += Nt * (mp[:, 0] - mp[:, 1]) / SQ2; Ez += Nt * np.sin((mp[:, 0] + mp[:, 1]) / SQ2) * np.exp(-varz / 2)
        p[a:a + chunk] = P_.real; g[a:a + chunk] = G_.real; Hs[a:a + chunk] = H_.real; Exp[a:a + chunk] = Ep.real; Esz[a:a + chunk] = Ez.real
    s = g / p[:, None]; H = -(Hs / p[:, None, None] - np.einsum('ni,nj->nij', s, s)); Exp /= p; Esz /= p
    dA = np.outer(trap_weights(vn), trap_weights(wn)).ravel(); pw = p * dA; mass = float(pw.sum()); pw = pw / pw.sum()
    sv, sw = s[:, 0], s[:, 1]; sq, sp = (sv + sw) / SQ2, (sv - sw) / SQ2
    R = float(np.sum(pw * (sq * Exp + sp * Esz)) / (s2 * np.sum(pw * (sv * sv + sw * sw))))
    # stable form (stage V): with m = y + sigma^2 s and z = x_q - pi, s.E[f|y] = s_v m_v - s_w m_w - s_p E[z - sin z | y]; the two O(1) parts
    # E[s_v v] = E[s_w w] = -1 cancel exactly by integration by parts, so E[s.Ef] = sigma^2 (J_vv - J_ww) - E[s_p (m_z - E[sin z|y])].
    # Needed on NON-uniform grids, where the trapezoid rule does not keep the integration-by-parts identity (the direct R loses ~1e-3 there).
    mz = ((Vv.ravel() + s2 * sv) + (Ww.ravel() + s2 * sw)) / SQ2
    R_stable = float((s2 * np.sum(pw * (sv * sv - sw * sw)) - np.sum(pw * sp * (mz - Esz))) / (s2 * np.sum(pw * (sv * sv + sw * sw))))
    u = (Vv.ravel() + Ww.ravel()) + s2 * (sv + sw)
    Hvv = H[:, 0, 0]; wfoc = np.clip(s2 * np.maximum(Hvv, 0), 0, 1)
    j = float(np.sum(pw * wfoc)); Zw = float(np.sum(pw * wfoc * u * u)); Jww = float(np.sum(pw * sw * sw)); Jvv = float(np.sum(pw * sv * sv))
    f = {'n': n, 'h': np.array([vn[1] - vn[0], 0.0]), 'sig': sig, 'mass': mass, 'pw': pw.reshape(n), 'sv': sv.reshape(n), 'sw': sw.reshape(n),
         'Hvv': Hvv.reshape(n), 'Hvw': H[:, 0, 1].reshape(n), 'Hww': H[:, 1, 1].reshape(n), 'u': u.reshape(n), 'wcol': wn}
    f['h'] = [vn[1] - vn[0], wn]          # np.gradient(., h[1]) uses the (non-uniform) w nodes
    diag = {'R': R_stable, 'R_direct': R, 'gap_over_sigma': (1 - R_stable) / sig, 'gap_over_sigma_direct': (1 - R) / sig, 'j': j, 'Z_w': Zw, 'Jww': Jww, 'Jvv': Jvv, 'y_over_sigma': sig * Jww / j,
            'd': float(np.sqrt(Zw / (j * sig))), 'P': Zw * Jww / j ** 2, 'grid_mass': mass, 'p_min': float(p.min()), 'n_points': int(N)}
    diag['F'] = diag['d'] ** 2 / 8 + 2 * diag['y_over_sigma']
    return f, diag


def mixture_terms(ws, mus, Ss):
    """real Gaussian mixture given in (q, p) coordinates -> terms in (v, w) coordinates relative to the top."""
    Rm = np.array([[1, 1], [1, -1]]) / SQ2           # rows V, W
    top = np.array([np.pi, 0.0])
    return [(complex(np.log(w_)), Rm @ (np.asarray(m) - top), Rm @ np.asarray(S) @ Rm.T) for w_, m, S in zip(ws, mus, Ss)]


def ripple_terms(sig, Lv, Lw, c, k0, Rv, ell, q0=0.0):
    """rho = N(0, diag(Lv^2, Lw^2)) [1 + c exp(-v^2/(2 Rv^2) - w^2/(2 ell^2)) cos(k v) cos(q w)],  k = k0/sig, q = q0/sig  (absolute lengths).
    rho >= 0 for c <= 1.  The window factor times the background is again a Gaussian (covariance S2)."""
    Sb = np.diag([Lv ** 2, Lw ** 2]); S2 = np.linalg.inv(np.linalg.inv(Sb) + np.diag([1 / Rv ** 2, 1 / ell ** 2]))
    amp = c * np.sqrt(np.linalg.det(S2) / np.linalg.det(Sb)); k, q = k0 / sig, q0 / sig
    terms = [(0j, np.zeros(2, complex), Sb)]
    waves = [(a, 0.0) for a in (1, -1)] if q0 == 0 else [(a, b) for a in (1, -1) for b in (1, -1)]
    share = 0.5 if q0 == 0 else 0.25
    for a, b in waves:
        kap = np.array([a * k, b * q]); terms.append((np.log(amp * share) - 0.5 * kap @ S2 @ kap + 0j, 1j * (S2 @ kap), S2))
    return terms
