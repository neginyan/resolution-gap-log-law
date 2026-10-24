"""Stage U1: the pieces of a I(p~) and of Lambda (notation of t_tools; omega' from the exact identity omega' = Cov(s_w, phi | line)).
(i)   a I(p~) = E_w[omega s_W^2] + 2 E_w[s_W omega'] + W_d,   W_d = E_w[omega'^2/omega]           (identity: (p_W omega)'/p_W = s_W omega + omega')
      Minkowski + omega <= 1:      sqrt(a I(p~)) <= sqrt(E_w[omega s_W^2]) + sqrt(W_d) <= sqrt(I_W) + sqrt(W_d)
      Cauchy-Schwarz on the line:  omega'^2 <= I_F(w) Var(phi | w) = I_F(w) omega D(w),   D := Var(phi|line)/omega
      =>  W_d <= E_w[I_F(w) D(w)] =: Dbar I_F   and   a I(p~) <= (sqrt(I_W) + sqrt(Dbar I_F))^2 <= (1 + Dbar) J_ww          [proved]
      Gaussian lines: D = 2 omega <= 2, so c0 <= 3 for every state whose lines are Gaussian.
(ii)  Lambda J_ww = sigma^2 E[(s_v s_w - 2 H_vw)^2]  =>  sqrt(Lambda) <= sqrt(Lambda0) + 2 sqrt(h),
      Lambda0 := sigma^2 E[s_v^2 s_w^2] / J_ww (= a for product states),  h := sigma^2 E[H_vw^2] / J_ww (= 0 for product states)      [proved]
Python 3.11 compatible."""
import numpy as np


def u_terms(f):
    s2 = f['sig'] ** 2; P = f['pw']; sv, sw, Hvv, Hvw = f['sv'], f['sw'], f['Hvv'], f['Hvw']
    Pc = P.sum(0); keep = Pc > 1e-300 * max(Pc.max(), 1e-300); Pc_ = np.where(keep, Pc, 1.0)
    L = lambda g: np.sum(P * g, 0) / Pc_; Ew = lambda g: float(np.sum(Pc * g)); E = lambda g: float(np.sum(P * g))
    om = s2 * L(sv * sv); phi = s2 * (2 * Hvv - sv * sv); Vl = np.maximum(L(phi * phi) - L(phi) ** 2, 0)
    sW = L(sw); IFl = L((sw - sW[None, :]) ** 2); Jww = E(sw * sw); IW = Ew(sW * sW); IF = Jww - IW; a = Ew(om)
    domi = s2 * L((sw - sW[None, :]) * (2 * Hvv - sv * sv))                         # omega' (identity)
    pos = om > 0; omp = np.where(pos, om, 1.0)
    t1 = Ew(om * sW * sW); tx = 2 * Ew(sW * domi); Wd = Ew(np.where(pos, domi ** 2 / omp, 0.0))
    g = sv * sw - 2 * Hvw; dlog = s2 * L(sv * g); aIpt = Ew(np.where(pos, dlog ** 2 / omp, 0.0))
    D = np.where(pos, Vl / omp, 0.0); Dbar = Ew(IFl * D) / IF if IF > 0 else 0.0
    Lam = s2 * E(g * g) / Jww; Lam0 = s2 * E(sv * sv * sw * sw) / Jww; h = s2 * E(Hvw * Hvw) / Jww
    lines_sig = Pc > 1e-6 * Pc.max()
    return {'aIpt_over_Jww': aIpt / Jww, 'term_omega_sW2': t1 / Jww, 'term_cross': tx / Jww, 'term_Wd': Wd / Jww,
            'decomp_residual': (t1 + tx + Wd - aIpt) / max(aIpt, 1e-300),
            'I_W_over_Jww': IW / Jww, 'I_F_over_Jww': IF / Jww, 'Wd_over_IF': Wd / IF if IF > 0 else 0.0, 'Dbar': Dbar,
            'bound_minkowski': (np.sqrt(t1) + np.sqrt(Wd)) ** 2 / Jww, 'bound_D': (np.sqrt(IW) + np.sqrt(Dbar * IF)) ** 2 / Jww, 'bound_1pD': 1 + Dbar,
            'D_max_line': float(np.max(np.where(lines_sig, D, 0))), 'Lambda': Lam, 'Lambda0': Lam0, 'h_Hvw': h,
            'bound_Lambda_coupling': (np.sqrt(Lam0) + 2 * np.sqrt(h)) ** 2, 'a': a}


def x_terms(f):
    """U2.  r(v, w) := sigma^2 p s_v^2 / a (the 2-d weight whose w-marginal is p~), g := u - (w - w_top) = v + sigma^2 (s_v + s_w) (relative),
    mu(w) = E_r[g | w].  Exact (differentiate the r-conditional mean; d_w g = -sigma^2 (H_vw + H_ww), d_w log r = s_w - 2 H_vw / s_v):
        x - 1 = E_p~[mu'] = -sigma^2 E_r[H_vw + H_ww] + sigma^2 E[(g - mu)(s_v^2 s_w - 2 H_vw s_v)] / a                       [proved]
    and, Cauchy-Schwarz with E_r[(d_w log r)^2] = Lambda J_ww / a,
        |x - 1| <= sigma^2 |E_r[H_vw + H_ww]| + sqrt(S_line Lambda J_ww) / a,   S_line := sigma^2 E[s_v^2 (g - mu)^2]           [proved]
    (no integration by parts against p~, so separated clusters with different line centres cost nothing).  With nu := S_line/(sigma^2 a)
    (= 3 for Gaussian lines) and J_ww <= j/(2 sigma) (y/sigma <= 0.5):  second term <= sqrt(sigma) sqrt(nu Lambda j / (2 a)) = O(sqrt(sigma))."""
    s2 = f['sig'] ** 2; P = f['pw']; sv, sw, Hvv, Hvw, Hww = f['sv'], f['sw'], f['Hvv'], f['Hvw'], f['Hww']
    Pc = P.sum(0); Pc_ = np.where(Pc > 0, Pc, 1.0); L = lambda q: np.sum(P * q, 0) / Pc_; E = lambda q: float(np.sum(P * q))
    a = s2 * E(sv * sv); Jww = E(sw * sw); g = f['u'] - f['wcol'][None, :]
    Iv = L(sv * sv); mu = L(g * sv * sv) / np.where(Iv > 0, Iv, 1.0)
    gc = g - mu[None, :]
    t1 = -s2 * s2 * E(sv * sv * (Hvw + Hww)) / a
    t2 = s2 * E(gc * (sv * sv * sw - 2 * Hvw * sv)) / a
    S = s2 * E(sv * sv * gc * gc); Lam = s2 * E((sv * sw - 2 * Hvw) ** 2) / Jww
    j = E(np.clip(s2 * np.maximum(Hvv, 0), 0, 1))
    om = s2 * Iv; pt = Pc * om / a; dmu = np.gradient(mu, f['h'][1])
    return {'x_minus_1_identity': t1 + t2, 'x_minus_1_fd': float(np.sum(pt * dmu)), 'x_term_curv': t1, 'x_term_cov': t2,
            'x_bound_new': abs(t1) + np.sqrt(S * Lam * Jww) / a, 'nu_line': S / (s2 * a), 'j_over_a': j / a,
            'x_bound_new_over_sqrt_sigma': (abs(t1) + np.sqrt(S * Lam * Jww) / a) / np.sqrt(f['sig']),
            'sd_ptilde_mu_over_sigma': float(np.sqrt(max(np.sum(pt * mu * mu) - np.sum(pt * mu) ** 2, 0))) / f['sig']}
