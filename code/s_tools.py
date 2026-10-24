"""Stage S (paper E): the weight-free score moment
    P# = sigma^2 E[u^2 s_v^2] J_ww / (sigma^2 J_vv)^2        (u = sqrt2 (m_q - pi), m = y + sigma^2 s; pendulum, V/W frame)
and the line reduction used to bound it from below.

Notation.  p = rho * N(0, sigma^2 I) on the (v, w) grid of i_tools.grid (axis 0 = V, axis 1 = W).  A "line" is {y : y_w = w}.
  * Conditional along a line: p(v | w) is the sigma-blur of a 1-d measure, so I_v(w) := E[s_v^2 | w] in (0, 1/sigma^2]
    (the 1-d Fisher information of the line), and s_v = d_v log p(v | w) exactly.
  * omega(w) := sigma^2 I_v(w) in (0, 1]  (the v-sharpness of the line);  E_w[omega] = sigma^2 J_vv =: a.
  * ubar(w) := E[u s_v^2 | w] / I_v(w)  (s_v^2-weighted line mean of u),  U(w) := E[u^2 s_v^2 | w] / I_v(w) >= ubar^2 (Jensen).
  * s_W(w) := E[s_w | w] = d_w log p_W(w) (score of the w-marginal), I_W := E[s_W^2],  I_F := J_ww - I_W = E[(s_w - s_W)^2] >= 0.
Chain (each step rigorous; the last one needs an assumption, see below):
  (L)  A := sigma^2 E[u^2 s_v^2] = E_w[omega U] >= E_w[omega ubar^2]                                    (exact + Jensen)
  (W)  E_w[omega ubar^2] I_W >= ( E_w[omega ubar'] + E_w[omega' ubar] )^2                               (Cauchy-Schwarz, omega <= 1,
                                                                                                          integration by parts in w)
  (S)  |E_w[omega' ubar]| <= sqrt(E_w[omega'^2/omega]) sqrt(E_w[omega ubar^2])                          (Cauchy-Schwarz)
  =>   P# = A J_ww / a^2 >= (X/a)^2 / (1 + beta)^2,   X := E_w[omega ubar'],  beta^2 := E_w[omega'^2/omega] / I_W
  (D)  derivative of the line information (proved by integration by parts along the line):
         d_w I_v(w) = E[ s_w^c (2 H_vv - s_v^2) | w ],  s_w^c := s_w - s_W(w)
       hence |omega'| <= omega sqrt(Gamma(w) I_F(w)),  Gamma(w) := Var(2 H_vv - s_v^2 | w) / I_v(w)^2  (= 2 for a Gaussian line),
       and beta^2 <= E_w[omega Gamma I_F(w)] / I_W.
"""
import numpy as np
from i_tools import grid, V, W


def fields(ws, mus, Ss, sig, per_sd=6, max_points=2_000_000):
    Y, dA, n = grid(ws, mus, Ss, sig, per_sd=per_sd, max_points=int(min(max_points, 2.0e7 / len(ws)))); s2 = sig * sig
    T = [S + s2 * np.eye(2) for S in Ss]; Ti = [np.linalg.inv(t) for t in T]
    lg = [np.log(w) - 0.5 * np.log(np.linalg.det(t)) - np.log(2 * np.pi) for w, t in zip(ws, T)]
    logc = np.stack([l - 0.5 * np.einsum('ni,ij,nj->n', Y - m, ti, Y - m) for l, m, ti in zip(lg, mus, Ti)], 1)
    mx = logc.max(1); e = np.exp(logc - mx[:, None]); p = e.sum(1) * np.exp(mx); pi = e / e.sum(1, keepdims=True); del e, logc
    uk = np.stack([-(Y - m) @ ti for m, ti in zip(mus, Ti)], 1); s = np.einsum('nk,nkd->nd', pi, uk)
    H = np.einsum('nk,kij->nij', pi, np.array(Ti)) - (np.einsum('nk,nki,nkj->nij', pi, uk, uk) - np.einsum('ni,nj->nij', s, s)); del uk
    pw = p * dA; mass = float(pw.sum()); pw = pw / pw.sum(); m = Y + s2 * s
    R = np.stack([V, W], 1); Hf = np.einsum('ia,nij,jb->nab', R, H, R)
    G = Y.reshape(n + (2,)); h = np.array([(G[1, 0] - G[0, 0]) @ V, (G[0, 1] - G[0, 0]) @ W])
    f = {'n': n, 'h': h, 'sig': sig, 'mass': mass, 'pw': pw.reshape(n), 'sv': (s @ V).reshape(n), 'sw': (s @ W).reshape(n),
         'Hvv': Hf[:, 0, 0].reshape(n), 'Hvw': Hf[:, 0, 1].reshape(n), 'Hww': Hf[:, 1, 1].reshape(n),
         'u': (np.sqrt(2) * (m[:, 0] - np.pi)).reshape(n), 'wcol': G[0, :] @ W - np.array([np.pi, 0.0]) @ W}   # stage T: w - w_top of each column
    return f


def chain(f):
    s2 = f['sig'] ** 2; P = f['pw']; sv, sw, u, Hvv = f['sv'], f['sw'], f['u'], f['Hvv']; h1 = f['h'][1]
    E = lambda a: float(np.sum(P * a))
    Jvv, Jww, Jvw = E(sv * sv), E(sw * sw), E(sv * sw); a = s2 * Jvv
    A = s2 * E(u * u * sv * sv); Psharp = A * Jww / a ** 2
    Pc = P.sum(0); ok = Pc > 1e-300 * Pc.max(); Pc_ = np.where(ok, Pc, 1.0)
    L = lambda g: np.sum(P * g, 0) / Pc_                     # line (fixed w) conditional mean, function of the column
    Iv = L(sv * sv); om = s2 * Iv; ub = L(u * sv * sv) / np.where(Iv > 0, Iv, 1.0); Uw = L(u * u * sv * sv) / np.where(Iv > 0, Iv, 1.0)
    sW = L(sw); IFl = L((sw - sW[None, :]) ** 2); Ew = lambda g: float(np.sum(Pc * g))
    IW = Ew(sW * sW); IF = Jww - IW
    dub = np.gradient(ub, h1); dom = np.gradient(om, h1)
    var_l = L((2 * Hvv - sv * sv) ** 2) - L(2 * Hvv - sv * sv) ** 2; Gam = var_l / np.where(Iv > 0, Iv, 1.0) ** 2
    dIv_pred = L((sw - sW[None, :]) * (2 * Hvv - sv * sv))                 # (D): d_w I_v
    Aj = Ew(om * ub * ub); X = Ew(om * dub); Ysl = Ew(dom * ub); Wd = Ew(np.where(om > 0, dom ** 2 / np.where(om > 0, om, 1.0), 0.0))
    beta = np.sqrt(Wd / IW) if IW > 0 else np.inf
    Gam_eff = Wd / IF if IF > 0 else 0.0
    # S2: the two correction factors of stage R4 in line form
    eta = s2 * Hvv; T1 = A; zeta = s2 * sv * sv - eta                     # E[zeta | line] = 0 exactly (integration by parts along the line)
    covl = L(u * u * zeta) - L(u * u) * L(zeta); T2_line = Ew(covl)
    sdu2 = np.sqrt(np.maximum(L(u ** 4) - L(u * u) ** 2, 0)); sdz = np.sqrt(np.maximum(L(zeta * zeta) - L(zeta) ** 2, 0))
    phi = 2 * eta - s2 * sv * sv; varphi = np.maximum(L(phi * phi) - L(phi) ** 2, 0)
    s2stuff = {'T2_over_T1': (s2 * E(u * u * sv * sv) - s2 * E(u * u * Hvv)) / T1, 'T2_over_T1_line': T2_line / T1, 'T2_bound_line': Ew(sdu2 * sdz) / T1,
               'line_mean_zeta_max': float(np.max(np.abs(L(zeta)) * (Pc > 1e-9 * Pc.max()))),
               'delta': E(np.maximum(-eta, 0)) / a, 'delta_bound_disp': Ew(np.sqrt(varphi)) / (2 * a), 'line_mean_phi_minus_omega': float(np.max(np.abs(L(phi) - om) * (Pc > 1e-9 * Pc.max())))}
    # simple candidates suggested in the instruction (all are rigorous lower bounds of P#; the question is how strong they are)
    dvu = 1 - s2 * (Hvv + f['Hvw'])                              # d_v u = c_vv + c_wv
    cand = {'cand_CS_naive': s2 * E(u * sv) ** 2 * Jww / a ** 2,                                   # E[u^2 s_v^2] >= (E[u s_v])^2
            'cand_g_sw': s2 * E(u * sv * sw) ** 2 / a ** 2,                                          # g = s_w   (E[g^2] = J_ww)
            'cand_g_dvu': s2 * E(u * sv * dvu) ** 2 * Jww / (E(dvu * dvu) * a ** 2),                 # g = d_v u
            'cand_g_uHvv': s2 * E(u * u * sv * Hvv) ** 2 * Jww / (E(u * u * Hvv * Hvv) * a ** 2),    # g = u H_vv
            }
    # uncertainty lemma with (e, f) = (q, V):  E[u^2] J_vv >= (1 - sigma^2 (J_vv + J_vw))^2   (bounds E[u^2], not E[u^2 s_v^2])
    cand['lemma_u_V_rhs_over_lhs'] = (1 - a - s2 * Jvw) ** 2 / (E(u * u) * Jvv)
    # curvature bound route: E[s_v^2 - H_vv | line] = 0 exactly; pointwise s_v^2 - H_vv = d_v^2 p / p
    cand['curv_E_dv2p_over_p_rel'] = abs(E(sv * sv - Hvv)) / Jvv
    cvw = -s2 * f['Hvw']; cww = 1 - s2 * f['Hww']
    x_main = s2 * E(sv * sv * (cvw + cww)) / a                  # the 'main term' of X/a: s_v^2-weighted mean slope d_w u = c_vw + c_ww
    bound_sharp = (max(X, 0) ** 2 * Jww / (a ** 2 * (np.sqrt(IW) + np.sqrt(Wd)) ** 2)) if IW > 0 else 0.0
    return {**cand, **s2stuff, 'Wd': Wd, 'Gamma_eff': Gam_eff, 'x_main': x_main, 'bound_sharp': bound_sharp, 'bound_G': (max(X, 0) / a) ** 2 / (1 + Gam_eff),
            'omGam_sup': float(np.max(np.where(Pc > 1e-6 * Pc.max(), om * Gam, 0))), 'omGam_mean_IF': Ew(om * Gam * IFl) / IF if IF > 0 else 0.0,
            'sigma': f['sig'], 'a': a, 'Jvv': Jvv, 'Jww': Jww, 'Jvw': Jvw, 'A': A, 'P_sharp': Psharp, 'A_line': Ew(om * Uw),
            'A_jensen': Aj, 'I_W': IW, 'I_F': IF, 'Jww_over_IW': Jww / IW, 'X_slope': X, 'Y_slope': Ysl, 'X_over_a': X / a, 'Y_over_a': Ysl / a,
            'CR_lhs': Aj * IW, 'CR_rhs': (X + Ysl) ** 2, 'bound_W': (X + Ysl) ** 2 / a ** 2, 'beta': beta,
            'bound_final': (max(X, 0) / a) ** 2 / (1 + beta) ** 2, 'E_om_Gam_IF': Ew(om * Gam * IFl), 'beta2_via_D': Ew(om * Gam * IFl) / IW,
            'Gamma_max_weighted': float(np.max(np.where(Pc > 1e-6 * Pc.max(), Gam, 0))), 'Gamma_mean': Ew(Gam),
            'dIv_check': float(np.sum(Pc * np.abs(np.gradient(Iv, h1) - dIv_pred)) / max(np.sum(Pc * np.abs(dIv_pred)), 1e-3 * Ew(Iv) * np.sqrt(IW), 1e-300)),
            'ibp_check': (Ew(om * ub * sW) + Ew(om * dub + dom * ub)) / max(abs(Ew(om * dub + dom * ub)), 1e-300),
            'Z_w': E(np.clip(s2 * np.maximum(Hvv, 0), 0, 1) * u * u), 'j': E(np.clip(s2 * np.maximum(Hvv, 0), 0, 1)), 'Eu2': E(u * u),
            'edge_mass_w': float(Pc[0] + Pc[-1])}


def all_terms(ws, mus, Ss, sig, per_sd=6):
    f = fields(ws, mus, Ss, sig, per_sd); c = chain(f); c['y_over_sigma'] = sig * c['Jww'] / c['j']; c['P'] = c['Z_w'] * c['Jww'] / c['j'] ** 2
    c['d'] = float(np.sqrt(c['Z_w'] / (c['j'] * sig))); c['n_grid'] = int(np.prod(f['n'])); c['grid_mass'] = f['mass']; return c
