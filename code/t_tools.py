"""Stage T: the slope term Y = E_w[omega' ubar] of the stage-S chain, bounded with the ABSOLUTE line variance.
Notation as in s_tools (lines y_w = const, omega = sigma^2 E[s_v^2 | line], ubar, I_W, I_F(w), X, Y).
On a line, phi := sigma^2 (2 H_vv - s_v^2), E[phi | line] = omega, V(w) := Var(phi | line).
  (D')  |omega'(w)| <= sqrt(I_F(w) V(w))                         (stage S identity + Cauchy-Schwarz; proved)
  (A)   V(w) <= 12 for every line                                   (main line: phi in [2 - 2E[Z^2|v], 2 - E[Z^2|v]], Jensen, E Z^4 = 3)
Then |Y| <= E_w[sqrt(I_F(w) V(w)) |ubar|] <= sqrt(I_F) sqrt(R),  R := E_w[V ubar^2]   (Cauchy-Schwarz over lines; proved),
and with Q = E_w[omega ubar^2] the stage-S argument gives, for K_V := R / Q,
  (T)   P# >= X^2 J_ww / (a^2 (sqrt I_W + sqrt(K_V I_F))^2) >= x^2 / (1 + K_V)          (proved; K_V is a property of the state)
K_V is a ubar^2 omega-weighted average of the line dispersion D = V/omega.  Candidates for a uniform bound of K_V are evaluated:
  K_abs  := C_V E_w[ubar^2] / Q            (V <= C_V absolute; C_V = 12 proved, ~4.4 best found)
  K_log  := C_L E_w[omega log(e/omega) ubar^2] / Q   (conjectured V <= C_L omega log(e/omega), C_L ~ 5.6 from the comb family)
  K_split(omega*) := [E_w[V ubar^2 1(omega >= omega*)] + C_V E_w[ubar^2 1(omega < omega*)]] / Q   (is an upper bound of K_V)
"""
import numpy as np
from s_tools import fields, chain

C_V_PROVED = 12.0


def t_terms(f, C_V=C_V_PROVED, C_L=5.6, om_star=(0.05, 0.1, 0.2)):
    s2 = f['sig'] ** 2; P = f['pw']; sv, sw, u, Hvv = f['sv'], f['sw'], f['u'], f['Hvv']; h1 = f['h'][1]
    Pc = P.sum(0); keep = Pc > 1e-300 * max(Pc.max(), 1e-300); Pc_ = np.where(keep, Pc, 1.0)
    L = lambda g: np.sum(P * g, 0) / Pc_; Ew = lambda g: float(np.sum(Pc * g))
    E = lambda g: float(np.sum(P * g))
    Iv = L(sv * sv); om = s2 * Iv; ub = L(u * sv * sv) / np.where(Iv > 0, Iv, 1.0)
    phi = s2 * (2 * Hvv - sv * sv); V = np.maximum(L(phi * phi) - L(phi) ** 2, 0)
    sW = L(sw); IFl = L((sw - sW[None, :]) ** 2); IW = Ew(sW * sW); Jww = E(sw * sw); IF = Jww - IW; a = Ew(om)
    dub = np.gradient(ub, h1); dom = np.gradient(om, h1); X = Ew(om * dub); Y = Ew(dom * ub)
    dom_id = s2 * L((sw - sW[None, :]) * (2 * Hvv - sv * sv))          # omega' from the exact identity (no finite difference)
    Dl = V / np.where(om > 0, om, 1.0)
    Q = Ew(om * ub * ub); R = Ew(V * ub * ub); KV = R / Q
    out = {'a': a, 'Q': Q, 'R': R, 'K_V': KV, 'X_over_a': X / a, 'Y_over_a': Y / a, 'I_W': IW, 'I_F': IF, 'Jww': Jww,
           'V_max_line': float(np.max(np.where(Pc > 1e-9 * Pc.max(), V, 0))),
           'D_omega2_excess': Ew(np.maximum(V - 2 * om * om, 0) * ub * ub) / Q,
           'Y_bound_abs': np.sqrt(IF * R) / a,
           'dprime_ratio_max': float(np.max(np.where((Pc > 1e-9 * Pc.max()) & (IFl * V > 0), np.abs(dom_id) / np.sqrt(np.maximum(IFl * V, 1e-300)), 0))),
           'identity_vs_fd': float(np.sum(Pc * np.abs(dom - dom_id)) / max(np.sum(Pc * np.abs(dom_id)), 1e-3 * a * np.sqrt(max(IW, 0)), 1e-300)),
           'D_max_weighted': float(np.max(np.where(om * ub * ub * Pc > 1e-6 * np.max(om * ub * ub * Pc), Dl, 0))),
           'bound_T': (max(X, 0) ** 2 * Jww / (a ** 2 * (np.sqrt(IW) + np.sqrt(KV * IF)) ** 2)) if IW > 0 else 0.0,
           'bound_T_simple': (max(X, 0) / a) ** 2 / (1 + KV),
           'K_abs_12': C_V * Ew(ub * ub) / Q, 'K_abs_best': 4.42 * Ew(ub * ub) / Q,
           'K_log': C_L * Ew(om * np.log(np.e / np.maximum(om, 1e-300)) * ub * ub) / Q,
           'log_weight_mean': Ew(om * np.log(np.e / np.maximum(om, 1e-300)) * ub * ub) / Q}
    for D0 in (2.5, 3, 4, 6):
        bad = Dl > D0
        out[f'M_D{D0:g}'] = Ew(ub * ub * bad) / Q                           # u^2 mass of strongly non-Gaussian lines, relative to Q
        out[f'K_Dsplit_{D0:g}'] = (Ew(V * ub * ub * (~bad)) + Ew(V * ub * ub * bad)) / Q   # = K_V (bookkeeping)
        out[f'K_Dbound_{D0:g}'] = D0 + C_V * out[f'M_D{D0:g}']            # K_V <= D0 + C_V M(D0)
    for w_ in om_star:
        big = om >= w_
        out[f'K_split_{w_:g}'] = (Ew(V * ub * ub * big) + C_V * Ew(ub * ub * (~big))) / Q
    # T2: x - 1 = E_w[omega mu'] / a,  mu(w) := ubar(w) - (w - w_top)  (s_v^2-weighted line centre in v, plus sigma^2 (s_v + s_w))
    mu = ub - f['wcol']; dmu = np.gradient(mu, h1)
    out['x_minus_1'] = Ew(om * dmu) / a; out['x_minus_1_check'] = X / a - 1
    out['mu_rms_om_over_sigma'] = np.sqrt(Ew(om * mu * mu) / a) / f['sig']; out['mu_rms_V_over_sigma'] = np.sqrt(Ew(V * mu * mu) / a) / f['sig']
    # rigorous: |E_w[omega mu']| <= sqrt(E_w[omega mu^2] I_W) + sqrt(I_F E_w[V mu^2])
    out['x_bound'] = (np.sqrt(Ew(om * mu * mu) * IW) + np.sqrt(IF * Ew(V * mu * mu))) / a
    out['x_bound_over_sqrt_sigma'] = out['x_bound'] / np.sqrt(f['sig'])
    out['P_sharp'] = s2 * E(u * u * sv * sv) * Jww / a ** 2
    # ---- the weighted-marginal route (no division by omega): p~(w) := p_W(w) omega(w) / a.
    #   (p_W omega)'(w) = sigma^2 p_W E[s_v (s_v s_w - 2 H_vw) | w]          (d_w of int p s_v^2 dv; d_w s_v = -H_vw)     [proved]
    #   Cramer-Rao under p~:  E_p~[ubar^2] I(p~) >= (E_p~[ubar'])^2 = x^2     =>   P# >= Q J_ww / a^2 >= x^2 J_ww / (a I(p~))  [proved]
    #   Cauchy-Schwarz on each line:  a I(p~) <= sigma^2 E[(s_v s_w - 2 H_vw)^2] = Lambda J_ww   =>   P# >= x^2 / Lambda    [proved]
    #   In posterior variables (zeta = E[Z | y], Z the blur noise):  Lambda = E[(2 E[Z_v Z_w | y] - zeta_v zeta_w)^2] / E[zeta_w^2].
    Hvw = f['Hvw']; g = sv * sw - 2 * Hvw
    dlog = s2 * L(sv * g)                                                  # = (p_W omega)' / p_W
    aIpt = Ew(np.where(om > 0, dlog ** 2 / np.where(om > 0, om, 1.0), 0.0))
    Lam = s2 * E(g * g) / Jww
    pt = Pc * om / a; mu_ = ub - f['wcol']; var_mu = float(np.sum(pt * mu_ ** 2) - np.sum(pt * mu_) ** 2)
    out.update({'sd_ptilde_mu_over_sigma': np.sqrt(max(var_mu, 0)) / f['sig'],
                'x_bound_ptilde': np.sqrt(max(var_mu, 0)) * np.sqrt(aIpt / a),            # |x - 1| <= sd_p~(mu) sqrt(I(p~))      [proved]
                'x_bound_Lambda': np.sqrt(max(var_mu, 0)) * np.sqrt(Lam * Jww / a)})        # <= sd_p~(mu) sqrt(Lambda J_ww / a)   [proved]
    out.update({'aI_ptilde': aIpt, 'aIpt_over_Jww': aIpt / Jww, 'Lambda': Lam, 'bound_ptilde': (max(X, 0) / a) ** 2 * Jww / aIpt, 'bound_Lambda': (max(X, 0) / a) ** 2 / Lam,
                'ptilde_identity_vs_fd': float(np.sum(Pc * np.abs(np.gradient(Pc * om, h1) / np.where(Pc > 0, Pc, 1) * Pc - dlog * Pc)) / max(np.sum(Pc * np.abs(dlog)), 1e-300))})
    j = E(np.clip(s2 * np.maximum(Hvv, 0), 0, 1)); out['y_over_sigma'] = f['sig'] * Jww / j
    return out


def all_t(ws, mus, Ss, sig, per_sd=6):
    f = fields(ws, mus, Ss, sig, per_sd); t = t_terms(f); c = chain(f); t.update({'bound_sharp_S': c['bound_sharp'], 'Gamma_eff': c['Gamma_eff'], 'grid_mass': f['mass']}); return t


def t_all(ws, mus, Ss, sig, per_sd=6):
    return all_t(ws, mus, Ss, sig, per_sd)


def t3_terms(ws, mus, Ss, sig, per_sd=6, max_points=2_000_000):
    """Stage T3: the two correction factors of R4.
    delta = E[(sigma^2 H_vv)_-] / a.  For a Gaussian mixture, H = sum_k pi_k T_k^{-1} - B, B = sum_k pi_k (u_k - s)(u_k - s)^T (between-component
    posterior spread, u_k = component score), and 0 < sigma^2 (T_k^{-1})_vv <= 1, hence (sigma^2 H_vv)_- <= sigma^2 B_vv:
        delta <= delta_mix := sigma^2 E[B_vv] / a                                     [proved; zero for a single Gaussian]
    T2/T1 = 2 E[(d_v u)^2 + u d_v^2 u] / E[u^2 s_v^2],  d_v u = 1 - sigma^2 (H_vv + H_vw) (closed form), d_v^2 u by a difference along V:
        |T2/T1| <= 2 (E[(d_v u)^2] + sqrt(E[u^2] E[(d_v^2 u)^2])) / E[u^2 s_v^2]     [proved, Cauchy-Schwarz]"""
    from i_tools import grid, V, W
    Y, dA, n = grid(ws, mus, Ss, sig, per_sd=per_sd, max_points=int(min(max_points, 2.0e7 / len(ws)))); s2 = sig * sig
    T = [S + s2 * np.eye(2) for S in Ss]; Ti = [np.linalg.inv(t) for t in T]
    lg = [np.log(w) - 0.5 * np.log(np.linalg.det(t)) - np.log(2 * np.pi) for w, t in zip(ws, T)]
    logc = np.stack([l - 0.5 * np.einsum('ni,ij,nj->n', Y - m, ti, Y - m) for l, m, ti in zip(lg, mus, Ti)], 1)
    mx = logc.max(1); e = np.exp(logc - mx[:, None]); p = e.sum(1) * np.exp(mx); pi = e / e.sum(1, keepdims=True); del e, logc
    uk = np.stack([-(Y - m) @ ti for m, ti in zip(mus, Ti)], 1); s = np.einsum('nk,nkd->nd', pi, uk)
    Bvv = np.einsum('nk,nk->n', pi, ((uk - s[:, None, :]) @ V) ** 2); Tvv = np.einsum('nk,k->n', pi, np.array([V @ ti @ V for ti in Ti]))
    H = np.einsum('nk,kij->nij', pi, np.array(Ti)) - (np.einsum('nk,nki,nkj->nij', pi, uk, uk) - np.einsum('ni,nj->nij', s, s))
    pw = p * dA; pw = pw / pw.sum(); E = lambda g: float(np.sum(pw * g)); m = Y + s2 * s
    Hvv = np.einsum('i,nij,j->n', V, H, V); Hvw = np.einsum('i,nij,j->n', V, H, W); sv = s @ V; u = np.sqrt(2) * (m[:, 0] - np.pi)
    a = s2 * E(sv * sv); delta = E(np.maximum(-s2 * Hvv, 0)) / a; delta_mix = s2 * E(Bvv) / a
    dvu = 1 - s2 * (Hvv + Hvw); G = dvu.reshape(n); h0 = (Y.reshape(n + (2,))[1, 0] - Y.reshape(n + (2,))[0, 0]) @ V
    d2 = np.zeros_like(G); d2[1:-1] = (G[2:] - G[:-2]) / (2 * h0); d2 = d2.ravel()
    U2 = u.reshape(n); fd = np.zeros_like(U2); fd[1:-1] = (U2[2:] - U2[:-2]) / (2 * h0); fd = fd.ravel()
    T1 = s2 * E(u * u * sv * sv); T2 = s2 * E(u * u * (sv * sv - Hvv))
    T2_formula = 2 * s2 * E(dvu * dvu + u * d2)
    bound = 2 * s2 * (E(dvu * dvu) + np.sqrt(E(u * u) * E(d2 * d2))) / T1
    return {'delta': delta, 'delta_mix': delta_mix, 'delta_mix_ok': bool(delta <= delta_mix * (1 + 1e-9) + 1e-15),
            'Tvv_max': float(s2 * Tvv.max()), 'T2_over_T1': T2 / T1, 'T2_formula_over_T1': T2_formula / T1, 'T2_bound': bound,
            'dvu_check': float(np.sqrt(E((fd - dvu) ** 2 * (np.abs(fd) > 0)))), 'E_dvu2': E(dvu * dvu), 'sigma': sig}
