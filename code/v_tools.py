"""Stage V: the 'two eyes' lower bounds of the weight-free moment (notation of t_tools / u_tools).
  P# >= B2 := Q J_ww / a^2 = J_ww E_p~[ubar^2] / a                     (Jensen on the lines; proved, stage S/T)
  focus eye  (Cramer-Rao under p~):  E_p~[ubar^2] >= x^2 / I(p~)   =>  B2 >= B1 := x^2 J_ww / (a I(p~))     (proved; so max(B1, B2) = B2)
  spread eye:                        E_p~[ubar^2] >= Var_p~(ubar)  =>  B2 >= Bsp := J_ww Var_p~(ubar) / a  (trivial)
  Provable form of the spread eye (stage V):
    (S1) omega <= omega_max on every line  =>  p~ = p_W omega / a <= Omega p_W,  Omega := omega_max / a       (pointwise)
    (S2) sup p_W <= sqrt(I_W) / 2   (p_W(w) = int_{-inf}^w p_W' = -int_w^inf p_W'  =>  2 p_W(w) <= int |p_W'| = E|s_W| <= sqrt(I_W))
    (S3) a density bounded by M has variance >= 1 / (12 M^2)   (the uniform density is extremal)
    =>  Var_p~(w) >= 1 / (3 Omega^2 I_W) = a^2 / (3 omega_max^2 I_W)
    with ubar = (w - w_top) + mu:  sd_p~(ubar) >= sd_p~(w) - sd_p~(mu), hence
    B2 >= Bsp_proved := (J_ww / a) [ a / (omega_max sqrt(3 I_W)) - sd_p~(mu) ]_+^2
         ~ a J_ww / (3 omega_max^2 I_W) >= a / (3 omega_max^2)           when sd_p~(mu) is negligible.
Python 3.11 compatible."""
import numpy as np


def v_terms(f):
    s2 = f['sig'] ** 2; P = f['pw']; sv, sw, u, Hvv, Hvw = f['sv'], f['sw'], f['u'], f['Hvv'], f['Hvw']; h1 = f['h'][1]
    Pc = P.sum(0); keep = Pc > 1e-300 * max(Pc.max(), 1e-300); Pc_ = np.where(keep, Pc, 1.0)
    L = lambda g: np.sum(P * g, 0) / Pc_; Ew = lambda g: float(np.sum(Pc * g)); E = lambda g: float(np.sum(P * g))
    Iv = L(sv * sv); om = s2 * Iv; ub = L(u * sv * sv) / np.where(Iv > 0, Iv, 1.0)
    sW = L(sw); IW = Ew(sW * sW); Jww = E(sw * sw); a = Ew(om)
    X = Ew(om * np.gradient(ub, h1)); x = X / a
    g = sv * sw - 2 * Hvw; dlog = s2 * L(sv * g); pos = om > 0
    aIpt = Ew(np.where(pos, dlog ** 2 / np.where(pos, om, 1.0), 0.0))
    pt = Pc * om / a; wc = f['wcol']; mu = ub - wc
    Ept = lambda g_: float(np.sum(pt * g_))
    Eu2 = Ept(ub * ub); Vu = Eu2 - Ept(ub) ** 2; Vw = Ept(wc * wc) - Ept(wc) ** 2; Vmu = max(Ept(mu * mu) - Ept(mu) ** 2, 0)
    VpW = Ew(wc * wc) - Ew(wc) ** 2
    sig_lines = Pc > 1e-12 * Pc.max(); om_max = float(np.max(np.where(sig_lines, om, 0)))
    # density of p_W per unit w (column masses / spacing; non-uniform nodes allowed)
    wn = np.asarray(wc, float); dw = np.gradient(wn); pW_dens = Pc / dw; sup_pW = float(pW_dens.max())
    Q = Ew(om * ub * ub)
    B1 = max(x, 0) ** 2 * Jww / aIpt; B2 = Q * Jww / a ** 2
    Bsp = Jww * Vu / a; Bsp_w = Jww * Vw / a
    Bsp_proved = (Jww / a) * max(a / (om_max * np.sqrt(3 * IW)) - np.sqrt(Vmu), 0) ** 2
    j = E(np.clip(s2 * np.maximum(Hvv, 0), 0, 1)); Z = E(np.clip(s2 * np.maximum(Hvv, 0), 0, 1) * u * u)
    return {'a': a, 'x': x, 'Jww': Jww, 'I_W': IW, 'aIpt_over_Jww': aIpt / Jww, 'B1': B1, 'B2': B2, 'B2_check': Jww * Eu2 / a,
            'P_sharp': s2 * E(u * u * sv * sv) * Jww / a ** 2, 'Bsp': Bsp, 'Bsp_w': Bsp_w, 'Bsp_proved': Bsp_proved,
            'max_B1_Bsp': max(B1, Bsp), 'max_B1_Bsp_proved': max(B1, Bsp_proved), 'omega_max': om_max, 'Omega': om_max / a,
            'spread_ratio': Vw / (a * VpW) if VpW > 0 else 0.0, 'Var_pt_w_over_Var_pW': Vw / VpW if VpW > 0 else 0.0,
            'I_pt_Var_pt': aIpt / a * Vw, 'J_Var_pW': Jww * VpW,
            'check_S2': sup_pW / (np.sqrt(IW) / 2) if IW > 0 else 0.0, 'check_Var_bound': Vw / (a * a / (3 * om_max ** 2 * IW)) if IW > 0 else np.inf,
            'sd_pt_mu_over_sigma': np.sqrt(Vmu) / f['sig'], 'sd_pt_w_over_sqrt_sigma': np.sqrt(Vw / f['sig']),
            'P': Z * Jww / j ** 2, 'delta': E(np.maximum(-s2 * Hvv, 0)) / a, 'y_over_sigma': f['sig'] * Jww / j}
