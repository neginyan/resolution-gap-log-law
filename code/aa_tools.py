"""Stage AA1: the x-bar bound of Theorem Z without K_J and K_rho (closed forms; numpy only).

Proved chain (any t in (0,1); s1^2 = (1 - t) sigma^2, s2^2 = t sigma^2, rho_{s1} = rho * N(0, s1^2 I)):
  (A) split Jensen:   t xbar_w(p) <= ln( nu_t / |d_w p|^2 ),   nu_t = int xi_w^2 exp(-s2^2 xi_w^2) |rho_{s1}^|^2 = |d_w (rho blurred by diag(s1^2, sigma^2))|^2
  (B)                 nu_t <= |rho_{s1}|^2 / (e s2^2)                                   (xi^2 exp(-s2^2 xi^2) <= 1/(e s2^2))
  (C) kernel ratio:   |rho_{s1}|^2 <= (sigma^2/s1^2) |p|^2                              (rho >= 0; N(d; 2 s1^2 I)/N(d; 2 sigma^2 I) <= sigma^2/s1^2)
  (D) L2 Heisenberg:  |d_w p|^2 >= |p|^4 / (8 |y_z p|^2) = |p|^2 / (8 E_Q[y_z^2]),  Q = p^2/|p|^2   (slice-wise in w, centre w = -v, then Cauchy-Schwarz in v)
  (E) pair midpoints: Q = mixture over pairs (x, x') of N(y; (x + x')/2, sigma^2/2 I)  =>  E_Q[y_z^2] <= pi^2 + sigma^2/2 under H0
  =>  xbar_w(p) <= Xt := (1/t) [ ln(1/sigma^2) + ln( 8 E_Q[y_z^2] / (e t (1 - t)) ) ]   (state form)   <= (1/t)[2 ln(1/sigma) + ln(8 (pi^2 + sigma^2/2)/(e t (1-t)))]
Python 3.11."""
import numpy as np

EZ = np.array([1.0, 1.0]) / np.sqrt(2.0)


def pair_l2(terms, blur, want_yz2=False):
    """for f = rho * N(0, blur):  I0 = |f|^2, Iww = |d_w f|^2, Ivvww = |d_v d_w f|^2  and (optionally) Y2 = int y_z^2 f^2  (closed-form Gaussian overlaps)."""
    A = [np.asarray(S, float) + blur for _, _, S in terms]; I0 = Iww = Ivvww = Y2 = 0j
    for k, (lck, mk, _) in enumerate(terms):
        Ak_i = np.linalg.inv(A[k])
        for l, (lcl, ml, _) in enumerate(terms):
            Sg = A[k] + A[l]; Q = np.linalg.inv(Sg); d = np.asarray(mk, complex) - np.asarray(ml, complex); g = Q @ d
            Phi = np.exp(complex(lck) + complex(lcl) - np.log(2 * np.pi * np.sqrt(np.linalg.det(Sg))) - 0.5 * d @ Q @ d)
            gv, gw = g; Qvv, Qww, Qvw = Q[0, 0], Q[1, 1], Q[0, 1]
            I0 += Phi; Iww += (Qww - gw * gw) * Phi
            Ivvww += (gv * gv * gw * gw - Qvv * gw * gw - Qww * gv * gv - 4 * Qvw * gv * gw + Qvv * Qww + 2 * Qvw * Qvw) * Phi
            if want_yz2:
                Al_i = np.linalg.inv(A[l]); Cp = np.linalg.inv(Ak_i + Al_i); mp = Cp @ (Ak_i @ np.asarray(mk, complex) + Al_i @ np.asarray(ml, complex))
                Y2 += (EZ @ Cp @ EZ + (EZ @ mp) ** 2) * Phi
    return float(I0.real), float(Iww.real), float(Ivvww.real), float(Y2.real)


def strip_l2(P, bv, bw, want_yz2=False):
    """the same four integrals for the ripple strip of z_strip (separable closed form, 80-bit; the ripple pieces do not overlap)."""
    LD = np.longdouble
    sig, k0, c, ell, Lw, Lv = (LD(P[k]) for k in ('sig', 'k0', 'c', 'ell', 'Lw', 'Lv')); kap = k0 / sig; bv, bw = LD(bv), LD(bw)
    n = lambda S: 1 / (2 * np.sqrt(LD(np.pi)) * S); d = lambda S: 1 / (4 * np.sqrt(LD(np.pi)) * S ** 3)
    Le = 1 / np.sqrt(1 / Lw ** 2 + 1 / ell ** 2); gam = Le / Lw; win = c * c / 2 * gam * gam
    Sv, Sw, Se = np.sqrt(Lv * Lv + bv), np.sqrt(Lw * Lw + bw), np.sqrt(Le * Le + bw)
    damp = np.exp(-kap * kap * Lv * Lv * bv / (Sv * Sv)); xi0 = kap * Lv * Lv / (Sv * Sv); xi2 = xi0 * xi0 + 1 / (2 * Sv * Sv)
    Ib, Iw = n(Sv) * n(Sw), win * damp * n(Sv) * n(Se)
    I0 = Ib + Iw; Iww = n(Sv) * d(Sw) + win * damp * n(Sv) * d(Se); Ivvww = d(Sv) * d(Sw) + win * damp * n(Sv) * xi2 * d(Se)
    Y2 = (Ib * (Sv * Sv + Sw * Sw) + Iw * (Sv * Sv + Se * Se)) / 4          # f^2 of each piece is a Gaussian with half the variances
    return I0, Iww, Ivvww, Y2


def xbar_chain(l2, sig, t):
    """l2(bv, bw, want_yz2) -> (I0, Iww, Ivvww, Y2).  Returns every step of the AA1 chain (left >= right)."""
    LD = np.longdouble; s2 = LD(sig) ** 2; s1sq, s2sq = (1 - t) * s2, t * s2; f = float
    I0, Iww, Ivvww, Y2 = (LD(q) for q in l2(f(s2), f(s2), True))
    xw = float(s2 * Ivvww / Iww); EQ = float(Y2 / I0)
    nu = LD(l2(f(s1sq), f(s2), False)[1]); R1 = LD(l2(f(s1sq), f(s1sq), False)[0])
    out = {'t': t, 'xbar_w': xw, 'E_Q_yz2': EQ,
           'A_jensen': [float(np.log(nu / Iww)), t * xw],
           'B_cut': [float(R1 / (np.e * s2sq)), float(nu)],
           'C_kernel': [float(s2 / s1sq * I0), float(R1)],
           'D_heis_L2': [float(Iww / I0), 1 / (8 * EQ)],
           'X_t_state': float((np.log(1 / s2) + np.log(8 * EQ / (np.e * t * (1 - t)))) / t),
           'X_t_uniform': float((np.log(1 / s2) + np.log(8 * (np.pi ** 2 + float(s2) / 2) / (np.e * t * (1 - t)))) / t)}
    out['final'] = [out['X_t_state'], xw]
    return out


def best_t(sig, EQ=np.pi ** 2 + 0.5):
    """t maximising t / (2 ln(1/sigma) + ln(8 EQ/(e t (1-t))))  (grid search)."""
    ts = np.linspace(0.01, 0.999, 2000); L = np.log(1 / sig)
    f = ts / (2 * L + np.log(8 * EQ / (np.e * ts * (1 - ts)))); i = int(np.argmax(f)); return float(ts[i]), float(f[i])
