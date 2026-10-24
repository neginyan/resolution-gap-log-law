"""Independent evaluator of the ripple strips (written separately from the code in code/; shares no code with it).
State: rho(v,w) = N(v;0,Lv^2) N(w;0,Lw^2) [1 + c exp(-w^2/2 l^2) cos(k v/s)], s = sigma.
Own derivation (main session):
  * rho = sum of separable terms; in v the ripple term is Re[N(v;0,L^2) e^{i b v}], b = k/s, and
        N(x;0,L^2) e^{i b x} = e^{-b^2 L^2/2} N(x; i b L^2, L^2)          (complex Gaussian)
    so every term is (weight) x (Gaussian prior with possibly complex mean); its blur and its posterior are closed form:
        blur:      N(y; mu0, T),  T = L^2 + s^2
        posterior: mean (s^2 mu0 + L^2 y)/T, variance L^2 s^2 / T.
  * In w the window factor N(w;0,Lw^2) e^{-w^2/2l^2} = alpha N(w;0,Lp^2), 1/Lp^2 = 1/Lw^2 + 1/l^2, alpha = Lp/Lw.
  * D(y) = E[z - sin z | y], z = (x_v + x_w)/sqrt2, evaluated term by term with Gaussian posteriors:
        E[z - sin z] = (mu - sin mu) + sin(mu) (1 - e^{-V/2})   (series for mu - sin mu: no cancellation)
  * gap (main-session derivation, see check of stage S/V): 1 - R = (2 s^2 J_ww + E[s_p D]) / (s^2 (J_vv + J_ww)), s_p = (s_v - s_w)/sqrt2.
  * Two scales: theta = b Kv y_v is treated as an independent phase (average over one period), y_v -> slow Y elsewhere;
    error ~ (period / Lv)^2.
"""
import numpy as np, json, sys

def gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n); return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

def zms(mu):                         # mu - sin(mu), accurate for small complex mu
    out = np.empty_like(mu); small = np.abs(mu) < 0.05
    m = mu[small]; m2 = m * m
    out[small] = m * m2 / 6 * (1 - m2 / 20 * (1 - m2 / 42 * (1 - m2 / 72 * (1 - m2 / 110))))
    out[~small] = mu[~small] - np.sin(mu[~small]); return out

def Ng(y, T): return np.exp(-y * y / (2 * T)) / np.sqrt(2 * np.pi * T)

def evaluate(sig, k, lam, M, c, Lv, nY=48, nth=24, panels=12):
    s2 = sig * sig; l = lam * np.sqrt(sig); Lw = M * l
    Lp2 = 1 / (1 / Lw ** 2 + 1 / l ** 2); alpha = np.sqrt(Lp2) / Lw
    Tv, Tw, Tp = Lv ** 2 + s2, Lw ** 2 + s2, Lp2 + s2
    Kv, Kw, Kp = Lv ** 2 / Tv, Lw ** 2 / Tw, Lp2 / Tp
    b = k / sig; B = b * Kv
    E1 = c * np.exp(-0.5 * b * b * Lv ** 2 * s2 / Tv)            # = c e^{-u^2/2}
    imv = b * Lv ** 2 * s2 / Tv                                   # imaginary part of the v posterior mean of the ripple term
    Vv = Lv ** 2 * s2 / Tv; Vw = Lw ** 2 * s2 / Tw; Vp = Lp2 * s2 / Tp
    # nodes
    Y, wY = gl(-9 * np.sqrt(Tv), 9 * np.sqrt(Tv), nY)
    th = 2 * np.pi * np.arange(nth) / nth; wth = np.full(nth, 1.0 / nth)        # phase AVERAGE
    sp, s0 = np.sqrt(Tp), np.sqrt(Tw)
    br = [0.0] + list(sp * np.array([0.5, 1, 2, 3, 4, 6, 8, 12])) + list(np.geomspace(14 * sp, 10 * s0, panels))
    br = sorted(set([x for x in br if x <= 10 * s0]))
    yw, wyw = [], []
    for a_, b_ in zip(br[:-1], br[1:]):
        x, w = gl(a_, b_, 10); yw += list(x) + list(-x); wyw += list(w) * 2
    yw, wyw = np.array(yw), np.array(wyw)
    Yg, TH, YW = np.meshgrid(Y, th, yw, indexing='ij'); W = wY[:, None, None] * wth[None, :, None] * wyw[None, None, :]
    cth, sth = np.cos(TH), np.sin(TH)
    Nv = Ng(Yg, Tv); Nw0 = Ng(YW, Tw); Nw1 = alpha * Ng(YW, Tp)
    p = Nv * (Nw0 + E1 * Nw1 * cth)
    dvp = Nv * (-Yg / Tv * (Nw0 + E1 * Nw1 * cth) - E1 * Nw1 * B * sth)
    dvvp = Nv * ((Yg ** 2 / Tv ** 2 - 1 / Tv) * Nw0 + E1 * Nw1 * ((Yg ** 2 / Tv ** 2 - 1 / Tv - B * B) * cth + 2 * Yg / Tv * B * sth))
    dwp = Nv * (-YW / Tw * Nw0 - YW / Tp * E1 * Nw1 * cth)
    sv, sw = dvp / p, dwp / p; Hvv = sv * sv - dvvp / p
    # D * p, term by term (posterior means/variances)
    r2 = np.sqrt(2.0)
    mu0 = (Kv * Yg + Kw * YW) / r2; V0 = (Vv + Vw) / 2
    Dp = Nv * Nw0 * (zms(mu0.astype(complex)).real + np.sin(mu0) * (-np.expm1(-V0 / 2)))
    V1 = (Vv + Vp) / 2
    for sgn in (+1, -1):
        mu1 = (Kv * Yg + 1j * sgn * imv + Kp * YW) / r2
        Wt = 0.5 * E1 * Nv * Nw1 * np.exp(1j * sgn * TH)
        Dp = Dp + (Wt * (zms(mu1) + np.sin(mu1) * (-np.expm1(-V1 / 2)))).real
    D = Dp / p
    # ---- N = -E[d_p D] with d_y E[g|y] = s^-2 Cov(g, x | y)  (no cancellation; main-session derivation) ----
    # mixture of (complex) Gaussian posteriors: Cov = sum_t w_t Cov_t  +  between-term part
    w0 = Nv * Nw0 / p
    wpm = {sg: 0.5 * E1 * Nv * Nw1 * np.exp(1j * sg * TH) / p for sg in (+1, -1)}
    E0 = np.exp(-V0 / 2); E1v = np.exp(-V1 / 2)
    one_m_cos = lambda mu, Ev: 2 * np.sin(mu / 2) ** 2 + np.cos(mu) * (1 - Ev)
    # within: Cov_t(z - sin z, x_p) = Cov_t(z, x_p) E_t[1 - cos z],  Cov_t(z, x_p) = (V_v - V_w,t)/2,  divided by s^2:
    cz0 = 0.5 * (s2 / Tw - s2 / Tv)            # (Vv - Vw)/(2 s^2) = (Kv - Kw)/2
    cz1 = 0.5 * (s2 / Tp - s2 / Tv)            # (Vv - Vp)/(2 s^2)
    within = w0 * cz0 * one_m_cos(mu0, E0)
    dpart, dmu_list, dD_list = 0, [], []
    for sg in (+1, -1):
        mu1 = (Kv * Yg + 1j * sg * imv + Kp * YW) / r2
        within = within + (wpm[sg] * cz1 * one_m_cos(mu1, E1v)).real
        dmu = (1j * sg * imv + (s2 / Tp - s2 / Tw) * YW) / r2            # mu_p,1 - mu_p,0
        dz = (1j * sg * imv + (s2 / Tw - s2 / Tp) * YW) / r2             # mu_z,1 - mu_z,0
        m = mu0 + dz / 2; sd = np.sin(dz / 2)
        dD = (dz ** 3 / 24 - dz ** 5 / 1920) + 2 * sd * (2 * np.sin(m / 2) ** 2) + 2 * sd * np.cos(m) * (-np.expm1(-V1 / 2)) \
             - np.sin(mu0) * E0 * np.expm1(-(V1 - V0) / 2)
        dmu_list.append(wpm[sg] * dmu); dD_list.append(wpm[sg] * dD); dpart = dpart + wpm[sg] * dD * dmu
    between = (dpart - (dD_list[0] + dD_list[1]) * (dmu_list[0] + dmu_list[1])).real / s2
    dpD = within + between
    E = lambda f: float(np.sum(W * p * f))
    mass = E(np.ones_like(p))
    Jvv, Jww = E(sv * sv) / mass, E(sw * sw) / mass
    Nn = -E(dpD) / mass
    N_direct = E((sv - sw) / r2 * D) / mass
    j = E(np.clip(s2 * Hvv, 0, 1)) / mass
    gap = (2 * s2 * Jww + Nn) / (s2 * (Jvv + Jww)) / sig
    return dict(N_direct=N_direct, mass=mass, J_vv=Jvv, J_ww=Jww, N=Nn, a=s2 * Jvv, j=j, y_over_sigma=sig * Jww / j, gap_over_sigma=gap)
