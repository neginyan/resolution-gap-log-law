"""Stage W1: the generalised ripple-strip family, exact and cancellation-free (built on v3_tools).
rho(v, w) = N(v; 0, Lv^2) * sum_b beta_b N(w; c_b, L_b^2) * [ 1 + sum_r c_r exp(-(w - w_r)^2 / (2 ell_r^2)) cos(m_r k_b v / sigma) ],
  background: B Gaussian components in W (free centres, widths, weights: flat, two-sided, mixed backgrounds, transitions);
  windows:    R ripple windows (free centres, widths, amplitudes) with wavenumbers m_r * k_b (integers m_r: superpositions k, 2k; 'W-varying k0'
              as adjacent windows with 2 k_b and 3 k_b);  sum_r c_r <= 1  =>  rho >= 0.
Each (background x window) product is a Gaussian times a plane wave (closed form), so v3_tools evaluates everything exactly.
Strip limit (sigma -> 0, the regime of stage V): one V-period of the base wavenumber at v = 0, Lv = 1e10 sigma (Lv << ell), sigma = 1e-24.
The gap uses the cancellation-free formula 1 - R = (2 sigma^2 J_ww + N)/(sigma^2 (J_vv + J_ww)), N = -E[d_p D] (stage V).
Lengths in W are given in units of sqrt(sigma).  Python 3.11 compatible."""
import numpy as np
from v3_tools import grid_sums
from u_cgm import nodes

SIG = 1e-24; LV_SIG = 1e10


def terms_of(bg, win, kb, sig=SIG, Lv_sig=LV_SIG, Rv_sig=None):
    """bg: list of (beta, c, L) [c, L in sqrt(sigma) units]; win: list of (c_r, w_r, ell_r, m_r).  Returns native v3_tools terms.
    Rv_sig: optional V-envelope exp(-v^2/2 Rv^2) of the windows (ripple confined in v; default: the ripple fills the strip)."""
    s = np.sqrt(sig); Lv = Lv_sig * sig; T = []
    Svw = Lv ** 2 if Rv_sig is None else 1 / (1 / Lv ** 2 + 1 / (Rv_sig * sig) ** 2); Av = np.sqrt(Svw) / Lv
    bsum = sum(b[0] for b in bg)
    for beta, c, L in bg:
        T.append((np.log(beta / bsum), np.array([0.0, c * s]), np.diag([Lv ** 2, (L * s) ** 2]), np.zeros(2)))
    for beta, c, L in bg:
        for wn_ in win:
            cr, wr, er, mr = wn_[:4]; ph = wn_[4] if len(wn_) > 4 else 0.0           # optional phase (stage X): cos(m k v + phi)
            Lb2, e2 = (L * s) ** 2, (er * s) ** 2
            Sbr = 1 / (1 / Lb2 + 1 / e2); mbr = Sbr * (c * s / Lb2 + wr * s / e2)
            Abr = Av * np.sqrt(Sbr / Lb2) * np.exp(-(c * s - wr * s) ** 2 / (2 * (Lb2 + e2)))
            if Abr <= 0 or cr <= 0: continue
            k = mr * kb / sig
            for sgn in (1, -1):
                T.append((np.log(beta / bsum * cr * Abr / 2) + (1j * sgn * ph if ph else 0.0), np.array([0.0, mbr]), np.diag([Svw, Sbr]), np.array([sgn * k, 0.0])))
    return T


def w_nodes(bg, win, sig=SIG, nper=40, coarse=400):
    s = np.sqrt(sig)
    lo = min(c - 6.5 * L for b_, c, L in bg); hi = max(c + 6.5 * L for b_, c, L in bg)
    pts = [np.linspace(lo, hi, 4 * coarse + 1)]                                   # global coarse grid (covers gaps / transitions)
    for b_, c, L in bg: pts.append(np.arange(c - 6.5 * L, c + 6.5 * L + 1e-12, L / coarse))   # each background component resolved
    for wn_ in win:
        cr, wr, er, mr = wn_[:4]
        a, b = max(wr - 10 * er, lo), min(wr + 10 * er, hi)
        if b > a: pts.append(np.arange(a, b + 1e-12, er / nper))
    w = np.unique(np.concatenate(pts)); return w * s


def evaluate(bg, win, kb, sig=SIG, nv_per=32, nper=40, coarse=400, Lv_sig=LV_SIG):
    mmax = max([w_[3] for w_ in win] + [1]); per = 2 * np.pi * sig / kb; nv = nv_per * mmax
    vn = (np.arange(nv) + 0.5) * per / nv - per / 2; wv = np.full(nv, per / nv)
    T = terms_of(bg, win, kb, sig, Lv_sig); wn = w_nodes(bg, win, sig, nper, coarse)
    out = grid_sums(T, sig, vn, wn, rows=16, wv=wv); out['gap_over_sigma_N'] = out['gap_over_sigma']; out['gap_over_sigma'] = out['gap_over_sigma_ibp']; out['n_w'] = len(wn); out['n_v'] = nv
    out['mass_ratio'] = out['grid_mass'] / (per / np.sqrt(2 * np.pi * (Lv_sig ** 2 + 1)) / sig)   # 1 when the W grid captures all mass
    return out
