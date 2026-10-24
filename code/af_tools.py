"""Stage AF: pointwise eigen-split of the curvature term -(H^3)_ww (all quantities in units of t: K = C/t = I - tH).
At each point: eigenvalues mu_i of K (posterior covariance / t), unit eigenvectors f_i, a_i = (e_w . f_i)^2, lambda_i = mu_i - 1.
  -(tH)^3_ww = sum_i lambda_i^3 a_i  <=  sum_i (lambda_i)_+^3 a_i, and with thresholds A, a0 ((1 + A) a0 =: c1 > 1):
  (i)   lambda_i <= A:                lambda_i^3 a_i <= A lambda_i^2 a_i <= A (tH)^2_ww                     [shallow, -> A tR]
  (ii)  lambda_i > A, a_i >= a0:      lambda_i^3 a_i <= K_ww^3 / a0^2,  and then K_ww >= mu_i a_i >= c1    [w-visible: the w-line lemma]
  (iii) lambda_i > A, a_i < a0:       Y := the rest                                                      [deep valleys within angle asin(sqrt a0) of v]
Also the stage-AE split: diag = -(tH_ww)^3, cross = -(tH_vw)^2 (tH_vv + 2 tH_ww), N_eta, X_eta (eta = 0.1).
decompose(W, Hvv, Hvw, Hww, t, J) -> dict of expectations divided by t J_ww (dimensionless).  Python 3.11."""
import numpy as np

SPLITS = ((19.0, 0.1), (9.0, 0.2), (3.0, 0.5), (1.0, 0.55))
ETA = 0.1


def eig2(Kvv, Kvw, Kww):
    tr = Kvv + Kww; d = np.sqrt(np.maximum((Kvv - Kww) ** 2 / 4 + Kvw ** 2, 0)); m1 = tr / 2 + d; m2 = tr / 2 - d
    # eigenvector of m1: (Kvw, m1 - Kvv) or (m1 - Kww, Kvw); a1 = w-component^2
    x1, y1 = Kvw, m1 - Kvv; x2, y2 = m1 - Kww, Kvw; n1 = x1 * x1 + y1 * y1; n2 = x2 * x2 + y2 * y2
    use1 = n1 >= n2; nx = np.where(use1, x1, x2); ny = np.where(use1, y1, y2); nn = np.where(use1, n1, n2)
    a1 = np.where(nn > 1e-300, ny * ny / np.where(nn > 1e-300, nn, 1.0), np.where(Kww >= Kvv, 1.0, 0.0))
    return m1, m2, a1, 1 - a1


def decompose(W, Hvv, Hvw, Hww, t, J):
    W = W / W.sum(); Kvv, Kvw, Kww = 1 - t * Hvv, -t * Hvw, 1 - t * Hww
    m1, m2, a1, a2 = eig2(Kvv, Kvw, Kww); l1, l2 = m1 - 1, m2 - 1; tJ = t * J
    tot = -((t * Hww) ** 3 + (t * Hvw) ** 2 * (t * Hvv + 2 * t * Hww))
    eig_tot = l1 ** 3 * a1 + l2 ** 3 * a2
    H2 = (t * Hvw) ** 2 + (t * Hww) ** 2
    out = {'kappa_H': np.sum(W * tot) / tJ, 'eig_identity_max': float(np.max(np.abs(tot - eig_tot) / (1 + np.abs(tot)))), 'tR': np.sum(W * H2) / tJ,
           'diag': np.sum(W * -(t * Hww) ** 3) / tJ, 'cross': np.sum(W * -(t * Hvw) ** 2 * (t * Hvv + 2 * t * Hww)) / tJ}
    q = np.clip(-(t * Hvv + 2 * t * Hww), 0, None); d = np.clip(-t * Hww, 0, None)
    out['N_eta'] = np.sum(W * d ** 3 * (d > ETA)) / tJ; out['X_eta'] = np.sum(W * (t * Hvw) ** 2 * q * (q > 3 * ETA)) / tJ
    for A, a0 in SPLITS:
        c1 = (1 + A) * a0; key = 'A%g_a%g' % (A, a0); sh = 0.0; vis = 0.0; Y = 0.0
        for l, a in ((l1, a1), (l2, a2)):
            lp = np.clip(l, 0, None); deep = l > A
            sh = sh + lp ** 3 * a * (~deep); vis = vis + lp ** 3 * a * (deep & (a >= a0)); Y = Y + lp ** 3 * a * (deep & (a < a0))
        out[key] = {'shallow_over_AtR': (np.sum(W * sh) / tJ) / max(A * out['tR'], 1e-300), 'visible': np.sum(W * vis) / tJ,
                    'visible_bound_Kww3': np.sum(W * 2 * Kww ** 3 * (Kww >= c1)) / tJ / a0 ** 2, 'Y': np.sum(W * Y) / tJ,
                    'vis_pointwise_ok': bool(np.all((vis <= 2 * Kww ** 3 * (Kww >= c1) / a0 ** 2 + 1e-9 * (1 + vis)) | (m2 < -1e-8)))}   # the bound needs K >= 0 (round-off far from the mass can break it)
    out['nonpsd_mass'] = float(np.sum(W * (m2 < -1e-8)))
    out['ratio_X_over_N_etatR_1'] = out['X_eta'] / (out['N_eta'] + ETA * out['tR'] + 1)
    return {k: (float(v) if not isinstance(v, dict) else v) for k, v in out.items()}
