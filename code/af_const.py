"""Stage AF: constants of the line lemma for general weights, and the two-sided improvement of (*).
Two-sided (*):  with Delta_+ + Delta_- >= D and f(x) = (e^{x/2} - 1)_+^2 convex nondecreasing on R (f = 0 for x <= 0),
  int_{|y - y0| <= u} p s^2 >= 4 p(y0) (f(Delta_+) + f(Delta_-))/u >= 8 p(y0) (e^{D/4} - 1)^2 / u     (Jensen),
which halves every constant of stage AE.  Line lemma with weight g on {c >= c_lo}:  E[g(c) 1{c >= c_lo}] <= kappa_g J  (t = 1),
  kappa_g = sum over bands 2 u_k u_k sup_band g(c) / (F (e^{D(u_k, c)/4} - 1)^2),  F = 4 (one-sided, stage AE) or 8 (two-sided).
Weights: g = (c - 1)^3 (stage AE, c_lo = 1 + eta) and g = c^3 (the w-visible part (ii) of the eigen split, c_lo = c1 = (1 + A) a0).
-> results/af_const.json.  Python 3.11."""
import json
import numpy as np
from ae_const import Dv


def kappa_g(g, c_lo, F=8.0, n=240, c_max=2.0e4, nu=160, sub=6):
    cg = np.concatenate([[c_lo], c_lo + np.geomspace(1e-3, c_max - c_lo, n * sub)])        # band grid in c (first sub-interval [c_lo, c_lo + 1e-3])
    U = np.geomspace(1e-3, 60.0, nu); lo, hi = cg[:-1], cg[1:]
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        Dm = Dv(U[:, None], lo[None, :]); Fm = np.where(Dm > 0, g(hi)[None, :] * U[:, None] / (F * np.expm1(np.minimum(Dm, 1400) / 4) ** 2), np.inf)
    nb = n; best = np.full(nb + 1, np.inf); best[0] = 0.0; arg = [None] * (nb + 1)
    for i in range(nb):
        if not np.isfinite(best[i]): continue
        run = np.zeros(nu)
        for j in range(i + 1, nb + 1):
            run = np.maximum(run, Fm[:, (j - 1) * sub:j * sub].max(axis=1)); v = 2 * U * run; k = int(np.argmin(v))
            if best[i] + v[k] < best[j]: best[j] = best[i] + v[k]; arg[j] = (i, float(U[k]), float(v[k]))
    ct = np.geomspace(c_max, 1e7, 4000)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        tail = min((2 * u * np.max(g(ct[1:]) * u / (F * np.expm1(np.minimum(Dv(u, ct[:-1]), 1400) / 4) ** 2)), u) for u in (20.0, 40.0, 80.0))
    bands = []; j = nb
    while j > 0:
        i, u, v = arg[j]; bands.append((float(cg[i * sub]), float(cg[j * sub]), u, v)); j = i
    return float(best[nb] + tail[0]), bands[::-1]


if __name__ == '__main__':
    out = {'cube_minus1': {}, 'cube': {}}
    for eta in (0.1, 0.5, 1.0):
        k1, _ = kappa_g(lambda c: (c - 1) ** 3, 1 + eta, F=4.0); k2, b = kappa_g(lambda c: (c - 1) ** 3, 1 + eta, F=8.0); k3, _ = kappa_g(lambda c: (c - 1) ** 3, 1 + eta, F=8.0, n=360)
        out['cube_minus1']['%g' % eta] = {'one_sided': k1, 'two_sided': k2, 'two_sided_finer': k3, 'bands': b}
        print('weight (c-1)^3, eta %.1f: one-sided %.1f, two-sided %.1f (finer %.1f)' % (eta, k1, k2, k3), flush=True)
    for c1 in (1.1, 2.0):
        k, b = kappa_g(lambda c: c ** 3, c1, F=8.0); kf, _ = kappa_g(lambda c: c ** 3, c1, F=8.0, n=360)
        out['cube']['%g' % c1] = {'two_sided': k, 'two_sided_finer': kf, 'bands': b}
        print('weight c^3, c >= %.1f: two-sided %.1f (finer %.1f)' % (c1, k, kf), flush=True)
    json.dump(out, open('results/af_const.json', 'w'), indent=1)
