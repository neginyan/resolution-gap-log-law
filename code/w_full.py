"""Stage W: the strip with its TRUE V-envelope (finite Lv), without resolving ~Lv/sigma ripple periods.
The integrand is (fast ripple, period 2 pi sigma/k_b) x (slow envelope in v, scale Lv >> sigma).  The integral over v is split exactly as
    int f dv = int dv0 [ (1/per) int_{one period at v0} f dv ],
the inner period average is taken on a uniform grid (spectral for periodic integrands) and the outer one by the trapezoid rule over
v0 in [-ext Lv, ext Lv] (spectral for the Gaussian-like slow factor).  Field values are exact at every point (v3_tools.point_fields with the
cancellation-free fixes of stage W).  The one-period 'V-limit' of stage V is the special case of a single v0 = 0.  Python 3.11."""
import numpy as np
from v3_tools import _pre, point_fields, _finish
from u_cgm import trap_weights

KEYS = ('svv', 'sww', 'N', 'Nibp', 'j', 'Z', 'sEf', 'u2svv')


def period_sums(pre, sig, v0, per, nv, wn, ww, rows=16):
    vn = v0 + (np.arange(nv) + 0.5) * per / nv - per / 2; wv = np.full(nv, 1.0 / nv)        # period AVERAGE
    acc = {k: 0.0 for k in KEYS}; mass = 0.0
    for a in range(0, nv, rows):
        V, W = np.meshgrid(vn[a:a + rows], wn, indexing='ij'); x = np.stack([V.ravel(), W.ravel()], 1); dA = np.outer(wv[a:a + rows], ww).ravel()
        p, q = point_fields(pre, sig, x); pw = p * dA; mass += pw.sum()
        for k in acc: acc[k] += float(np.sum(pw * q[k]))
    return acc, mass


def full_sums(terms, sig, kb, Lv, wn, nv=32, n0=41, ext=7.0, Rv=None):
    """Rv: optional narrower V-scale (a window confined in v); the v0 nodes then also resolve [-ext Rv, ext Rv].
    Accuracy (stage X): the one-period average of slow x fast leaves an oscillation of relative size ~ period/Lv at the ripple frequency in v0,
    which the coarse v0 trapezoid aliases: error ~ period/Lv (3e-3 at the sigma = 1e-6 optimum, period/Lv = 1/138; negligible for sigma <= 1e-12)."""
    pre = _pre(terms, sig); per = 2 * np.pi * sig / kb; ww = trap_weights(wn)
    if ext * Lv / per > 1e12:      # absolute positions: v0 + (fraction of a period) loses the phase once v0/period ~ 1/eps (stage W finding)
        raise ValueError('positional precision: ext*Lv/period = %.1e > 1e12' % (ext * Lv / per))
    v0s = np.linspace(-ext * Lv, ext * Lv, n0)
    if Rv is not None and Rv < Lv / 3: v0s = np.unique(np.concatenate([v0s, np.linspace(-ext * Rv, ext * Rv, n0)]))
    w0 = trap_weights(v0s)
    tot = {k: 0.0 for k in KEYS}; mass = 0.0
    for v0, c in zip(v0s, w0):
        a, m = period_sums(pre, sig, v0, per, nv, wn, ww); mass += c * m
        for k in tot: tot[k] += c * a[k]
    S = {k: v / mass for k, v in tot.items()}; out = _finish(S, sig, 'Nibp'); out['grid_mass'] = mass; out['N_ibp'] = S['Nibp']
    out['gap_over_sigma_ibp'] = out['gap_over_sigma']
    return out
