"""Stage V2: the 'ripple strip' family at small sigma (exact, u_cgm): a V-ripple filling a narrow V-strip (Lv = 40-100 sigma) with a narrow
W window (ell = lambda sqrt(sigma)) on a background long in W (Lw = A_w sqrt(sigma), A_w = 6-100).  Built to test whether faint fine
structure can make P (and F) small while y/sigma <= 0.5; the TRUE gap is computed from the posterior.  sigma = 1e-6 (Lw <= 0.1: near the top).
Non-uniform W grid (fine in the window, coarse on the background); the gap uses the integration-by-parts-stable R of u_cgm
(the direct R is wrong by O(1e-3) on non-uniform grids: an early run reported gaps of 450-11000 sigma, all artefacts).  -> results/v2_strip.json (restartable by key).  Python 3.11 compatible."""
import json, os, time
import numpy as np
from u_cgm import evaluate, ripple_terms, nodes
from v_tools import v_terms

GRID = [(1.5, 1.65, 16, 100), (1.5, 1.65, 40, 100), (1.5, 1.65, 100, 100), (1.5, 1.2, 100, 100), (1.0, 1.5, 40, 100), (2.0, 1.2, 100, 100),
        (1.5, 1.65, 8, 100), (1.5, 1.65, 12, 100), (2.0, 1.2, 12, 100), (2.0, 1.2, 16, 100), (2.0, 0.8, 8, 100), (2.5, 1.0, 10, 100),
        (1.5, 1.65, 16, 40), (2.0, 1.2, 16, 40), (1.5, 1.0, 6, 100), (1.25, 1.65, 12, 100), (1.5, 2.0, 12, 100), (1.5, 1.65, 12, 200)]


def strip(sg, k0, lam, Aw, Lv_s, c=1.0, fine=1.0, fine_w=None, ext=6.5):
    s = np.sqrt(sg); Lv = Lv_s * sg; Lw = Aw * s; ell = lam * s; fw = fine if fine_w is None else fine_w
    T = ripple_terms(sg, Lv, Lw, c, k0, 10 * Lv, ell, 0.0)
    hv = min(0.35 * sg, 2 * np.pi * sg / k0 / 12) / fine
    vn = nodes(ext * Lv, hv, ext * Lv, hv); wn = nodes(7 * ell, ell / 40 / fw, ext * Lw, max(Lw / 120, ell / 40) / fw)
    while len(vn) * len(wn) > 8.0e6: hv *= 1.05; vn = nodes(ext * Lv, hv, ext * Lv, hv)          # <= ~1.3 GB
    f, dg = evaluate(T, sg, vn, wn); v = v_terms(f); dg.update({k: v[k] for k in ('B1', 'B2', 'P_sharp', 'a', 'delta', 'aIpt_over_Jww', 'Bsp_proved', 'omega_max')})
    return dg


if __name__ == '__main__':
    fn = 'results/v2_strip.json'; R = json.load(open(fn)) if os.path.exists(fn) else {}
    sg = 1e-6
    for k0, lam, Aw, Lv in GRID:
        key = 'k%g_lam%g_Aw%g_Lv%g' % (k0, lam, Aw, Lv)
        if key in R: continue
        t = time.time(); dg = strip(sg, k0, lam, Aw, Lv)
        R[key] = dict(dg, k0=k0, lam=lam, Aw=Aw, Lv_over_sigma=Lv, sigma=sg); json.dump(R, open(fn, 'w'), indent=1, default=float)
        print(key, 'P %.3f F %.3f y %.3f gap %.0f B2 %.3f delta %.2f (%.0fs)' % (dg['P'], dg['F'], dg['y_over_sigma'], dg['gap_over_sigma'], dg['B2'], dg['delta'], time.time() - t), flush=True)
    json.dump(R, open(fn, 'w'), indent=1, default=float)
