"""Stage W1: the true minimum gap of the ripple-strip family at a GIVEN sigma, g_min(sigma), with the true V-envelope (w_full) and the
stage-W cancellation-free point fields.  State: rho = N(v; 0, Lv^2) N(w; 0, Lw^2) (1 + c exp(-w^2/2 ell^2) cos(k0 v/sigma)),
ell = lam sqrt(sigma), Lw = M ell; free parameters (k0, lam, M, Lv, c).  Constraint y/sigma <= 0.5 (re-checked on finer grids).
usage: python3 w1_gmin.py SIGMA [MAXEVAL] [NSTART]   -> results/w1_gmin_opt_s<SIGMA>.json (checkpointed, restartable).  Python 3.11."""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize
from w_tools import terms_of, w_nodes
from w_full import full_sums

SIG = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-24; MAXEV = int(sys.argv[2]) if len(sys.argv) > 2 else 600
NSTART = int(sys.argv[3]) if len(sys.argv) > 3 else 99          # use only the first NSTART starts
OUT = 'results/w1_gmin_opt_s%g.json' % SIG


def state(x):
    k0, lam, M, eta = np.exp(x[:4]); c = 1 / (1 + np.exp(-x[4]))
    Lv_sig = eta * lam / np.sqrt(SIG)                     # Lv = eta * ell
    return k0, lam, M, Lv_sig, c


def gap(x, fine=1, n0=15, nv=32):
    k0, lam, M, Lv_sig, c = state(x)
    bg = [(1.0, 0.0, lam * M)]; win = [(c, 0.0, lam, 1)]
    T = terms_of(bg, win, k0, SIG, Lv_sig); wn = w_nodes(bg, win, SIG, 25 * fine, 200 * fine)
    return full_sums(T, SIG, k0, np.sqrt((Lv_sig * SIG) ** 2 + SIG ** 2), wn, nv=nv * fine, n0=n0 * fine, ext=6.5)


def objective(x, log):
    try:
        k0, lam, M, Lv_sig, c = state(x)
        if not (0.3 < k0 < 12 and 0.1 < lam < 50 and 1 < M < 1e60 and 1 < Lv_sig and M * lam * np.sqrt(SIG) < 2.0): return 60.0   # background near the top
        if Lv_sig < 30.0 / k0: return 62.0          # the period average needs Lv >~ 5 ripple periods (a stage-W run at 1e-4 went to Lv ~ sigma: invalid)
        o = gap(x); g = float(o['gap_over_sigma'])
        if not np.isfinite(g) or o['y_over_sigma'] > 0.5: return 40.0
        log.append(g); return g
    except (MemoryError, ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return 70.0


def starts():
    """k0 = 1.2, 2, 3 with M = 10 e^{k0^2}, capped so that the background stays near the top (Lw = M lam sqrt(sigma) <= 0.3);
    for sigma >= 1e-6 also the stage-V finite-strip optimum when one is stored (v2_gapmin)."""
    out = []; Mcap = 0.3 / (1.5 * np.sqrt(SIG))
    for k0 in (1.2, 2.0, 3.0):
        M = min(10 * np.exp(k0 ** 2), Mcap); eta = max(0.05, 2 * np.sqrt(SIG))          # Lv = eta ell >= 3 sigma
        out.append(np.log([k0, 1.5, max(M, 2.0), eta]).tolist() + [3.0])
    fn = 'results/v2_gapmin_s%g_seed0.json' % SIG
    if os.path.exists(fn):
        p = json.load(open(fn))['best']['par']; lam = p['lam']
        out.insert(0, np.array([np.log(p['k0']), np.log(lam), np.log(p['Aw'] / lam), np.log(p['Lv_s'] * np.sqrt(SIG) / lam), np.log(p['c'] / (1 - p['c']))]))
    return [np.array(s) for s in out][:NSTART]


def save(st):
    tmp = OUT + '.tmp'; json.dump(st, open(tmp, 'w'), indent=1); os.replace(tmp, OUT)


if __name__ == '__main__':
    np.seterr(all='ignore')
    st = json.load(open(OUT)) if os.path.exists(OUT) else {'sigma': SIG, 'done': [], 'runs': [], 'best': None}
    for i, x0 in enumerate(starts()):
        if i in st['done']: continue
        if st.get('cur') == i and st.get('cur_x'): x0 = np.array(st['cur_x'])
        log = []; t0 = time.time()
        for rnd in range(3):
            r = minimize(objective, x0, args=(log,), method='Nelder-Mead', options={'maxfev': MAXEV // 3, 'xatol': 1e-3, 'fatol': 1e-5, 'adaptive': True})
            x0 = r.x; st['cur'] = i; st['cur_x'] = list(map(float, r.x)); save(st)
        k0, lam, M, Lv_sig, c = state(r.x)
        rec = {'start': i, 'opt_gap_over_sigma': float(r.fun), 'x': list(map(float, r.x)), 'k0': k0, 'lam': lam, 'M': M, 'Lv_over_sigma': Lv_sig,
               'Lv_over_ell': Lv_sig * np.sqrt(SIG) / lam, 'c': c, 'seconds': time.time() - t0}
        st['runs'].append(rec); st['done'].append(i); st['cur_x'] = None
        if st['best'] is None or rec['opt_gap_over_sigma'] < st['best']['opt_gap_over_sigma']: st['best'] = rec
        save(st); print('sigma %g start %d: gap %.5f k0 %.3f lam %.3f M %.3g Lv/ell %.3g c %.3f' % (SIG, i, r.fun, k0, lam, M, rec['Lv_over_ell'], c), flush=True)
    b = st['best']; f = gap(np.array(b['x']), fine=2)
    b['final'] = {k: float(f[k]) for k in ('gap_over_sigma', 'y_over_sigma', 'P', 'F', 'P_sharp', 'grid_mass', 'J_vv', 'J_ww', 'N')}
    b['feasible_on_final_grids'] = bool(f['y_over_sigma'] <= 0.5)
    st['finished'] = True; save(st); print('final', SIG, b['final'], flush=True)
