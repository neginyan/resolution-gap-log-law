"""Stage X3: is there a state that goes down faster than the single ripple strip?  Searched with the separate evaluator x_indep (phase-based),
re-evaluated at the end by the stage-W evaluators (w_full true envelope, and the one-period limit + Lv^2/(8 sigma)), which share nothing with it.
usage: python3 x3_search.py FAMILY SIGMA [MAXEVAL_PER_ROUND] -> results/x3_opt_<FAMILY>_s<SIGMA>.json (checkpointed per round, restartable)
families (single Gaussian background in W, Lw = M ell_1, Lv = eta ell_1):
  H2  k and 2k superposed: windows m = 1 and m = 2 (free centres, widths, amplitudes, relative phase)
  H3  harmonics 1, 2, 3
  MW  three windows with the same k (free centres, widths, amplitudes, phases): overlapping windows interfere
  CH  W-varying k0: three windows side by side with m = 2, 3, 4 of a base k_b (free centres)
Positivity sum_r c_r exp(-(w - w_r)^2 / 2 ell_r^2) <= 1;  y/sigma <= 0.5;  Lw <= 2;  Lv >= 30/k sigma.
Start: the single-strip optimum at the same sigma (results/w1_gmin_indep_opt.json), embedded (extra windows with small amplitude) + 2 perturbations.
Python 3.11."""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize
from x_indep import multi_gap, positivity

FAM = sys.argv[1]; SIG = float(sys.argv[2]); MAXR = int(sys.argv[3]) if len(sys.argv) > 3 else 300
OUT = 'results/x3_opt_%s_s%g.json' % (FAM, SIG)
MS = {'H2': [1, 2], 'H3': [1, 2, 3], 'MW': [1, 1, 1], 'CH': [2, 3, 4]}[FAM]; R = len(MS)
B = json.load(open('results/w1_gmin_indep_opt.json'))['%g' % SIG]


def unpack(x):
    kb = np.exp(x[0]); M = np.exp(x[1]); eta = np.exp(x[2]); win = []
    for r in range(R):
        c, wr, le, ph = x[3 + 4 * r: 7 + 4 * r]
        win.append((1 / (1 + np.exp(-c)), wr, np.exp(le), MS[r], ph if r > 0 else 0.0))
    return kb, win, M, eta


def objective(x, log):
    try:
        kb, win, M, eta = unpack(x)
        pos = positivity(win)
        if pos > 1 + 1e-12: return 50.0 + pos
        ell1 = win[0][2]; Lw = M * ell1 * np.sqrt(SIG); Lv_sig = eta * ell1 / np.sqrt(SIG)
        if not (0.3 < kb * max(MS) < 12 and Lw < 2.0 and M > 1 and Lv_sig > 30 / (kb * min(MS)) and all(0.05 < w_[2] < 50 for w_ in win)): return 60.0
        o = multi_gap(SIG, kb, win, M, eta)
        g = o['gap_over_sigma']
        if not np.isfinite(g) or o['y_over_sigma'] > 0.5: return 40.0
        log.append(g); return g
    except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
        return 70.0


def starts():
    k0, lam, M, eta, c = B['k0'], B['lam'], B['M'], B['Lv_over_ell'], min(B['c'], 0.97)
    lg = lambda p: np.log(p / (1 - p))
    if FAM in ('H2', 'H3'):
        x = [np.log(k0), np.log(M), np.log(eta), lg(c * 0.9), 0.0, np.log(lam), 0.0] + sum([[lg(0.03), 0.0, np.log(lam), 0.0] for _ in range(R - 1)], [])
    elif FAM == 'MW':
        x = [np.log(k0), np.log(M), np.log(eta), lg(c * 0.9), 0.0, np.log(lam), 0.0] + [lg(0.03), 2.5 * lam, np.log(lam), 0.0] + [lg(0.03), -2.5 * lam, np.log(lam), np.pi]
    else:   # CH: base k_b = k0/3 so that the middle window has k0
        x = [np.log(k0 / 3), np.log(M), np.log(eta), lg(0.03), -4 * lam, np.log(lam), 0.0] + [lg(c * 0.9), 0.0, np.log(lam), 0.0] + [lg(0.03), 4 * lam, np.log(lam), 0.0]
    x = np.array(x, float); rng = np.random.default_rng(int(1e6 * SIG ** 0.01) % 1000 + R)
    out = [x]
    for i in range(2):
        y = x.copy(); y[3:] += rng.normal(0, 0.6, len(y) - 3); y[3::4] = np.minimum(y[3::4], lg(0.6)); out.append(y)
    return out


def save(st):
    tmp = OUT + '.tmp'; json.dump(st, open(tmp, 'w'), indent=1); os.replace(tmp, OUT)


if __name__ == '__main__':
    np.seterr(all='ignore')
    st = json.load(open(OUT)) if os.path.exists(OUT) else {'family': FAM, 'sigma': SIG, 'm': MS, 'single_strip_gap': B['final_gap'], 'done': [], 'runs': [], 'best': None}
    for i, x0 in enumerate(starts()):
        if i in st['done']: continue
        if st.get('cur') == i and st.get('cur_x'): x0 = np.array(st['cur_x']); rd0 = st.get('cur_round', 0)
        else: rd0 = 0
        log = []; t0 = time.time(); fx = objective(x0, [])
        for rd in range(rd0, 3):
            r = minimize(objective, x0, args=(log,), method='Nelder-Mead', options={'maxfev': MAXR, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            x0, fx = r.x, float(r.fun); st['cur'] = i; st['cur_x'] = list(map(float, x0)); st['cur_round'] = rd + 1; st['cur_gap'] = fx; save(st)
            print('%s %g start %d round %d: %.5f (%.0fs)' % (FAM, SIG, i, rd + 1, fx, time.time() - t0), flush=True)
        kb, win, M, eta = unpack(x0)
        rec = {'start': i, 'opt_gap_over_sigma': fx, 'x': list(map(float, x0)), 'kb': kb, 'M': M, 'eta': eta,
               'windows': [{'c': c, 'w_center': wr, 'ell': er, 'm': m, 'phase': ph, 'k': m * kb} for c, wr, er, m, ph in win]}
        st['runs'].append(rec); st['done'].append(i); st['cur_x'] = None; st['cur_round'] = 0
        if st['best'] is None or fx < st['best']['opt_gap_over_sigma']: st['best'] = rec
        save(st)
    b = st['best']; kb, win, M, eta = unpack(np.array(b['x']))
    f = multi_gap(SIG, kb, win, M, eta, n_theta=48 * max(MS), n_slow=24, n_gh=64, n_gl=16, panels_per_ell=5, bg_panels=96)
    b['final'] = {k: float(f[k]) for k in ('gap_over_sigma', 'y_over_sigma', 'P_sharp', 'a', 'J_ww', 'N')}
    b['positivity_max'] = positivity(win); b['feasible'] = bool(f['y_over_sigma'] <= 0.5 and b['positivity_max'] <= 1 + 1e-12)
    b['vs_single_strip'] = b['final']['gap_over_sigma'] / st['single_strip_gap'] - 1
    st['finished'] = True; save(st); print('final', FAM, SIG, b['final'], 'vs single %+.4f' % b['vs_single_strip'], flush=True)
