"""Stage T, task T1 (ii): how large can Var(phi) be on a line?   (1-d: Y = X + Z, sigma = 1, phi = 2 - 2 E[Z^2|y] + E[Z|y]^2,
omega = E[phi] = E[E[Z|y]^2] = sigma^2 I in (0, 1].)
Proved (main line): Var(phi) <= E[(phi - 2)^2] <= 4 E[E[Z^2|y]^2] <= 4 E[Z^4] = 12.
Here: adversarial MAXIMISATION over K-component Gaussian mixtures (K = 2, 3, 4, 8) of
    'V'      Var(phi)                         (best absolute constant)
    'Vsqrt'  Var(phi) / sqrt(omega)          (does Var(phi) <= C omega^(1/2) hold?)
    'Vlog'   Var(phi) / (omega (1 + log(1/omega)))   (does Var(phi) <= C omega log(e/omega) hold?)
plus the periodic comb family (s1_gamma1d.periodic) scanned over spacing and height for the same three ratios.
Usage: python t1_var1d.py <steps> <restarts> <K list>   -> results/t1_var1d.json"""
import numpy as np, json, os, sys
from s1_gamma1d import unpack, disp, periodic

MODES = {'V': lambda d: d['Var_phi'], 'Vsqrt': lambda d: d['Var_phi'] / np.sqrt(d['omega']),
         'Vlog': lambda d: d['Var_phi'] / (d['omega'] * (1 + np.log(1 / d['omega'])))}


def obj(x, K, mode):
    if np.any(np.abs(x[:K]) > 12) or np.any(x[2 * K:] < -12) or np.any(x[2 * K:] > 8) or np.ptp(x[K:2 * K]) > 60: return -1.0
    d = disp(*unpack(x, K), ny=8001); v = MODES[mode](d)
    return float(v) if np.isfinite(v) else -1.0


def search(K, mode, steps, restarts, seed):
    rng = np.random.default_rng(seed); best = None; allr = []
    for rs in range(restarts):
        x = np.concatenate([rng.normal(0, 1, K), rng.normal(0, 3, K), rng.uniform(-8, 3, K)]); fx = obj(x, K, mode); step = 0.5; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.4); fy = obj(y, K, mode)
            if fy > fx: x, fx = y, fy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 2.0)); acc = 0
        allr.append(fx)
        if best is None or fx > best[0]: best = (fx, x.copy())
    d = disp(*unpack(best[1], K), ny=40001); d2 = disp(*unpack(best[1], K), ny=160001); w, m, t = unpack(best[1], K)
    return {'K': K, 'mode': mode, 'opt_value': MODES[mode](d), 'opt_value_fine': MODES[mode](d2), 'Var_phi': d['Var_phi'], 'omega': d['omega'],
            'restarts': allr, 'w': w.tolist(), 'm': (m - m.mean()).tolist(), 't': t.tolist(), 'x': best[1].tolist()}


def comb_scan():
    out = []
    for Dl in [8, 6, 5, 4, 3.5, 3, 2.5, 2.25, 2, 1.75, 1.5]:
        for l in np.exp(np.linspace(-8, 8, 65)):
            r = periodic(Dl, l); r['Var_phi'] = r['D'] * r['E_phi']; out.append(r)
    return out


if __name__ == '__main__':
    steps, restarts = int(sys.argv[1]), int(sys.argv[2]); Ks = [int(v) for v in sys.argv[3].split(',')]
    fn = 'results/t1_var1d.json'; out = json.load(open(fn)) if os.path.exists(fn) else {'searches': [], 'comb': None}
    if out['comb'] is None:
        C = comb_scan(); out['comb'] = C
        out['comb_max'] = {m: max(MODES[m](dict(c, omega=c['omega'])) for c in C if c['omega'] > 0) for m in MODES}
        json.dump(out, open(fn, 'w'), indent=1, default=float); print('comb', out['comb_max'], flush=True)
    for mode in MODES:
        for K in Ks:
            if any(o['K'] == K and o['mode'] == mode for o in out['searches']): continue
            o = search(K, mode, steps, restarts, 700 + K + 10 * list(MODES).index(mode)); out['searches'].append(o)
            print(mode, K, round(o['opt_value'], 4), round(o['opt_value_fine'], 4), 'omega', round(o['omega'], 5), flush=True)
            json.dump(out, open(fn, 'w'), indent=1, default=float)
