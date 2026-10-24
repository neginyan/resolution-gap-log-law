"""Stage P, task P3(a): maximise rho in the box {d <= D, y/sigma <= Y} with the constraints held EXACTLY.
Objective (to maximise): rho if the state is in the box, otherwise -1000 - (violation): every infeasible state is worse than every
feasible one, so the hill climb first walks into the box and then never leaves it; the reported state satisfies the constraints
with a margin of 0.1 % on the search grid (per_sd 3) and is re-checked on the fine grids (per_sd 6 and 8) -- a state that leaves the box on a fine grid is
reported separately as 'reference only'.
Pendulum, sigma = 1e-3 and 1e-2, K = 4, 8, 16, 32, box (D, Y) = (2, 0.3) (and (2, 0.5), (2.5, 0.3) for K <= 16).
Starts: (0) the stage-O box maximiser for the same box and sigma with K' <= K (padded with light copies of its components),
(1) the K heaviest components of the stage-L polished state, (2..) random needle-like states.  Adaptive random hill climbing.
Checkpoints results/p3_ckpt/, restartable.   Region "F <= c" (d^2/8 + 2 y/sigma <= c) is given as D = -1, Y = c, e.g. -1:1.0.
Usage: python p3_rhomax.py <steps> <restarts> <K list> <D,Y list e.g. 2:0.3,2:0.5,-1:1.0> [workers]
-> results/p3_rhomax.json"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from h2_adversarial import unpack
from o2_rhomax import diag_y, stored


def viol(d, D, Y):
    """box {d <= D, y/sigma <= Y}; with D < 0 the region {d^2/8 + 2 y/sigma <= Y} (the states where the explicit branch F alone
    gives less than Y), written as D = -1."""
    if D < 0: return max(0.0, d['d'] ** 2 / 8 + 2 * d['y_over_sigma'] - Y)
    return max(0.0, d['d'] - D) + max(0.0, d['y_over_sigma'] - Y)


def obj(x, K, sig, D, Y, per_sd=3):
    sh = x[3 * K:].reshape(K, 3)
    if np.any(np.abs(x[K:3 * K]) > 30) or np.any(sh[:, [0, 2]] > 3) or np.any(sh[:, [0, 2]] < -12) or np.any(np.abs(sh[:, 1]) > 8) or np.any(np.abs(x[:K]) > 15):
        return -1e6, None
    try: d = diag_y(*unpack(x, K, sig), sig, per_sd=per_sd)
    except (MemoryError, ValueError, np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError): return -1e6, None
    if not np.isfinite(d['rho']): return -1e6, None
    v = viol(d, D - 1e-3 * abs(D) if D > 0 else D, Y - 1e-3 * Y)      # search margin 0.1 %: the fine grids differ from per_sd 3 by ~1e-4
    return (d['rho'] if v == 0 else -1000 - v), d


def pad(x, K0, K, rng):
    """embed a K0-component parameter vector into K components (extra components: copies of random ones, weight e^-6 relative)."""
    w = x[:K0]; c = x[K0:3 * K0].reshape(K0, 2); s = x[3 * K0:].reshape(K0, 3); j = rng.integers(0, K0, K - K0)
    return np.concatenate([np.concatenate([w, w.max() - 6 + rng.normal(0, 0.1, K - K0)]), np.concatenate([c, c[j] + rng.normal(0, 0.05, (K - K0, 2))]).ravel(),
                           np.concatenate([s, s[j] + rng.normal(0, 0.05, (K - K0, 3))]).ravel()])


def starts(K, sig, D, Y, rng):
    out = []
    prev = []
    for fn in ['results/o2_rhomax.json', 'results/p3_rhomax.json']:
        try: prev += [o for o in json.load(open(fn)) if ((o.get('D') == D and o['Y'] == Y) or (D < 0 and o.get('D') == 2.0 and o['Y'] == 0.3))
                      and o['sigma'] == sig and o['K'] <= K and o.get('feasible_found', True)]     # region F: start from the box (2, 0.3) maximiser
        except (FileNotFoundError, json.JSONDecodeError): pass          # (the main process may be writing the file)
    if prev:
        o = max(prev, key=lambda o: o['per_sd6']['rho']); x = np.array(o['x']); out.append(('box_maximiser_K%d' % o['K'], x if o['K'] == K else pad(x, o['K'], K, rng)))
    out.append(('stored_heaviest', stored(K, sig) if K <= 16 else pad(stored(16, sig), 16, K, rng)))
    return out


def job(a):
    K, sig, D, Y, steps, restarts = a; tag = f'K{K}_s{sig:g}_D{D:g}_Y{Y:g}'; ck = f'results/p3_ckpt/{tag}.json'; os.makedirs('results/p3_ckpt', exist_ok=True)
    st = json.load(open(ck)) if os.path.exists(ck) else {'done': [], 'best': None, 'all': []}
    seeds = np.random.default_rng(5000 + K * 13 + int(D * 10) + int(Y * 100) + (0 if sig < 5e-3 else 999)).integers(0, 2 ** 31, restarts)
    S0 = starts(K, sig, D, Y, np.random.default_rng(int(seeds[0])))
    for rs in range(restarts):
        if rs in st['done']: continue
        rng = np.random.default_rng(int(seeds[rs]))
        if rs < len(S0): src, x = S0[rs]
        else: src = 'random'; x = np.concatenate([rng.normal(0, 1, K), rng.normal(0, 0.5, 2 * K), np.ravel([[rng.uniform(-4, 1), rng.normal(0, 1), rng.uniform(-6, -1)] for _ in range(K)])])
        fx, dx = obj(x, K, sig, D, Y); v0 = fx; step = 0.2; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy, dy = obj(y, K, sig, D, Y)
            if fy > fx: x, fx, dx = y, fy, dy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.8)); acc = 0
        st['done'].append(rs); st['all'].append({'restart': rs, 'src': src, 'start_value': v0, 'value': fx})
        if fx > -1000 and (st['best'] is None or fx > st['best']['value']): st['best'] = {'restart': rs, 'src': src, 'value': fx, 'x': x.tolist()}
        json.dump(st, open(ck, 'w'))
    if st['best'] is None: return {'K': K, 'sigma': sig, 'D': D, 'Y': Y, 'feasible_found': False, 'all_restarts': st['all']}
    b = st['best']; x = np.array(b['x']); ws, mus, Ss = unpack(x, K, sig)
    d3 = diag_y(ws, mus, Ss, sig, per_sd=3); d6 = diag_y(ws, mus, Ss, sig, per_sd=6); d8 = diag_y(ws, mus, Ss, sig, per_sd=8)
    from l_tools import diag
    dl = diag(ws, mus, Ss, sig)
    fine_ok = viol(d6, D, Y) == 0 and viol(d8, D, Y) == 0
    return {'K': K, 'sigma': sig, 'D': D, 'Y': Y, 'feasible_found': True, 'feasible_on_fine_grids': fine_ok, 'steps': steps, 'restarts': restarts,
            'best_src': b['src'], 'all_restarts': st['all'], 'per_sd3': d3, 'per_sd6': d6, 'per_sd8': d8, 'bound_over_sigma': dl['bound_over_sigma'], 'x': b['x']}


if __name__ == '__main__':
    steps, restarts = int(sys.argv[1]), int(sys.argv[2]); Ks = [int(v) for v in sys.argv[3].split(',')]
    boxes = [tuple(float(u) for u in v.split(':')) for v in sys.argv[4].split(',')]; Wk = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    fn = 'results/p3_rhomax.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(o['K'], o['sigma'], o['D'], o['Y']) for o in out}
    jobs = [(K, s, D, Y, steps, restarts) for (D, Y) in boxes for K in Ks for s in [1e-3, 1e-2] if (K, s, D, Y) not in done]
    with Pool(Wk) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r); json.dump(out, open(fn, 'w'), indent=1, default=float)
            if r['feasible_found']:
                d = r['per_sd6']; print(f"K={r['K']} sigma {r['sigma']:g} box ({r['D']:g}, {r['Y']:g}): rho {d['rho']:.4f} (per_sd3 {r['per_sd3']['rho']:.4f}, 8 {r['per_sd8']['rho']:.4f})  d {d['d']:.4f}  "
                      f"y/sigma {d['y_over_sigma']:.4f}  fine-grid feasible {r['feasible_on_fine_grids']}  gap {d['gap']:.4f}  geo {d['geo']:.4f}  from {r['best_src']}", flush=True)
            else: print(f"K={r['K']} sigma {r['sigma']:g} box ({r['D']:g}, {r['Y']:g}): no feasible state found", flush=True)
