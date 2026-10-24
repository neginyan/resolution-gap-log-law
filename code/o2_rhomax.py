"""Stage O, task O2(a): adversarial MAXIMISATION of the focus-slope ratio rho = -E[u d_w w]/E[w d_w u] among W-smooth states,
i.e. under the constraint y/sigma <= Y (y/sigma = sigma J_ww / j, the W-information of the two-variable form of stage M).
Pendulum, free mixtures (h2_adversarial parametrisation), sigma = 1e-3 and 1e-2, K = 4, 8, 16, Y = 0.3, 1, 3, 10.
Objective: rho - 100 (y/sigma - Y)_+^2 (diagnostics of l_tools.diag, recomputed here together with y/sigma on the same grid).
Starts: (0) the K heaviest components of a stored low-gap state (stage-L polished K = 16/24 or stage-H K = 8), (1..) random
needle-like states near the top (as in l1_rho_max).  Adaptive random hill climbing; per_sd 3 in the search, final per_sd 6 and 8.
Checkpoints (results/o2_ckpt/), restartable.  Second variant (6th argument D list): also d <= D (d = focus distance of stage M), penalty 100 (d - D)_+^2 -- the region that
neither the large-y branch nor the focus-distance branch covers.
Usage: python o2_rhomax.py <steps> <restarts> <K list> <Y list> [workers] [D list]
-> results/o2_rhomax.json"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from h2_adversarial import unpack
from j2_smooth import arrays


def diag_y(ws, mus, Ss, sig, per_sd=3):
    """l_tools.diag (same formulas) plus y/sigma = sigma J_ww / j and the parts of Theta' = geo (1 - rho) sqrt(J_ww / E[w s_w^2])."""
    A = arrays(ws, mus, Ss, sig, per_sd=per_sd, max_points=int(min(2_500_000, 2.4e7 / len(ws)))); p = A['pw']; s2 = A['s2']
    w, u, sw, dwu = A['w'], A['u'], A['sw'], A['dwu']; Jww = np.sum(p * A['Hww']); j = np.sum(p * w)
    Lam = -np.sum(p * w * u * sw); Ews2 = np.sum(p * w * sw ** 2); geo = np.sum(p * w * dwu); dww = np.gradient(w, A['hw'], axis=1)
    slope = np.sum(p * u * dww); d2 = np.sum(p * w * u * u) / (j * sig)
    return {'R': A['R'], 'gap': (1 - A['R']) / sig, 'rho': -slope / geo, 'geo': geo / j, 'slope': slope / j, 'ThetaP': abs(Lam) / j * np.sqrt(Jww / Ews2),
            'y_over_sigma': sig * Jww / j, 'd': np.sqrt(d2), 'P': d2 * sig * Jww / j, 'ibp_check': (geo + slope - Lam) / Lam, 'Jww_sigma': Jww * sig, 'j': j}


YMAX = None


def obj(x, K, sig, Y, D=None):
    sh = x[3 * K:].reshape(K, 3)
    if np.any(np.abs(x[K:3 * K]) > 30) or np.any(sh[:, [0, 2]] > 3) or np.any(sh[:, [0, 2]] < -12) or np.any(np.abs(sh[:, 1]) > 8) or np.any(np.abs(x[:K]) > 15):
        return -1e6, None
    try: d = diag_y(*unpack(x, K, sig), sig)
    except (MemoryError, ValueError, np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError): return -1e6, None
    v = d['rho'] - 100 * max(0.0, d['y_over_sigma'] - Y) ** 2 - (0 if D is None else 100 * max(0.0, d['d'] - D) ** 2)
    return (v if np.isfinite(v) else -1e6), d


def stored(K, sig):
    for f in [f'results/l1_polish_sigma{sig:g}.json', 'results/l1_polish_sigma0.001.json']:
        if os.path.exists(f):
            for r in json.load(open(f)):
                Kr = r['K']; x = np.array(r['x'])
                if Kr >= K:
                    o = np.argsort(x[:Kr])[::-1][:K]
                    return np.concatenate([x[:Kr][o], x[Kr:3 * Kr].reshape(Kr, 2)[o].ravel(), x[3 * Kr:].reshape(Kr, 3)[o].ravel()])
    c = json.load(open('results/h4_growk.json')); b = max([x for ch in c for x in ch['chain'] if x['K'] == 8], key=lambda x: x['gstar10']); x = np.array(b['x'])
    o = np.argsort(x[:8])[::-1][:K]; return np.concatenate([x[:8][o], x[8:24].reshape(8, 2)[o].ravel(), x[24:].reshape(8, 3)[o].ravel()])


def job(a):
    K, sig, Y, steps, restarts, D = a; tag = f'K{K}_s{sig:g}_Y{Y:g}' + ('' if D is None else f'_D{D:g}'); ck = f'results/o2_ckpt/{tag}.json'; os.makedirs('results/o2_ckpt', exist_ok=True)
    st = json.load(open(ck)) if os.path.exists(ck) else {'done': [], 'best': None}
    rng0 = np.random.default_rng(1000 + K * 7 + int(Y * 10) + (0 if sig < 5e-3 else 500))
    seeds = rng0.integers(0, 2 ** 31, restarts)
    for rs in range(restarts):
        if rs in st['done']: continue
        rng = np.random.default_rng(int(seeds[rs]))
        x = stored(K, sig) if rs == 0 else np.concatenate([rng.normal(0, 1, K), rng.normal(0, 1.0, 2 * K), np.ravel([[rng.uniform(-4, 1), rng.normal(0, 1), rng.uniform(-6, -1)] for _ in range(K)])])
        fx, dx = obj(x, K, sig, Y, D); x0v = fx; step = 0.3; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy, dy = obj(y, K, sig, Y, D)
            if fy > fx: x, fx, dx = y, fy, dy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.8)); acc = 0
        rec = {'restart': rs, 'start_value': x0v, 'value': fx, 'x': x.tolist()}
        st['done'].append(rs)
        if st['best'] is None or fx > st['best']['value']: st['best'] = rec
        json.dump(st, open(ck, 'w'))
    b = st['best']; x = np.array(b['x']); ws, mus, Ss = unpack(x, K, sig)
    d6 = diag_y(ws, mus, Ss, sig, per_sd=6); d8 = diag_y(ws, mus, Ss, sig, per_sd=8)
    from l_tools import diag
    dl = diag(ws, mus, Ss, sig)
    return {'K': K, 'sigma': sig, 'Y': Y, 'D': D, 'steps': steps, 'restarts': restarts, 'best_restart': b['restart'], 'search_value': b['value'],
            'per_sd6': d6, 'per_sd8': d8, 'bound_over_sigma': dl['bound_over_sigma'], 'rho_ltools': dl['rho'], 'x': b['x']}


if __name__ == '__main__':
    steps, restarts = int(sys.argv[1]), int(sys.argv[2]); Ks = [int(v) for v in sys.argv[3].split(',')]; Ys = [float(v) for v in sys.argv[4].split(',')]
    Wk = int(sys.argv[5]) if len(sys.argv) > 5 else 1; Ds = [float(v) for v in sys.argv[6].split(',')] if len(sys.argv) > 6 else [None]
    fn = 'results/o2_rhomax.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(o['K'], o['sigma'], o['Y'], o.get('D')) for o in out}
    jobs = [(K, s, Y, steps, restarts, D) for D in Ds for K in Ks for Y in Ys for s in [1e-3, 1e-2] if (K, s, Y, D) not in done]
    with Pool(Wk) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r); json.dump(out, open(fn, 'w'), indent=1, default=float); d = r['per_sd6']
            print(f"K={r['K']} sigma {r['sigma']:g} Y {r['Y']:g} D {r['D']}: rho {d['rho']:.4f} (per_sd8 {r['per_sd8']['rho']:.4f})  y/sigma {d['y_over_sigma']:.3f}  geo {d['geo']:.4f}  "
                  f"gap {d['gap']:.4f}  Theta' {d['ThetaP']:.4f}  d {d['d']:.3f}  P {d['P']:.4f}", flush=True)
