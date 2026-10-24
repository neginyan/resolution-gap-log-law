"""Stage T1 (iii): adversarial MAXIMISATION of Lambda = sigma^2 E[(s_v s_w - 2 H_vw)^2] / J_ww (the one constant left in the
chain P# >= x^2 / Lambda) and of a I(p~) / J_ww, in the region y/sigma <= 0.5 (held exactly; grid guard of stage S; checkpoint per
restart).  Starts: stage-S hourglass, stage-L state, tilted-grating states, random.  Final state: per_sd 6 and 8.
Usage: python t1_lambda_max.py <target: Lambda|aIpt> <K> <sigma> <steps> <restarts>  -> results/t1_max_<target>_K<K>_s<sigma>.json"""
import numpy as np, json, os, sys, time
from h2_adversarial import unpack
from p3_rhomax import pad
from s_tools import fields
from t_tools import t_terms, t_all

Y0 = 0.5


def val(x, K, sig, target):
    sh = x[3 * K:].reshape(K, 3)
    if np.any(np.abs(x[K:3 * K]) > 30) or np.any(sh[:, [0, 2]] > 4) or np.any(sh[:, [0, 2]] < -12) or np.any(np.abs(sh[:, 1]) > 8) or np.any(np.abs(x[:K]) > 15): return -1e6
    try:
        f = fields(*unpack(x, K, sig), sig, per_sd=3, max_points=300_000)
        if abs(f['mass'] - 1) > 1e-4: return -1e6
        t = t_terms(f); P = f['pw']; s2 = sig * sig; j = float(np.sum(P * np.clip(s2 * np.maximum(f['Hvv'], 0), 0, 1)))
        y = sig * t['Jww'] / j
    except (MemoryError, ValueError, np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError): return -1e6
    v = t['Lambda'] if target == 'Lambda' else t['aIpt_over_Jww']
    if not np.isfinite(v): return -1e6
    return v if y <= Y0 * (1 - 1e-3) else -1000 - (y - Y0)


SIG = 1e-2


def start(rs, K, rng):
    if rs == 0:
        o = json.load(open('results/s1_psharp_refine_long_K16_s0.01.json')); x = np.array(o['x']); return 'hourglass (stage S)', x if o['K'] == K else (pad(x, o['K'], K, rng) if K > o['K'] else None)
    if rs == 1:
        from o2_rhomax import stored
        x = np.asarray(stored(K, 1e-3)); K0 = len(x) // 6; return 'stage-L state', x if K0 == K else pad(x, K0, K, rng)
    if rs == 2:     # tilted grating: thin strips along W + theta V, spacing 2.5-3 sigma
        w = np.zeros(K); c = np.zeros((K, 2)); s = np.zeros((K, 3)); th = rng.uniform(0.02, 0.1)
        for k in range(K): c[k] = [(k - (K - 1) / 2) * 2.8, 0]; s[k] = [np.log(2 * th / np.sqrt(SIG)), 1.0, -8.0]   # Cholesky of l^2 t t^T + eps^2 n n^T, l = 2 sqrt(sigma), t ~ W + th V
        return 'tilted grating', np.concatenate([w, c.ravel(), s.ravel()])
    return 'random', np.concatenate([rng.normal(0, 1, K), rng.normal(0, 0.5, 2 * K), np.ravel([[rng.uniform(-4, 1), rng.normal(0, 1), rng.uniform(-3, 0)] for _ in range(K)])])


if __name__ == '__main__':
    target, K, sig, steps, restarts = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); SIG = sig
    tag = f'{target}_K{K}_s{sig:g}'; ck = f'results/t1_ckpt/{tag}.json'; os.makedirs('results/t1_ckpt', exist_ok=True)
    st = json.load(open(ck)) if os.path.exists(ck) else {'done': [], 'all': [], 'best': None}
    for rs in range(restarts):
        if rs in st['done']: continue
        rng = np.random.default_rng(51000 + 13 * rs + K); src, x = start(rs, K, rng)
        if x is None: st['done'].append(rs); continue
        fx = val(x, K, sig, target); v0 = fx; step = 0.2; acc = 0; t0 = time.time()
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy = val(y, K, sig, target)
            if fy > fx: x, fx = y, fy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.8)); acc = 0
        st['done'].append(rs); st['all'].append({'restart': rs, 'src': src, 'start': v0, 'value': fx, 'seconds': time.time() - t0})
        if fx > -1000 and (st['best'] is None or fx > st['best']['value']): st['best'] = {'src': src, 'value': fx, 'x': x.tolist()}
        json.dump(st, open(ck, 'w')); print(tag, rs, src, round(v0, 4), '->', round(fx, 4), flush=True)
    out = {'target': target, 'K': K, 'sigma': sig, 'restarts': st['all'], 'feasible_found': st['best'] is not None}
    if st['best']:
        x = np.array(st['best']['x']); ws, mus, Ss = unpack(x, K, sig); q6 = t_all(ws, mus, Ss, sig, 6); q8 = t_all(ws, mus, Ss, sig, 8)
        out.update({'per_sd6': q6, 'per_sd8': q8, 'feasible_on_fine_grids': q6['y_over_sigma'] <= Y0 and q8['y_over_sigma'] <= Y0, 'x': st['best']['x'], 'best_src': st['best']['src']})
        print('done', tag, 'Lambda', round(q6['Lambda'], 4), round(q8['Lambda'], 4), 'aIpt/Jww', round(q6['aIpt_over_Jww'], 4), 'P#', round(q6['P_sharp'], 4),
              'bounds', round(q6['bound_ptilde'], 4), round(q6['bound_Lambda'], 4), 'y/sigma', round(q6['y_over_sigma'], 3), flush=True)
    json.dump(out, open(f'results/t1_max_{tag}.json', 'w'), indent=1, default=float)
