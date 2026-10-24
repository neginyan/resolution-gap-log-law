"""Stage S1 (3): adversarial MINIMISATION of the weight-free score moment
    P# = sigma^2 E[u^2 s_v^2] J_ww / (sigma^2 J_vv)^2
over K-component Gaussian mixtures (pendulum, parametrisation of h2_adversarial.unpack), with the single constraint y/sigma <= 0.5
(y/sigma = sigma J_ww / j, j = E[sigma^2 (H_vv)_+ clipped to 1], as in stages N-R), held EXACTLY: infeasible states score 1000 + violation;
the final state is re-evaluated on the grids per_sd = 6 and 8 and counts only if the constraint holds there too.
K = 8, 16, 32; sigma = 1e-3, 1e-2.  Starts: the stage-L polished state, the stage-Q theta minimiser, the N2 P-minimiser, the Q2
F-minimiser (padded to K), random needle-like states, 'hourglass' states (thin waist at the top, wide-in-V flares along W) and
'comb' states (thin components spaced < sigma in V).  Checkpoint per restart: results/s1_ckpt/<tag>.json.
For the final state: the full line-reduction chain (s_tools.all_terms) and the gap (l_tools.diag).
Usage: python s1_psharp_min.py <steps> <restarts> <K list> <sigma>   -> results/s1_psharp_K<K>_s<sigma>.json (one file per job)"""
import numpy as np, json, os, sys, time, glob
from h2_adversarial import unpack
from o2_rhomax import stored
from p3_rhomax import pad
from s_tools import fields, all_terms

Y0 = 0.5


def quick(ws, mus, Ss, sig, per_sd=3, max_points=300_000):
    f = fields(ws, mus, Ss, sig, per_sd=per_sd, max_points=max_points); s2 = sig * sig; P = f['pw']; E = lambda a: float(np.sum(P * a))   # search grid <= 3e5 points (final: full grids)
    sv, sw, u = f['sv'], f['sw'], f['u']; Jvv, Jww = E(sv * sv), E(sw * sw); a = s2 * Jvv; j = E(np.clip(s2 * np.maximum(f['Hvv'], 0), 0, 1))
    return {'P_sharp': s2 * E(u * u * sv * sv) * Jww / a ** 2, 'y_over_sigma': sig * Jww / j, 'grid_mass': f['mass']}


def obj(x, K, sig):
    sh = x[3 * K:].reshape(K, 3)
    if np.any(np.abs(x[K:3 * K]) > 30) or np.any(sh[:, [0, 2]] > 4) or np.any(sh[:, [0, 2]] < -12) or np.any(np.abs(sh[:, 1]) > 8) or np.any(np.abs(x[:K]) > 15):
        return 1e6, None
    try: q = quick(*unpack(x, K, sig), sig)
    except (MemoryError, ValueError, np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError): return 1e6, None
    if not (np.isfinite(q['P_sharp']) and np.isfinite(q['y_over_sigma'])): return 1e6, None
    if abs(q['grid_mass'] - 1) > 1e-4: return 1e6, None           # under-resolved on the (capped) search grid: rejected (stage S guard)
    v = max(0.0, q['y_over_sigma'] - Y0 * (1 - 1e-3)); return (q['P_sharp'] if v == 0 else 1000 + v), q


def structured(kind, K, rng):
    w = rng.normal(0, 0.3, K); c = np.zeros((K, 2)); s = np.zeros((K, 3))
    if kind == 'hourglass':        # component 0: thin waist at the top; the others: wide in V, placed along W on both sides
        w[0] += 1.5; s[0] = [-3, 0, rng.uniform(-1, 0.5)]
        for k in range(1, K):
            c[k] = [rng.normal(0, 1), rng.choice([-1, 1]) * rng.uniform(0.5, 3)]; s[k] = [rng.uniform(1, 3.5), rng.normal(0, 0.3), rng.uniform(-2, 0)]
    else:                          # comb: thin components spaced 0.3-0.8 sigma in V, a common W extent, plus a wide background
        dv = rng.uniform(0.3, 0.8)
        for k in range(K - 1): c[k] = [(k - (K - 2) / 2) * dv, rng.normal(0, 0.2)]; s[k] = [-4, 0, rng.uniform(-0.5, 0.5)]
        c[K - 1] = [0, 0]; s[K - 1] = [rng.uniform(1, 3), 0, rng.uniform(-0.5, 0.5)]; w[K - 1] += 1.0
    return np.concatenate([w, c.ravel(), s.ravel()])


def starts(K, sig, rng):
    x0 = np.asarray(stored(K, sig)); K0 = len(x0) // 6
    out = [('stage-L polished (stored)' + ('' if K0 == K else f' K{K0} padded'), x0 if K0 == K else pad(x0, K0, K, rng))]
    for fn, sel in [('results/q2_theta.json', lambda o: o.get('feasible_found') and 'x' in o), ('results/n2_pmin.json', lambda o: o.get('ymax') is not None),
                    ('results/q2_fmin.json', lambda o: 'x' in o)]:
        try:
            L = [o for o in json.load(open(fn)) if o['sigma'] == sig and o['K'] <= K and sel(o)]
            if L: o = max(L, key=lambda o: o['K']); x = np.array(o['x']); out.append((fn.split('/')[-1][:-5] + f"_K{o['K']}", x if o['K'] == K else pad(x, o['K'], K, rng)))
        except (FileNotFoundError, json.JSONDecodeError, KeyError): pass
    out += [('hourglass', structured('hourglass', K, rng)), ('comb', structured('comb', K, rng))]
    return out


def job(K, sig, steps, restarts):
    tag = f'psharp_K{K}_s{sig:g}'; ck = f'results/s1_ckpt/{tag}.json'; os.makedirs('results/s1_ckpt', exist_ok=True)
    st = json.load(open(ck)) if os.path.exists(ck) else {'done': [], 'best': None, 'all': []}
    seeds = np.random.default_rng(12000 + K + (0 if sig < 5e-3 else 777)).integers(0, 2 ** 31, restarts); S0 = starts(K, sig, np.random.default_rng(int(seeds[0])))
    for rs in range(restarts):
        if rs in st['done']: continue
        rng = np.random.default_rng(int(seeds[rs])); t0 = time.time()
        if rs < len(S0): src, x = S0[rs]
        elif rs % 3 == 0: src, x = 'hourglass', structured('hourglass', K, rng)
        elif rs % 3 == 1: src, x = 'comb', structured('comb', K, rng)
        else: src = 'random'; x = np.concatenate([rng.normal(0, 1, K), rng.normal(0, 0.5, 2 * K), np.ravel([[rng.uniform(-4, 1), rng.normal(0, 1), rng.uniform(-6, -1)] for _ in range(K)])])
        fx, qx = obj(x, K, sig); v0 = fx; step = 0.2; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy, qy = obj(y, K, sig)
            if fy < fx: x, fx, qx = y, fy, qy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.8)); acc = 0
        st['done'].append(rs); st['all'].append({'restart': rs, 'src': src, 'start_value': v0, 'value': fx, 'seconds': time.time() - t0})
        if fx < 1000 and (st['best'] is None or fx < st['best']['value']): st['best'] = {'restart': rs, 'src': src, 'value': fx, 'x': x.tolist()}
        json.dump(st, open(ck, 'w')); print(tag, rs, src, round(v0, 4), '->', round(fx, 4), f'({time.time() - t0:.0f} s)', flush=True)
    if st['best'] is None: return {'K': K, 'sigma': sig, 'feasible_found': False, 'all_restarts': st['all']}
    b = st['best']; x = np.array(b['x']); ws, mus, Ss = unpack(x, K, sig); q6 = all_terms(ws, mus, Ss, sig, per_sd=6); q8 = all_terms(ws, mus, Ss, sig, per_sd=8)
    from l_tools import diag
    dl = diag(ws, mus, Ss, sig)
    return {'K': K, 'sigma': sig, 'feasible_found': True, 'feasible_on_fine_grids': q6['y_over_sigma'] <= Y0 and q8['y_over_sigma'] <= Y0, 'steps': steps, 'restarts': restarts,
            'best_src': b['src'], 'all_restarts': st['all'], 'per_sd6': q6, 'per_sd8': q8, 'gap': (1 - dl['R']) / sig, 'rho': dl['rho'], 'x': b['x'],
            'components': [{'weight': float(w_), 'mean_v_over_sigma': float((m - np.array([np.pi, 0])) @ np.array([1, 1]) / np.sqrt(2) / sig),
                            'mean_w_over_sqrt_sigma': float((m - np.array([np.pi, 0])) @ np.array([1, -1]) / np.sqrt(2) / np.sqrt(sig)),
                            'sd_major_over_sqrt_sigma': float(np.sqrt(np.linalg.eigvalsh(S)[-1] / sig)), 'sd_minor_over_sigma': float(np.sqrt(max(np.linalg.eigvalsh(S)[0], 0)) / sig)}
                           for w_, m, S in zip(ws, mus, Ss)]}


def refine(K, sig, steps, restarts=2):
    """continue from the best state found so far for any K' <= K at this sigma (padded to K): results/s1_psharp_refine_K<K>_s<sigma>.json"""
    best = None
    for f in glob.glob(f'results/s1_psharp*_K*_s{sig:g}.json'):
        o = json.load(open(f))
        if o.get('feasible_found') and o.get('feasible_on_fine_grids') and o['K'] <= K and (best is None or o['per_sd6']['P_sharp'] < best['per_sd6']['P_sharp']): best = o
    rng = np.random.default_rng(31000 + K + (0 if sig < 5e-3 else 555)); x0 = np.array(best['x']); x0 = x0 if best['K'] == K else pad(x0, best['K'], K, rng)
    st = {'all': [], 'best': None}
    for rs in range(restarts):
        x = x0.copy(); fx, qx = obj(x, K, sig); v0 = fx; step = 0.05 * (rs + 1); acc = 0; t0 = time.time()
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy, qy = obj(y, K, sig)
            if fy < fx: x, fx, qx = y, fy, qy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.8)); acc = 0
        st['all'].append({'restart': rs, 'src': f"refine from K{best['K']} best", 'start_value': v0, 'value': fx, 'seconds': time.time() - t0})
        if fx < 1000 and (st['best'] is None or fx < st['best']['value']): st['best'] = {'restart': rs, 'src': f"refine from K{best['K']}", 'value': fx, 'x': x.tolist()}
        print('refine', K, sig, rs, round(v0, 4), '->', round(fx, 4), flush=True)
    b = st['best']; x = np.array(b['x']); ws, mus, Ss = unpack(x, K, sig); q6 = all_terms(ws, mus, Ss, sig, per_sd=6); q8 = all_terms(ws, mus, Ss, sig, per_sd=8)
    from l_tools import diag
    dl = diag(ws, mus, Ss, sig)
    return {'K': K, 'sigma': sig, 'feasible_found': True, 'feasible_on_fine_grids': q6['y_over_sigma'] <= Y0 and q8['y_over_sigma'] <= Y0, 'steps': steps, 'restarts': restarts,
            'best_src': b['src'], 'all_restarts': st['all'], 'per_sd6': q6, 'per_sd8': q8, 'gap': (1 - dl['R']) / sig, 'rho': dl['rho'], 'x': b['x'], 'components': []}


if __name__ == '__main__':
    if sys.argv[1] == 'refine':
        K, sig, steps = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
        sfx = sys.argv[5] if len(sys.argv) > 5 else ''; r = refine(K, sig, steps); r['label'] = 'refine' + sfx
        json.dump(r, open(f'results/s1_psharp_refine{sfx}_K{K}_s{sig:g}.json', 'w'), indent=1, default=float)
        print('refine done', K, sig, r['per_sd6']['P_sharp'], r['per_sd8']['P_sharp'], r['per_sd6']['y_over_sigma'], r['feasible_on_fine_grids'], flush=True); sys.exit(0)
    steps, restarts = int(sys.argv[1]), int(sys.argv[2]); Ks = [int(v) for v in sys.argv[3].split(',')]; sig = float(sys.argv[4])
    for K in Ks:
        fn = f'results/s1_psharp_K{K}_s{sig:g}.json'
        if os.path.exists(fn): continue
        r = job(K, sig, steps, restarts); json.dump(r, open(fn, 'w'), indent=1, default=float)
        if r['feasible_found']:
            q = r['per_sd6']; print(f"K={K} sigma {sig:g}: P# {q['P_sharp']:.4f} (8: {r['per_sd8']['P_sharp']:.4f})  P {q['P']:.4f}  y/sigma {q['y_over_sigma']:.4f}  "
                                     f"chain x {q['X_over_a']:.4f} Gamma_eff {q['Gamma_eff']:.3f} bound_sharp {q['bound_sharp']:.4f}  gap {r['gap']:.4f}  fine-feasible {r['feasible_on_fine_grids']}", flush=True)
