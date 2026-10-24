"""Stage U2: u_tools.x_terms (exact identity for x - 1 and the new bound without integration by parts against p~) on every state of
u1_chain, plus the two-cluster family (sd_p~(mu)/sigma unbounded at fixed y/sigma)  -> results/u2_chain.json (restartable).  Python 3.11."""
import json, os, glob, time
import numpy as np
from s_tools import fields
from t_tools import t_terms
from u_tools import x_terms
from h2_adversarial import unpack

V = np.array([1, 1]) / np.sqrt(2); W = np.array([1, -1]) / np.sqrt(2); TOP = np.array([np.pi, 0.0])


def two_clusters(V0, sig, ell_s=2.0, sep=4.0):
    ell = ell_s * np.sqrt(sig); W0 = sep * ell; S = (0.5 * sig) ** 2 * np.outer(V, V) + ell ** 2 * np.outer(W, W)
    return [0.5, 0.5], [TOP + V0 * sig * V + W0 * W, TOP - V0 * sig * V - W0 * W], [S, S]


if __name__ == '__main__':
    fn = 'results/u2_chain.json'; out = json.load(open(fn)) if os.path.exists(fn) else {'states': [], 'two_clusters': []}; done = {o['key'] for o in out['states']}
    from q2_uncert import all_states
    jobs = [('stored%d' % i, ws, mus, Ss, sig) for i, (tag, ws, mus, Ss, sig) in enumerate(all_states())]
    for f in sorted(glob.glob('results/s1_psharp*_K*_s*.json') + glob.glob('results/t4_*.json') + glob.glob('results/t1_max_*_K*_s*.json') + glob.glob('results/u1_relax_aIpt_K*_s*_Y*.json')):
        o = json.load(open(f))
        if o.get('feasible_found') and 'x' in o: jobs.append((os.path.basename(f), *unpack(np.array(o['x']), o['K'], o['sigma']), o['sigma']))
    t0 = time.time()
    for k, (key, ws, mus, Ss, sig) in enumerate(jobs):
        if key in done: continue
        try:
            f = fields(ws, mus, Ss, sig, 6); t = t_terms(f); q = x_terms(f)
            q.update({'x_minus_1_T': t['x_minus_1'], 'x_bound_ptilde_T': t['x_bound_ptilde'], 'y_over_sigma': t['y_over_sigma'], 'Lambda': t['Lambda']})
        except (MemoryError, ValueError, np.linalg.LinAlgError) as e: q = {'error': str(e)}
        q.update({'key': key, 'sigma': sig, 'K': len(ws)}); out['states'].append(q)
        if k % 20 == 0: json.dump(out, open(fn, 'w'), indent=1, default=float); print(k + 1, len(jobs), '(%.0f s)' % (time.time() - t0), flush=True)
    if not out['two_clusters']:
        for sig in (1e-2, 1e-3):
            for V0 in (2, 5, 20, 60):
                f = fields(*two_clusters(V0, sig), sig, 6); t = t_terms(f); q = x_terms(f)
                q.update({'V0_over_sigma': V0, 'sigma': sig, 'y_over_sigma': t['y_over_sigma'], 'x_bound_ptilde_T': t['x_bound_ptilde'], 'P_sharp': t['P_sharp'], 'grid_mass': f['mass']})
                out['two_clusters'].append(q)
    json.dump(out, open(fn, 'w'), indent=1, default=float)
