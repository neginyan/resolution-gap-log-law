"""Stage U1: u_tools.u_terms (pieces of a I(p~), Dbar, coupling bound of Lambda) on every state of t1_chain (361 stored + adversarial)
and on the stage-T maximisers of Lambda / a I(p~)  -> results/u1_chain.json (restartable by key).  Python 3.11 compatible."""
import json, os, glob, time
import numpy as np
from s_tools import fields
from u_tools import u_terms
from h2_adversarial import unpack

if __name__ == '__main__':
    fn = 'results/u1_chain.json'; out = json.load(open(fn)) if os.path.exists(fn) else []; done = {o['key'] for o in out}
    from q2_uncert import all_states
    jobs = [('stored%d' % i, tag, ws, mus, Ss, sig) for i, (tag, ws, mus, Ss, sig) in enumerate(all_states())]
    for f in sorted(glob.glob('results/s1_psharp*_K*_s*.json') + glob.glob('results/t4_*.json') + glob.glob('results/t1_max_*_K*_s*.json') + glob.glob('results/u1_relax_aIpt_K*_s*_Y*.json')):
        o = json.load(open(f))
        if o.get('feasible_found') and 'x' in o: jobs.append((os.path.basename(f), os.path.basename(f), *unpack(np.array(o['x']), o['K'], o['sigma']), o['sigma']))
    yk = {o['key']: o.get('y_over_sigma') for o in json.load(open('results/t1_chain.json')) if 'error' not in o}
    t0 = time.time()
    for k, (key, tag, ws, mus, Ss, sig) in enumerate(jobs):
        if key in done: continue
        try:
            f = fields(ws, mus, Ss, sig, 6); q = u_terms(f); P = f['pw']; s2 = sig * sig
            q['y_over_sigma'] = sig * float(np.sum(P * f['sw'] ** 2)) / float(np.sum(P * np.clip(s2 * np.maximum(f['Hvv'], 0), 0, 1))); q['grid_mass'] = f['mass']
        except (MemoryError, ValueError, np.linalg.LinAlgError) as e: q = {'error': str(e)}
        q.update({'key': key, 'tag': tag, 'sigma': sig, 'K': len(ws)}); out.append(q)
        if k % 20 == 0 or k == len(jobs) - 1: json.dump(out, open(fn, 'w'), indent=1, default=float); print(k + 1, len(jobs), '(%.0f s)' % (time.time() - t0), flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=float)
