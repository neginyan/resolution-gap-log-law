"""Stage T1: the absolute-variance form of the chain (t_tools.t_terms) on every stored state (361) and on every adversarial
state of stages S and T (results/s1_psharp*.json, results/t4_*.json)  -> results/t1_chain.json (restartable, one output file)."""
import json, os, sys, glob, time
import numpy as np
from t_tools import all_t
from h2_adversarial import unpack

if __name__ == '__main__':
    fn = 'results/t1_chain.json'; out = json.load(open(fn)) if os.path.exists(fn) else []; done = {o['key'] for o in out}
    from q2_uncert import all_states
    jobs = [(f'stored{i}', tag, ws, mus, Ss, sig) for i, (tag, ws, mus, Ss, sig) in enumerate(all_states())]
    for f in sorted(glob.glob('results/s1_psharp*_K*_s*.json') + glob.glob('results/t4_*.json')):
        o = json.load(open(f))
        if o.get('feasible_found') and 'x' in o: jobs.append((os.path.basename(f), os.path.basename(f), *unpack(np.array(o['x']), o['K'], o['sigma']), o['sigma']))
    t0 = time.time()
    for k, (key, tag, ws, mus, Ss, sig) in enumerate(jobs):
        if key in done: continue
        try: q = all_t(ws, mus, Ss, sig)
        except (MemoryError, ValueError, np.linalg.LinAlgError) as e: q = {'error': str(e)}
        q.update({'key': key, 'tag': tag, 'sigma': sig, 'K': len(ws)}); out.append(q)
        if k % 20 == 0 or k == len(jobs) - 1: json.dump(out, open(fn, 'w'), indent=1, default=float); print(k + 1, len(jobs), f'({time.time() - t0:.0f} s)', flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=float)
