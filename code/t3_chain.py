"""Stage T3: correction factors (t_tools.t3_terms) on every stored state and every adversarial state -> results/t3_chain.json"""
import json, os, glob, time
import numpy as np
from t_tools import t3_terms
from h2_adversarial import unpack

if __name__ == '__main__':
    fn = 'results/t3_chain.json'; out = json.load(open(fn)) if os.path.exists(fn) else []; done = {o['key'] for o in out}
    from q2_uncert import all_states
    jobs = [(f'stored{i}', ws, mus, Ss, sig) for i, (tag, ws, mus, Ss, sig) in enumerate(all_states())]
    for f in sorted(glob.glob('results/s1_psharp*_K*_s*.json') + glob.glob('results/t4_*.json')):
        o = json.load(open(f))
        if o.get('feasible_found') and 'x' in o: jobs.append((os.path.basename(f), *unpack(np.array(o['x']), o['K'], o['sigma']), o['sigma']))
    t0 = time.time()
    for k, (key, ws, mus, Ss, sig) in enumerate(jobs):
        if key in done: continue
        try: q = t3_terms(ws, mus, Ss, sig)
        except (MemoryError, ValueError, np.linalg.LinAlgError) as e: q = {'error': str(e)}
        q.update({'key': key, 'K': len(ws), 'sigma': sig}); out.append(q)
        if k % 20 == 0 or k == len(jobs) - 1: json.dump(out, open(fn, 'w'), indent=1, default=float); print(k + 1, len(jobs), f'({time.time() - t0:.0f} s)', flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=float)
