"""Stage S1: the line-reduction chain for P# on every stored state (q2_uncert.all_states, 361 states) -> results/s1_chain_<part>of<n>.json
(one output file per process; restartable).   Usage: python s1_chain.py <part> <nparts>"""
import json, os, sys, time
import numpy as np
from s_tools import all_terms

if __name__ == '__main__':
    part, nparts = int(sys.argv[1]), int(sys.argv[2]); fn = f'results/s1_chain_{part}of{nparts}.json'
    from q2_uncert import all_states
    S = list(enumerate(all_states()))[part::nparts]; out = json.load(open(fn)) if os.path.exists(fn) else []; done = {o['index'] for o in out}; t0 = time.time()
    for k, (i, (tag, ws, mus, Ss, sig)) in enumerate(S):
        if i in done: continue
        try: q = all_terms(ws, mus, Ss, sig)
        except (MemoryError, ValueError, np.linalg.LinAlgError) as e: q = {'error': str(e), 'sigma': sig}
        q['tag'] = tag; q['index'] = i; q['K'] = len(ws); out.append(q)
        if k % 10 == 0 or k == len(S) - 1: json.dump(out, open(fn, 'w'), indent=1, default=float); print(f'{k + 1}/{len(S)} ({time.time() - t0:.0f} s)', flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=float)
