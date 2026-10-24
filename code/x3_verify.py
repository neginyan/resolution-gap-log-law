"""Stage X3: re-evaluate the best state of every finished X3 search with the stage-W evaluators, which share nothing with x_indep:
  w_full (true V-envelope; w_tools states with phases; v3 point fields, d_p D form of N) and the one-period limit (w_tools.evaluate) + Lv^2/(8 sigma).
-> results/x3_verify.json.  Python 3.11."""
import json, glob
import numpy as np
from w_tools import terms_of, w_nodes, evaluate
from w_full import full_sums

out = []
for fn in sorted(glob.glob('results/x3_opt_*.json')):
    o = json.load(open(fn))
    if not o.get('finished'): continue
    b = o['best']; sg = o['sigma']; kb = b['kb']
    win = [(w['c'], w['w_center'], w['ell'], w['m'], w['phase']) for w in b['windows']]; ell1 = win[0][2]
    bg = [(1.0, 0.0, ell1 * b['M'])]; Lvs = b['eta'] * ell1 / np.sqrt(sg); mmax = max(w[3] for w in win)
    f = full_sums(terms_of(bg, win, kb, sg, Lvs), sg, kb, np.sqrt((Lvs * sg) ** 2 + sg ** 2), w_nodes(bg, win, sg, 40, 400), nv=32 * mmax, n0=21, ext=6.5)
    v = evaluate(bg, win, kb, sig=sg, Lv_sig=Lvs, nper=40, coarse=400)
    row = {'family': o['family'], 'sigma': sg, 'x_indep_final': b['final']['gap_over_sigma'], 'w_full': float(f['gap_over_sigma']),
           'one_period_plus_vspread': float(v['gap_over_sigma']) + (Lvs * sg) ** 2 / (8 * sg), 'single_strip': o['single_strip_gap'],
           'period_over_Lv': 2 * np.pi * sg / kb / (Lvs * sg)}
    row['rel_w_full'] = abs(row['w_full'] / row['x_indep_final'] - 1); row['rel_one_period'] = abs(row['one_period_plus_vspread'] / row['x_indep_final'] - 1)
    out.append(row); print(row, flush=True)
json.dump(out, open('results/x3_verify.json', 'w'), indent=1)
