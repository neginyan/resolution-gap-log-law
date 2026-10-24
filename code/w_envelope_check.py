"""Stage W1: the true-envelope evaluator (w_full) against the independent stage-V full 2-d grid code (u_cgm via v2_strip.strip) on the stage-V
sigma = 1e-6 strip optimum (sigma^2/Lv^2 ~ 1e-5 there, so the stage-W rounding issue plays no role in either code).
-> results/w_envelope_check.json.  Python 3.11."""
import json
import numpy as np
from v3_tools import ripple_terms_k
from w_full import full_sums
from u_cgm import nodes
from v2_strip import strip

o = json.load(open('results/v2_gapmin_s1e-06_seed0.json'))['best']['par']; sg = 1e-6; s = np.sqrt(sg)
k0, c, lam, Aw, Lvs = o['k0'], o['c'], o['lam'], o['Aw'], o['Lv_s']; Lv = Lvs * sg; Lw = Aw * s; ell = lam * s
T = ripple_terms_k(sg, Lv, Lw, c, k0, 10 * Lv, ell)
env = {}
for f, n0 in ((1, 41), (2, 81)):
    wn = nodes(7 * ell, ell / 40 / f, 6.5 * Lw, max(Lw / 120, ell / 40) / f)
    env[f] = float(full_sums(T, sg, k0, np.sqrt(Lv ** 2 + sg ** 2), wn, nv=48, n0=n0)['gap_over_sigma'])
u = {fw: float(strip(sg, k0, lam, Aw, Lvs, c=c, fine_w=fw)['gap_over_sigma']) for fw in (1, 2, 4)}
out = {'sigma': sg, 'par': o, 'envelope': env[2], 'envelope_coarse': env[1], 'u_cgm_fine_w': u, 'u_cgm_finest': u[4]}
json.dump(out, open('results/w_envelope_check.json', 'w'), indent=1); print(out)
