"""Stage W2 numerics: the second-order form of N and the window/background tug-of-war on mixtures (K32 hourglass minimum, sigma = 1e-2) and on
ripple strips (one-period V-limit and true V-envelope).  -> results/w2_states.json.  Python 3.11."""
import json, sys
import numpy as np
from u_cgm import mixture_terms, nodes, trap_weights
from h2_adversarial import unpack
from w_tools import terms_of, w_nodes
from w2_tools import line_sums, line_sums_envelope, analyse
R = {}
# --- K32 minimum (hourglass family), sigma = 1e-2: full 2D grid as in run_all_V
o = json.load(open('results/t4_K32_s0.01.json')); ws, mus, Ss = unpack(np.array(o['x']), 32, 0.01); sig = 0.01; T = mixture_terms(ws, mus, Ss)
ev = max(abs(m.real[0]) + 6 * np.sqrt(S[0, 0] + sig ** 2) for c, m, S in T); ew = max(abs(m.real[1]) + 6 * np.sqrt(S[1, 1] + sig ** 2) for c, m, S in T)
vn = nodes(ev, 0.25 * sig, ev, 0.25 * sig); wn = nodes(ew, ew / 700, ew, ew / 700)
L = line_sums(T, sig, vn, trap_weights(vn), wn)
for th in (1.0, 2.0): R['K32_s0.01_theta%g' % th] = analyse(L, wn, sig, th)
print('K32 done', flush=True)
# --- strips, one-period V-limit (sigma = 1e-24, Lv = ell/150)
for (k0, lam, M) in [(2.5, 1.5, 11943), (2.5, 1.5, 1e5), (4.0, 1.0, 8.9e12)]:
    sg = 1e-80 if k0 > 3 else 1e-24; bg = [(1.0, 0.0, lam * M)]; win = [(1.0, 0.0, lam, 1)]
    Lvs = lam / np.sqrt(sg) / 150; T = terms_of(bg, win, k0, sg, Lvs); wn = w_nodes(bg, win, sg, 40, 400)
    per = 2 * np.pi * sg / k0; nv = 32; vn = (np.arange(nv) + 0.5) * per / nv - per / 2
    L = line_sums(T, sg, vn, np.full(nv, per / nv), wn)
    R['strip_vlimit_k%g_M%.3g' % (k0, M)] = analyse(L, wn, sg, 1.0); print('strip', k0, M, flush=True)
# --- strip with the true V-envelope (sigma = 1e-24): Lv = ell/150 and a wider Lv
for eta in (1 / 150.0, 0.05):
    sg = 1e-24; k0, lam, M = 3.0, 1.5, 8.1e4; bg = [(1.0, 0.0, lam * M)]; win = [(1.0, 0.0, lam, 1)]
    Lvs = eta * lam / np.sqrt(sg); T = terms_of(bg, win, k0, sg, Lvs); wn = w_nodes(bg, win, sg, 25, 200)
    L = line_sums_envelope(T, sg, k0, np.sqrt((Lvs * sg) ** 2 + sg ** 2), wn)
    R['strip_envelope_k3_eta%.3g' % eta] = analyse(L, wn, sg, 1.0); print('env', eta, flush=True)
json.dump(R, open('results/w2_states.json', 'w'), indent=1, default=float)
for k, v in R.items():
    print('%-28s gap %.4f | N2/N %.3f gapN2 %.4f | G_win %.4f sqrtP#win %.4f | G_bg %.4f (Jww %.4f, N %.4f) T_bg %.4f | mass_win %.3g a_win share %.3f' % (
        k, v['gap_over_sigma'], v['N2_over_N'], v['gap_N2_over_sigma'], v['G_win'], v['sqrtPsharp_win'], v['G_bg'], v['Gbg_Jww'], v['Gbg_N'], v['T_bg'], v['mass_win'], v['a_win_share']))
