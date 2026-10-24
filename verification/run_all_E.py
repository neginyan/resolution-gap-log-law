"""Paper E: verification of the claims of the paper, from the stored results in results/ (plus a few cheap recomputations with
the packaged code).  Exit code 0 iff every check passes.  Runs in a few minutes on one core.  Python 3.11.
Run from anywhere:  python verification/run_all_E.py  (from the package root)"""
import sys, os, json, glob, ast
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PE, 'code')); sys.path.insert(0, HERE); os.chdir(PE)
os.environ.setdefault('OMP_NUM_THREADS', '1')
from vtools import check, finish, reproduce

J = lambda f: json.load(open(os.path.join('results', f + '.json')))

# ---------------- package
files = sorted(glob.glob('code/*.py') + glob.glob('*.py') + ['verification/run_all_E.py'])
bad = []
for fn in files:
    try: ast.parse(open(fn).read(), feature_version=(3, 11))
    except SyntaxError as e: bad.append('%s: %s' % (fn, e))
check('package: Python 3.11 grammar for every file (%d)' % len(files), not bad, '; '.join(bad))
ok, dev, where, _ = reproduce('results/paperE_numbers.json', 'make_numbers.py', loose={})
check('package: paperE_numbers.json reproduces from the stored results (make_numbers.py)', ok, '(%.1e at %s)' % (dev, where), dev, 5e-9)

# ---------------- 1. the cancellation-free gap formula and the mixture minimum (recomputed)
from u_cgm import mixture_terms, nodes
from v3_tools import grid_sums
from h2_adversarial import unpack
from l_tools import diag
o = J('t4_K32_s0.01'); ws, mus, Ss = unpack(np.array(o['x']), 32, 0.01) if 'x' in o else (None, None, None)
if ws is not None:
    sig = 0.01; T = mixture_terms(ws, mus, Ss)
    ev = max(abs(m.real[0]) + 6 * np.sqrt(S[0, 0] + sig ** 2) for c, m, S in T); ew = max(abs(m.real[1]) + 6 * np.sqrt(S[1, 1] + sig ** 2) for c, m, S in T)
    A = grid_sums(T, sig, nodes(ev, 0.25 * sig, ev, 0.25 * sig), nodes(ew, ew / 700, ew, ew / 700)); g0 = diag(ws, mus, Ss, sig)['gap_over_sigma']
    d = max(abs(A['gap_over_sigma_ibp'] / g0 - 1), abs(A['gap_over_sigma'] / g0 - 1), abs(g0 / o['gap'] - 1))
    check('E1: hourglass minimum %.5f: cancellation-free formula (two forms of N) = direct evaluation (l_tools.diag) = stored value' % g0, d < 1e-5, '(%.1e)' % d, d, 1e-4)
else:
    check('E1: hourglass minimum stored as %.5f (state parameters not in the package)' % o['gap'], abs(o['gap'] - 0.88401) < 1e-5)
RV = J('w_recheck_V'); d1 = max(r['rel_new_vs_old'] for r in RV['finite_strips'] + RV['ripple_U'])
check('E1: stage-V finite strips and stage-U ripples are unaffected by the rounding fix of stage W (old vs new point fields)', d1 < 1e-10, '(%.1e)' % d1, d1, 1e-9)
from w_tools import terms_of
from v3_tools import _pre, point_fields
from w_mpcheck import fields as dec_fields
sg = 1e-80; T2 = terms_of([(1.0, 0.0, 8.9e12)], [(1.0, 0.0, 1.0, 1)], 4.0, sig=sg, Lv_sig=1e-2 / np.sqrt(sg)); per = 2 * np.pi * sg / 4.0; s_ = np.sqrt(sg)
P2 = np.array([[f * per, g * s_] for f in (0.13, 0.37) for g in (0.2, 3.0, 3.6e12)]); p2, q2 = point_fields(_pre(T2, sg), sg, P2); D2 = dec_fields(T2, sg, P2)
dv = max(abs(q2['Nibp'][i] / D2[i]['Nibp'] - 1) for i in range(len(P2)))
check('E1: point fields (N = -d_p D, cancellation-free) at sigma = 1e-80 = independent 320-digit evaluation (window and background points)', dv < 5e-12, '(%.1e)' % dv, dv, 5e-11)

# ---------------- 2. lines, the p~ chain, Var(phi), lemma (x)
C = [o for o in J('t1_chain') if 'error' not in o]; R = [o for o in C if o['y_over_sigma'] <= 0.5]
check('E2: chain x^2/Lambda <= x^2 J_ww/(a I(p~)) <= P# on all %d states' % len(C), all(o['bound_Lambda'] <= o['bound_ptilde'] * (1 + 1e-9) and o['bound_ptilde'] <= o['P_sharp'] * (1 + 1e-9) for o in C))
check('E2 (numerical): in y/sigma <= 1/2 (%d states) Lambda <= 2.5 and x^2 J_ww/(a I(p~)) >= 0.75' % len(R), max(o['Lambda'] for o in R) <= 2.5 and min(o['bound_ptilde'] for o in R) >= 0.75)
S1 = J('s1_chain_0of1'); d = max(abs(o['A_line'] / o['A'] - 1) for o in S1 if 'A' in o)
check('E2: line decomposition A = sigma^2 E[u^2 s_v^2] = E_w[omega U] (exact) on the re-evaluated stored states', d < 1e-10, '(%.1e)' % d, d, 1e-8)
V1 = J('t1_var1d'); vs = max(max(o['Var_phi'] for o in V1['searches'] if o['mode'] == 'V'), max(c['Var_phi'] for c in V1['comb']))
from paperE_data import C_SPLIT
check('E3: Var(phi) <= (sqrt(3/4) + 1.5 sqrt 2)^2 = %.3f; largest value found %.3f' % (C_SPLIT, vs), vs <= C_SPLIT and abs(C_SPLIT - 8.9242) < 1e-3)
check('E4: lemma (x), bound 1: |x - 1| <= sd_p~(mu) sqrt(I(p~)) on all %d states' % len(C), all(abs(o['x_minus_1']) <= o['x_bound_ptilde'] * (1 + 1e-9) for o in C))
U2 = J('u2_chain'); S2 = [o for o in U2['states'] if 'error' not in o]; tc = U2['two_clusters']
check('E4: lemma (x), bound 2 (exact identity): |x - 1| <= new bound on all %d states; two clusters: sd_p~(mu)/sigma up to %.0f while the new bound <= %.3f' % (
      len(S2), max(o['sd_ptilde_mu_over_sigma'] for o in tc), max(o['x_bound_new'] for o in tc)),
      all(abs(o['x_minus_1_identity']) <= o['x_bound_new'] * (1 + 1e-9) for o in S2) and max(o['sd_ptilde_mu_over_sigma'] for o in tc) > 30 and max(o['x_bound_new'] for o in tc) < 0.05)

# ---------------- 5. counterexample families
L = J('u1_lattice'); rows = [r for r in L['rows'] if r['c'] <= 0.1]; dv = max(abs(r['aIpt_over_Jww'] / r['pred_4k0sq'] - 1) for r in rows)
check('E5: periodic lattice: a I(p~)/J_ww = 4 k0^2 (k0 = %s)' % ', '.join('%g' % k for k in sorted({r['k0'] for r in rows})), dv < 1e-5, '(%.1e)' % dv, dv, 1e-4)
cmb = [c for c in J('s1_gamma1d_comb') if c['spacing_over_sigma'] <= 3]; dc = max(abs(c['D'] / c['four_k0_sq'] - 1) for c in cmb)
check('E5: 1-d comb: line dispersion D = 4 k0^2 (spacings <= 3 sigma)', dc < 0.01, '(%.1e)' % dc, dc, 0.05)
P = J('u1_patch'); ph = [v for v in P.values() if v['sigma'] < 0.5 and v['q0'] < 0.5]; RP = J('u3_ripple')['ripple_local']
check('E5: patches (max %.2f) and localised ripples (min %.2f at y/sigma <= 1/2) exceed the mixtures (%.2f); the ripples keep gap >= %.2f sigma' % (
      max(v['aIpt_over_Jww'] for v in ph), min(r['aIpt_over_Jww'] for r in RP if r['Y0'] == 0.5), max(o['aIpt_over_Jww'] for o in R), min(r['gap_over_sigma'] for r in RP)),
      max(v['aIpt_over_Jww'] for v in ph) > 5 and min(r['aIpt_over_Jww'] for r in RP if r['Y0'] == 0.5) > 2 * 1.56 and min(r['gap_over_sigma'] for r in RP) > 6 and all(r['y_over_sigma'] <= r['Y0'] for r in RP))
VP = J('v1_vperiodic'); vr = sorted([r for r in VP['rows'] if r['k0'] == 1.5 and r['lam'] == 3.0 and r['M'] >= 100], key=lambda r: r['M'])
d = abs(vr[-1]['B2'] * 4 * 1.5 ** 2 - 1)
check('E5: V-periodic strip: B2 decreases with M towards 1/(4 k0^2) (k0 = 1.5: %.4f at M = %d)' % (vr[-1]['B2'], vr[-1]['M']), all(vr[i + 1]['B2'] < vr[i]['B2'] for i in range(len(vr) - 1)) and d < 0.05, '(%.3f)' % d, d, 0.25)

# ---------------- 7. strips: the sigma -> 0 limit, g_min(sigma), the log law
K = [r for r in J('w1_klimit')['rows'] if r['k0'] <= 6]; dk = max(abs(r['opt_gap_over_sigma'] * 2 * r['k0'] - 1) for r in K)
check('E7: strip limit (sigma = 1e-80): min gap/sigma = 1/(2 k0) for k0 = 1.5 ... 6', dk < 2e-3 and all(r['feasible'] for r in K), '(%.1e)' % dk, dk, 1e-2)
IC = J('w_indep_check'); dki = max(r['rel'] for r in IC['klimit'])
check('E7: the strip limit reproduced by the separate evaluator (w_indep: closed form, covariance form of N, two-scale quadrature)', dki < 1e-3, '(%.1e)' % dki, dki, 5e-3)
from w_indep import strip_gap
from x_indep import multi_gap
GI = J('w1_gmin_indep_opt'); dd = 0.0
for key in ('1e-12', '1e-24'):
    r = GI[key]; g1 = strip_gap(r['sigma'], r['k0'], r['lam'], r['M'], r['Lv_over_ell'], r['c'], n_theta=96, n_slow=24, n_gh=64, n_gl=16, win_panels_per_ell=5, bg_panels=96)['gap_over_sigma']
    g2 = multi_gap(r['sigma'], r['k0'], [(r['c'], 0.0, r['lam'], 1, 0.0)], r['M'], r['Lv_over_ell'])['gap_over_sigma']
    dd = max(dd, abs(g1 / r['final_gap'] - 1), abs(g2 / r['final_gap'] - 1))
check('E7: g_min optima at sigma = 1e-12, 1e-24 recomputed now by two separately written evaluators (w_indep, x_indep)', dd < 1e-6, '(%.1e)' % dd, dd, 1e-4)
X4 = J('x4_states_for_main'); m4 = max(max(s['rel_A_vs_B'].values()) for s in X4['states'])
check('E7: best strips at sigma = 1e-4, 1e-6, 1e-12, 1e-24: every term of the gap formula by two code-disjoint evaluators (also confirmed independently by the main line)', m4 < 1e-3, '(%.1e)' % m4, m4, 1e-2)
gi = sorted(GI.values(), key=lambda r: -r['sigma'])
check('E7: g_min(sigma) decreases monotonically from 1e-6 to 1e-96 (%s)' % ', '.join('%.4f' % r['final_gap'] for r in gi), all(gi[i + 1]['final_gap'] < gi[i]['final_gap'] for i in range(len(gi) - 1)) and all(r['feasible'] for r in gi))
SW = J('summary_W'); f = SW['fit_opt_loglaw']
check('E7: log-law fit g = A/sqrt(ln(1/sigma) - B) on sigma <= 1e-24: A = %.3f (heuristic A = 1), residual %.1e; local exponents tend to -1/2 (%s)' % (f['A'], f['max_rel_resid'], ', '.join('%.2f' % e['value'] for e in SW['local_exponent'])),
      f['max_rel_resid'] < 0.02 and 0.9 < f['A'] < 1.15 and all(SW['local_exponent'][i + 1]['value'] > SW['local_exponent'][i]['value'] for i in range(len(SW['local_exponent']) - 1)))
cr = SW['cross_0884']
check('E7: strips beat the mixtures (0.884 sigma) only below sigma ~ %.1e (between 1e-5 and 1e-6)' % cr['sigma_cross'], cr['between'] == [1e-05, 1e-06])
TW = J('w_twoway'); zm = max(abs(r['z_score']) for r in TW)
check('E7: grid-free Monte Carlo agrees with the grids on the corrected strips (max |z| %.1f)' % zm, zm < 3)

# ---------------- 8. N, Theorem W', Lemma X2
W2 = J('w2_states'); dn = max(abs(v['N2_over_N'] - 1) for v in W2.values())
check('E8: second-order form of N (from N = -sigma^-2 E[Cov(x_p, x_z - sin x_z | y)] and Tweedie) within 1% of N on all states', dn < 0.01, '(%.1e)' % dn, dn, 0.05)
X1 = J('x1_states')
check("E8: Theorem W': gap/sigma >= (a/den) sqrt(P_w)(1 - eps_+) on all %d states (mixtures, ripples, two clusters, strips)" % len(X1), all(v['bound_le_gap'] for v in X1.values()))
em = max(abs(v['eps']) for v in X1.values()); eb = max(v['eps_B_absbound'] / np.sqrt(v['sigma']) for v in X1.values())
check("E8: Theorem W': |eps| <= %.4f; explicit bound of B <= %.2f sqrt(sigma)" % (em, eb), em < 0.01 and eb < 5, '', em, 0.05)
rb = [v['R3_over_sa'] / v['B_over_sa'] for k, v in X1.items() if (k.startswith('mix') or k.startswith('two') or k.startswith('stored')) and abs(v['B_over_sa']) > 1e-7]
check('E8: R3/B = 1/2 for near-Gaussian posteriors (mixtures, clusters: %.3f ... %.3f)' % (min(rb), max(rb)), 0.45 < min(rb) and max(rb) < 0.53)
L2 = J('x2_lemma')
check('E8: Lemma X2 on %d stored random 1-d structures: max ratio %.4f <= 1' % (len(L2['random']), L2['max_ratio_random']), L2['max_ratio_random'] <= 1 + 1e-9)
rng = np.random.default_rng(7); n = 2 ** 14; h = 400.0 / n; v = (np.arange(n) - n / 2) * h; xi = 2 * np.pi * np.fft.fftfreq(n, d=h); worst = 0.0
for t in range(60):
    fsig = sum(rng.normal() * np.exp(-(v - rng.uniform(-80, 80)) ** 2 / (2 * rng.uniform(2, 40) ** 2)) * np.cos(rng.uniform(0, 6) * v + rng.uniform(0, 6.3)) for _ in range(rng.integers(1, 5)))
    Fh = np.fft.fft(fsig) * h; G = Fh * np.exp(-0.5 * xi ** 2); nf = np.sum(np.abs(Fh) ** 2); ng = np.sum(np.abs(G) ** 2); Fd = np.sum(xi ** 2 * np.abs(G) ** 2); xb = Fd / ng
    worst = max(worst, Fd / (xb * np.exp(-xb) * nf))
check('E8: Lemma X2 recomputed now on 60 fresh random structures (max ratio %.4f <= 1)' % worst, worst <= 1 + 1e-9)
cm = max(t['C_needed'] for t in L2['transfer_strips'])
check('E8 (observation): strip windows obey a_win/mu_win <= C xbar e^{-xbar} with C = %.2f' % cm, cm < 1)

# ---------------- 9. beyond the single strip
X3 = J('x3_verify')
check('E9: k + 2k, interfering windows, W-varying k0, harmonics (sigma = 1e-12, 1e-24): none below the single strip (min excess %+.2f%%); re-evaluated by two code-disjoint evaluators' % (
      100 * min(r['x_indep_final'] / r['single_strip'] - 1 for r in X3)), all(r['x_indep_final'] > r['single_strip'] for r in X3) and max(max(r['rel_w_full'], r['rel_one_period']) for r in X3) < 1e-3)

# ---------------- 10. stage Y3
Y = J('y3_r3'); mr = max(v['ratio_R3_to_bound'] for v in Y.values())
check('E10: R3 bound |R3| <= (s^4/2) E|sin m_z d_p H_zz| + (s^2/6) E[s_hat_p M6_hat] on %d mixture/cluster states (max |R3|/bound %.2f; eps_R3 bound <= %.3f)' % (len(Y), mr, max(v['eps_R3_bound'] for v in Y.values())),
      all(v['holds'] for v in Y.values()) and max(v['a_rel_check'] for v in Y.values()) < 1e-4)
A = J('y3_a2')
check('E10: P# >= x^2/(1 + Dbar) on all %d states (stage T chain + stage U constant); max ratio %.3f' % (A['chain_ii']['n'], A['chain_ii']['max_ratio']), A['chain_ii']['n_violations'] == 0)
check('E10: line bound D <= 8 xl + 2 sigma^4 E[s^4]/omega and sigma^4 E[s^4] <= 3 on %d 1-d blurred densities (max ratio %.3f)' % (A['line_iii']['n'], A['line_iii']['max_ratio']), A['line_iii']['max_ratio'] <= 1 + 1e-9 and A['line_iii']['max_sigma4_Es4'] <= 3)
check("E10: Lemma X2' (second spectral moment) on %d structures: max ratio %.4f <= 1" % (A['lemma_iv']['n'], A['lemma_iv']['max_ratio']), A['lemma_iv']['max_ratio'] <= 1 + 1e-9)
# ---------------- 11. stage Z: the conditional log-law theorem (Theorem Z)
SZ = J('summary_Z')
for fam, nm in (('steps_x1', 'mixtures, stored sample, localised ripples, two clusters, strips'), ('steps_stored', '377 stored + adversarial mixtures'), ('steps_strips', 'strip optima 1e-2 ... 1e-96')):
    T = SZ[fam]; viol = sum(e.get(s, {}).get('violations', 0) for e in T.values() for s in ('S0', 'S1', 'S2', 'S3', 'S5', 'S6', 'S8', 'S56'))
    check('E11: every step of Theorem Z holds on the %s (%d states; violations %d)' % (nm, sum(e['n'] for e in T.values()), viol), viol == 0)
v = SZ['vs_t1_chain']
check('E11: Lambda and P# of the stage-Z code agree with the stage-T code on %d states (max rel %.1e, %.1e)' % (v['n'], v['max_rel_Lambda'], v['max_rel_Psharp']), max(v['max_rel_Lambda'], v['max_rel_Psharp']) < 4e-3, '', max(v['max_rel_Lambda'], v['max_rel_Psharp']), 2e-2)
gc = SZ['strip_grid_checks']; mg = max(g['max_rel_twoscale_vs_grid'] for g in gc.values())
check('E11: strips: two-scale evaluator = full 2-d grid of complex Gaussian terms (max rel %.1e)' % mg, mg < 1e-5, '', mg, 1e-4)
check('E11: Lemma Z1 identity Lambda = 4 xbar_w(sqrt p) (FFT, max rel %.1e) and the Heisenberg, Cramer-Rao, 2-d Jensen inequalities on %d test states' % (SZ['identity']['max_rel_Lambda_4xbar_sqrtp'], SZ['identity']['n']),
      SZ['identity']['max_rel_Lambda_4xbar_sqrtp'] < 5e-4 and SZ['identity']['heis_ok'] and SZ['identity']['CR_ok'] and SZ['identity']['jensen_ok'], '', SZ['identity']['max_rel_Lambda_4xbar_sqrtp'], 5e-3)
tc = SZ['theorem_constants_strips']; rows = SZ['strips']
check('E11: Theorem Z curve c/sqrt(ln(1/sigma) + C) (c = %.4f, C = %.3f) lies below the strips g_min(sigma) at every sigma (min ratio %.1f); A/c = %.1f' % (tc['c'], tc['C'], min(r['gap_over_theorem'] for r in rows), tc['A_over_c']),
      all(r['gap_over_theorem'] > 1 for r in rows) and abs(tc['c'] - 1 / (2 * np.pi * np.sqrt(tc['K1_max_K_Lambda']))) < 1e-12)
sys.path.insert(0, os.path.join(PE, 'code'))
from z_strip import strip_params, strip_sums, strip_closed
from z_tools import chain as zchain
s = J('z_chain_strips')['strip_gmin_1e-12']; P = strip_params(1e-12, s['k0'], s['lam'], s['M'], s['eta'], s['c']); E_, m_ = strip_sums(P); cl, vz = strip_closed(P)
rr = zchain(1e-12, E_, {'p2': np.nan, 'pw2': np.nan, 'pvw2': np.nan, 'pv2': np.nan}, cl, vz); dd = max(abs(rr[k] / s[k] - 1) for k in ('Lambda', 'P_y', 'K_Lambda', 'Xi'))
check('E11: strip sigma = 1e-12 recomputed now: Lambda %.3f, P_y %.4f >= 1/Lambda, xbar_w %.3f <= Xi %.3f (rel change %.1e)' % (rr['Lambda'], rr['P_y'], rr['xbar_w'], rr['Xi'], dd),
      dd < 1e-6 and rr['P_y'] >= 1 / rr['Lambda'] and rr['xbar_w'] <= rr['Xi'], '', dd, 1e-5)
# ---------------- 12. stage AA: Theorem Z' (K_J and K_rho removed), K_Lambda probed, eps_m-free factor
SA = J('summary_AA'); st = SA['aa1_steps']
check("E12: AA1 steps (split Jensen, cut, kernel ratio, L2 Heisenberg, final xbar_w <= X_t) and AA3 steps hold on all %d rows (violations %d)" % (st['final']['n'], sum(e['violations'] for e in st.values())),
      sum(e['violations'] for e in st.values()) == 0)
check("E12: pair-midpoint bound: E_Q[y_z^2] = %.3f <= pi^2 + sigma^2/2 on every state" % SA['E_Q_max'], SA['E_Q_max'] <= np.pi ** 2)
check("E12: Theorem Z' with state constants lies below the gap on every state with the gap known (min gap/bound %.2f)" % SA['gap_over_bound_state_min'][0], SA['gap_over_bound_state_min'][0] > 1)
from aa_tools import strip_l2, pair_l2, xbar_chain
from z_strip import strip_terms
s = J('z_chain_strips')['strip_gmin_1e-06']; P = strip_params(1e-6, s['k0'], s['lam'], s['M'], s['eta'], s['c'])
r1 = xbar_chain(lambda bv, bw, w: strip_l2(P, bv, bw, w), 1e-6, 0.9)
check("E12: AA1 chain recomputed now on the sigma = 1e-6 strip (t = 0.9): xbar_w %.3f <= X_t %.2f, every step holds" % (r1['xbar_w'], r1['X_t_state']),
      all(r1[q][0] >= r1[q][1] for q in ('A_jensen', 'B_cut', 'C_kernel', 'D_heis_L2', 'final')))
PF = SA['product_family']
check("E12: K_Lambda has no universal bound: product family K_Lambda = %s (y/sigma <= 1/2) with sqrt(P_w) >= %.3g (large gap)" % (', '.join('%.1f' % r['K_Lambda'] for r in PF), min(r['sqrtPw_lower'] for r in PF)),
      PF[-1]['K_Lambda'] > 10 and max(r['y_over_sigma'] for r in PF) <= 0.5 and min(r['sqrtPw_lower'] for r in PF) > 1)
ab = SA['adv_best']
check("E12: small-gap regime: max K_Lambda %.3f (adversarial, sqrt(P_w) <= 1) and %.3f (stage-Z states)" % (ab['KL_gap']['K_Lambda'], SA['K_Lambda_max_small_gap_stored']), max(ab['KL_gap']['K_Lambda'], SA['K_Lambda_max_small_gap_stored']) < 1.5)
# ---------------- 13. H0 removed (Theorem Z'')
SB = J('summary_AB')
HS = SB['H0_strips']
check('E13: H0 removed on the strips: eps_pi <= 10^%.1f (sigma = 1e-2) and below; H0\' (E_Q <= pi^2 + sigma^2/2) and H0\'\' (Var(X_z) <= pi^2) hold on all 8 strips' % HS['strip_gmin_0.01']['log10_eps_pi_bound'],
      all(v['H0p_ok'] and v['H0pp_ok'] for v in HS.values()) and max(v['log10_eps_pi_bound'] for v in HS.values()) < -3)
# ---------------- 14. Lemma AC1 (Fisher form) and the directional de Bruijn identity
SC = J('summary_AC')
check('E14: Lemma AC1 E[q^2] <= 8 E[H_vw^2] + 2 J_ww/t on %d states x 13 blurs (min ratio %.2f); t H_vv <= 1 (max %.6f); sigma^4 E[q^2] <= 14 (max %.2f)' % (SC['n_states'], SC['L1_min_ratio'][0], SC['tHvv_max'], SC['q2_dimensionless_max']),
      SC['L1_min_ratio'][0] >= 1 and SC['tHvv_max'] <= 1 + 1e-9 and SC['q2_dimensionless_max'] <= 14)
check('E14: directional de Bruijn identity dJ_ww/dt = -E[H_vw^2 + H_ww^2] (%d checks, max rel %.1e)' % (SC['deBruijn_n'], SC['deBruijn_max_rel'][0]), SC['deBruijn_max_rel'][0] < 5e-3, '', SC['deBruijn_max_rel'][0], 5e-2)
from ac_heat import sums_lattice
B9 = np.array([[3.25, 3.25], [3.212, -3.212]]); E1 = sums_lattice(B9, 1.0); E0, E2 = sums_lattice(B9, 0.99), sums_lattice(B9, 1.01)
dlat = abs(E1['J'] * np.log(E2['J'] / E0['J']) / 0.02 / (-(E1['Hvw2'] + E1['Hww2'])) - 1)
check('E14: de Bruijn recomputed now on an oblique reference lattice (rel %.1e) and Lemma AC1 there (%.3f <= %.3f)' % (dlat, E1['q2'], 8 * E1['Hvw2'] + 2 * E1['J']), dlat < 1e-3 and E1['q2'] <= 8 * E1['Hvw2'] + 2 * E1['J'], '', dlat, 1e-2)
# ---------------- 15. Gamma_2 identity
SD = J('summary_AD')
from ad_gamma2 import identity_check
ic = identity_check([[2.5, 0.7], [0.3, 2.2]], 1.3, h=0.003)
check('E15: Gamma_2 identities recomputed now (oblique lattice, t = 1.3): J\' = -E|grad u|^2 (%.1e), J\'\' = E||Hess u||^2 + 2E[(H^3)_ww] (%.1e), J R\' = -E[(Lu+Ru)^2] - E[(H^3)_ww] (%.1e)' % (ic['G1'], ic['G2'], ic['G4']),
      max(ic['G1'], ic['G2'], ic['G4']) < 2e-3, '', max(ic['G1'], ic['G2'], ic['G4']), 1e-2)
check('E15: t^2 R\' <= kappa_H (Corollary 2) at that point: %.4f <= %.4f; stored identity checks max %.1e' % (ic['t2R1'], ic['kappa_H'], max(SD['gamma2_identities'].values())), ic['t2R1'] <= ic['kappa_H'] + 1e-6 and max(SD['gamma2_identities'].values()) < 2e-3)
# ---------------- 16. valley lemma and the 1-d / product-state theorem
SE = J('summary_AE')
from ae_oned import lemma_checks, sums, t2Rp
xx = np.array([-3.1, 0.0, 0.4, 4.2, 9.0]); ww_ = np.array([0.3, 0.1, 0.25, 0.2, 0.15]); m1, m2 = lemma_checks(xx, ww_, 1.3)
check('E16: Lemma AE1 (ln p second difference >= ln cosh(u sqrt(2c)) - u^2) recomputed now on a 5-point 1-d state: min margin %.1e; (*) min ratio %.2f >= 1' % (m1, m2), m1 > -1e-9 and m2 >= 1)
S1 = sums(xx, ww_, 1.3); tr, _ = t2Rp(xx, ww_, 1.3)
check("E16: 1-d theorem t^2R' <= eta tR + kappa_eta there: %.4f <= %.4f; N_eta/(tJ) = %.3f <= kappa_0.1 = %.0f" % (tr, 0.1 * S1['tR'] + SE['kappa_eta']['0.1'], S1['N_eta_over_tJ'], SE['kappa_eta']['0.1']),
      tr <= 0.1 * S1['tR'] + SE['kappa_eta']['0.1'] and S1['N_eta_over_tJ'] <= SE['kappa_eta']['0.1'])
check('E16: kappa_eta band sum stable under grid refinement (max rel %.1e); kappa_0.1 = %.1f' % (max(SE['kappa_eta_grid_check'].values()), SE['kappa_eta']['0.1']), max(SE['kappa_eta_grid_check'].values()) < 0.03)
check('E16: 1-d searches K = 2-32: N_eta/(tJ) <= %.2f (proved <= %.0f); t^2 R\' <= %.3f < 0; 1-d lattices t^2 R\' <= %.3f' % (SE['oned']['max_N_eta_over_tJ'], SE['kappa_eta']['0.1'], SE['oned']['max_t2Rp'], SE['oned']['lattice_1d']['max_t2Rp']),
      SE['oned']['max_N_eta_over_tJ'] < SE['kappa_eta']['0.1'] and SE['oned']['max_t2Rp'] < 0)
# ---------------- 17. two-sided valley constant; strips have no deep valleys
SF = J('summary_AF')
check('E17: two-sided (*) halves the valley constant: kappa_0.1 %.1f -> %.1f (grid check %.1e)' % (SF['kappa_one_sided']['0.1'], SF['kappa_two_sided']['0.1'], SF['kappa_grid_check']),
      abs(SF['kappa_two_sided']['0.1'] / SF['kappa_one_sided']['0.1'] - 0.5) < 1e-6 and SF['kappa_grid_check'] < 0.03)
sm = SF['strips_max']
check("E17: strip optima at t = sigma^2/4, /2, 1: no deep valleys (lambda_max %.2f), kappa_H <= %.3f, t^2 R' <= %.3f" % (sm['lambda_max'], sm['kappa_H'], sm['t2R1']), sm['lambda_max'] < 3 and sm['t2R1'] <= 1e-3 and sm['kappa_H'] < 0.1)
# ---------------- 18. strips on the whole window (Proposition AG1)
SG = J('summary_AG')
from ag_strip_lambda import lam_field
from z_strip import strip_params
zz = J('z_chain_strips')['strip_gmin_0.01']; PP = strip_params(zz['sigma'], zz['k0'], zz['lam'], zz['M'], zz['eta'], zz['c'])
lq, _ = lam_field(PP, 0.25 * zz['sigma'] ** 2, n_theta=1000, n_w=1801)
check('E18: strip sigma = 1e-2 at t = sigma^2/4 recomputed on another grid: lambda_max = %.4f (stored %.4f; main line 1.874)' % (lq.max(), SG['main_line_check_sigma_0.01']['quarter'][0]), abs(lq.max() - SG['main_line_check_sigma_0.01']['quarter'][0]) < 2e-3 and abs(lq.max() - 1.874) < 2e-3)
check('E18: strips: lambda_max <= %.3f < 3 on the whole window (8 optima x 25 blurs), monotone decreasing in t; Proposition AG1: t^2 R\' <= %.3f tR, observed kappa_H/tR <= %.3f' % (SG['strip_lambda_star'], SG['strip_lambda_star'], SG['strip_kappa_H_over_tR_max']),
      SG['strip_lambda_star'] < 3 and all(v['monotone_decreasing'] for v in SG['strip_lambda'].values()) and SG['strip_kappa_H_over_tR_max'] <= SG['strip_lambda_star'])
# ---------------- 19. the two further numbers quoted in the paper (recomputed from the stored rows, not from paperE_data)
mx = J('ac_heat_mix')['twoclusters_V2_s0.001']['rows']; Rr = {float(k): v['R'] for k, v in mx.items()}; tt = sorted(Rr)
r_lc = Rr[0.25] / Rr[1.0]; c_lc = min(Rr[a] / Rr[b] for a, b in zip(tt[:-1], tt[1:]))
check('E19: log-convexity is not exact: two clusters (V2, sigma = 1e-3) R(sigma^2/4)/R(sigma^2) = %.3f < 1 (smallest consecutive ratio %.3f, the stored stage-AC value %.3f)' % (r_lc, c_lc, J('summary_AC')['LC_min_ratio'][0]),
      r_lc < 1 and abs(c_lc - J('summary_AC')['LC_min_ratio'][0]) < 1e-12)
rows_ = []
for mode in ('lattice', 'points', 'mix', 'strip'):
    for k, v in J('ac_heat_%s' % mode).items():
        for key, r in v['rows'].items(): rows_.append((r['t'] * r['R'], np.log(1 / (r['t'] * r['J'])), 'AC:%s:%s' % (mode, k)))
for k, v in J('ad_qlc_lattice').items():
    for r in v['rows']:
        if r: rows_.append((r['tR'], np.log(1 / (r['t'] * r['J'])), 'AD-ladder:' + k))
for k, v in J('af_strip').items(): rows_.append((v['tR'], np.log(1 / v['tJ']), 'AF-strip:' + k))
bm = max(rows_, key=lambda r: r[0] - r[1])
check('E19: (T2) on the stored states: on all %d stored state-blur rows tR <= ln(1/(t J_ww)) + %.2f (largest at %s, a periodic comb)' % (len(rows_), bm[0] - bm[1], bm[2]),
      len(rows_) == 1778 and abs((bm[0] - bm[1]) - 6.544) < 1e-3 and 'comb' in bm[2])
finish()
