"""Paper E: every number of the paper, read from paperE/results ONLY (the single place where numbers are read).
numbers() returns a list of entries {sec, name, value, src} with src = 'file.json:key' (selectors in [..]).  Python 3.11."""
import os, json
import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
J = lambda f: json.load(open(os.path.join(RES, f + '.json')))
C_SPLIT = (np.sqrt(0.75) + 1.5 * np.sqrt(2.0)) ** 2           # the proved constant of Var(phi) <= 8.92 (stage T1)



def lc_two_clusters():
    rows = J('ac_heat_mix')['twoclusters_V2_s0.001']['rows']
    R = {float(k): v['R'] for k, v in rows.items()}; ts = sorted(R)
    cons = min(R[a] / R[b] for a, b in zip(ts[:-1], ts[1:]))
    return R[0.25] / R[1.0], cons


def stored_T2_rows():
    out = []
    for mode in ('lattice', 'points', 'mix', 'strip'):
        for k, v in J('ac_heat_%s' % mode).items():
            for key, r in v['rows'].items(): out.append(('ac_heat_%s.json:%s' % (mode, k), key, r['t'] * r['R'], float(np.log(1 / (r['t'] * r['J'])))))
    for k, v in J('ad_qlc_lattice').items():
        for r in v['rows']:
            if r: out.append(('ad_qlc_lattice.json:' + k, r['t'], r['tR'], float(np.log(1 / (r['t'] * r['J'])))))
    for k, v in J('af_strip').items(): out.append(('af_strip.json:' + k, v['m'], v['tR'], float(np.log(1 / v['tJ']))))
    return out


def numbers():
    E = []
    add = lambda sec, name, v, src: E.append({'sec': sec, 'name': name, 'value': float(v) if not isinstance(v, (str, bool)) else v, 'src': src})
    # ---- 1. mixtures (the hourglass minimum)
    T = J('summary_T'); m = T['T4_min']['opt_P_sharp']
    add('1 mixtures', 'min gap/sigma over Gaussian mixtures (K32 hourglass, sigma = 1e-2)', m['gap'], 'summary_T.json:T4_min.opt_P_sharp.gap')
    add('1 mixtures', 'min P# (same state)', m['opt_P_sharp'], 'summary_T.json:T4_min.opt_P_sharp.opt_P_sharp')
    add('1 mixtures', 'F of the same state', m['F'], 'summary_T.json:T4_min.opt_P_sharp.F')
    add('1 mixtures', 'y/sigma of the same state', m['y_over_sigma'], 'summary_T.json:T4_min.opt_P_sharp.y_over_sigma')
    # ---- 2. lines and the p~ chain
    C = [o for o in J('t1_chain') if 'error' not in o]; R = [o for o in C if o['y_over_sigma'] <= 0.5]
    add('2 chain', 'states evaluated (stored + adversarial)', len(C), 't1_chain.json:len')
    add('2 chain', 'states with y/sigma <= 1/2', len(R), 't1_chain.json:[y_over_sigma<=0.5]')
    add('2 chain', 'max Lambda in y/sigma <= 1/2', max(o['Lambda'] for o in R), 't1_chain.json:[y<=0.5].Lambda max')
    add('2 chain', 'min x^2 J_ww/(a I(p~)) (bound_ptilde) in y/sigma <= 1/2', min(o['bound_ptilde'] for o in R), 't1_chain.json:[y<=0.5].bound_ptilde min')
    add('2 chain', 'max a I(p~)/J_ww in y/sigma <= 1/2', max(o['aIpt_over_Jww'] for o in R), 't1_chain.json:[y<=0.5].aIpt_over_Jww max')
    add('2 chain', 'min x^2/Lambda (bound_Lambda) in y/sigma <= 1/2', min(o['bound_Lambda'] for o in R), 't1_chain.json:[y<=0.5].bound_Lambda min')
    add('2 chain', 'chain violations (bound_ptilde > P# or bound_Lambda > bound_ptilde), all states', sum(o['bound_ptilde'] > o['P_sharp'] * (1 + 1e-9) or o['bound_Lambda'] > o['bound_ptilde'] * (1 + 1e-9) for o in C), 't1_chain.json:all')
    # ---- 3. Var(phi) <= 8.92
    V1 = J('t1_var1d')
    add('3 Var(phi)', 'proved constant (sqrt(3/4) + 1.5 sqrt 2)^2', C_SPLIT, 'formula (stage T1)')
    add('3 Var(phi)', 'largest Var(phi) found (adversarial 1-d mixtures)', max(o['Var_phi'] for o in V1['searches'] if o['mode'] == 'V'), 't1_var1d.json:searches[mode=V].Var_phi max')
    add('3 Var(phi)', 'largest Var(phi) found (combs)', max(c['Var_phi'] for c in V1['comb']), 't1_var1d.json:comb.Var_phi max')
    # ---- 4. lemma (x): two upper bounds of |x - 1|
    add('4 lemma (x)', 'states where |x - 1| <= sd_p~(mu) sqrt(I(p~)) fails (bound 1)', sum(abs(o['x_minus_1']) > o['x_bound_ptilde'] * (1 + 1e-9) for o in C), 't1_chain.json:x_minus_1, x_bound_ptilde')
    add('4 lemma (x)', 'max x_bound/sqrt(sigma) (bound 1, y/sigma <= 1/2)', max(o['x_bound_over_sqrt_sigma'] for o in R), 't1_chain.json:[y<=0.5].x_bound_over_sqrt_sigma max')
    U2 = J('u2_chain'); S2 = [o for o in U2['states'] if 'error' not in o]; tc = U2['two_clusters']
    add('4 lemma (x)', 'states where |x - 1| <= new bound fails (bound 2, exact identity)', sum(abs(o['x_minus_1_identity']) > o['x_bound_new'] * (1 + 1e-9) for o in S2), 'u2_chain.json:states')
    add('4 lemma (x)', 'two clusters: max sd_p~(mu)/sigma (bound 1 is not uniform)', max(o['sd_ptilde_mu_over_sigma'] for o in tc), 'u2_chain.json:two_clusters.sd_ptilde_mu_over_sigma max')
    add('4 lemma (x)', 'two clusters: max new bound (bound 2 stays small)', max(o['x_bound_new'] for o in tc), 'u2_chain.json:two_clusters.x_bound_new max')
    # ---- 5. counterexample families of stage U
    L = J('u1_lattice'); rows = [r for r in L['rows'] if r['c'] <= 0.1]
    add('5 families', 'lattice: max rel. deviation of a I(p~)/J_ww from 4 k0^2 (c <= 0.1)', max(abs(r['aIpt_over_Jww'] / r['pred_4k0sq'] - 1) for r in rows), 'u1_lattice.json:rows[c<=0.1]')
    seen = set()
    for r in sorted(rows, key=lambda r: (r['k0'], r['q0'])):
        if r['k0'] in (1.0, 2.0, 3.0) and r['k0'] not in seen:
            seen.add(r['k0']); add('5 families', 'lattice k0 = %g: a I(p~)/J_ww (4 k0^2 = %g)' % (r['k0'], 4 * r['k0'] ** 2), r['aIpt_over_Jww'], 'u1_lattice.json:rows[k0=%g,q0=%g,c=%g]' % (r['k0'], r['q0'], r['c']))
    P = J('u1_patch'); ph = [v for v in P.values() if v['sigma'] < 0.5 and v['q0'] < 0.5]
    add('5 families', 'normalised patches: max a I(p~)/J_ww', max(v['aIpt_over_Jww'] for v in ph), 'u1_patch.json:[sigma<0.5,q0<0.5] max')
    RP = J('u3_ripple')['ripple_local']; r5 = [r for r in RP if r['Y0'] == 0.5]
    add('5 families', 'localised ripples (y/sigma <= 1/2): min a I(p~)/J_ww', min(r['aIpt_over_Jww'] for r in r5), 'u3_ripple.json:ripple_local[Y0=0.5] min')
    add('5 families', 'localised ripples: min gap/sigma (not counterexamples to the gap)', min(r['gap_over_sigma'] for r in RP), 'u3_ripple.json:ripple_local.gap_over_sigma min')
    add('5 families', 'mixtures: max a I(p~)/J_ww (y/sigma <= 1/2)', max(o['aIpt_over_Jww'] for o in R), 't1_chain.json (= 2 chain)')
    cmb = J('s1_gamma1d_comb'); add('5 families', '1-d comb: max |D/(4 k0^2) - 1|', max(abs(c['D'] / c['four_k0_sq'] - 1) for c in cmb if c['spacing_over_sigma'] <= 3), 's1_gamma1d_comb.json:[spacing<=3]')
    VP = J('v1_vperiodic'); vr = sorted([r for r in VP['rows'] if r['k0'] == 1.5 and r['lam'] == 3.0 and r['M'] >= 100], key=lambda r: r['M'])
    add('5 families', 'V-periodic strip k0 = 1.5: B2 at the largest M (limit 1/(4 k0^2) = 0.1111)', vr[-1]['B2'], 'v1_vperiodic.json:rows[k0=1.5,lam=3,M=%d].B2' % vr[-1]['M'])
    # ---- 6. cancellation-free gap: unaffected values, artefact
    RV = J('w_recheck_V')
    for r in RV['finite_strips']: add('6 gap formula', 'finite strip sigma = %g (stage V optimum), gap/sigma' % r['sigma'], r['new_ibp'], 'w_recheck_V.json:finite_strips[sigma=%g].new_ibp' % r['sigma'])
    add('6 gap formula', 'stage-U ripples: max rel. difference old/new point fields', max(r['rel_new_vs_old'] for r in RV['ripple_U']), 'w_recheck_V.json:ripple_U')
    # ---- 7. strip family
    K = J('w1_klimit')
    for r in K['rows']:
        if r['k0'] <= 6: add('7 strips', 'sigma -> 0 (sigma = 1e-80): min gap/sigma at k0 = %g (1/(2k0) = %.4f)' % (r['k0'], 1 / (2 * r['k0'])), r['opt_gap_over_sigma'], 'w1_klimit.json:rows[k0=%g].opt_gap_over_sigma' % r['k0'])
    r7 = [r for r in K['rows'] if r['k0'] == 7][0]; add('7 strips', 'k0 = 7 at sigma = 1e-80 (too expensive)', r7['opt_gap_over_sigma'], 'w1_klimit.json:rows[k0=7]')
    SW = J('summary_W')
    X4v = {'%g' % s['parameters']['sigma']: s['A_x_indep']['gap_over_sigma'] for s in J('x4_states_for_main')['states']}
    for s, g in sorted(SW['gmin_best'].items(), key=lambda t: -float(t[0])):
        if s == '0.0001':   # stage W listed the capped-grid u_cgm value 1.1484; the converged value (x_indep = full grid = main line) is used (stage Y)
            add('7 strips', 'g_min(sigma = %s)/sigma (converged; the u_cgm search grid, capped at 8e6 points, gave %.4f)' % (s, g), X4v['0.0001'], 'x4_states_for_main.json:states[sigma=0.0001].A_x_indep.gap_over_sigma')
        else: add('7 strips', 'g_min(sigma = %s)/sigma' % s, g, 'summary_W.json:gmin_best[%s]' % s)
    f = SW['fit_opt_loglaw']; add('7 strips', 'log-law fit A (g = A/sqrt(ln(1/sigma) - B), sigma <= 1e-24)', f['A'], 'summary_W.json:fit_opt_loglaw.A')
    add('7 strips', 'log-law fit B', f['B'], 'summary_W.json:fit_opt_loglaw.B'); add('7 strips', 'log-law fit max residual', f['max_rel_resid'], 'summary_W.json:fit_opt_loglaw.max_rel_resid')
    for e in SW['local_exponent']: add('7 strips', 'local exponent d ln g/d ln ln(1/sigma), %g -> %g' % tuple(e['between']), e['value'], 'summary_W.json:local_exponent')
    add('7 strips', 'sigma below which strips beat 0.884 sigma', SW['cross_0884']['sigma_cross'], 'summary_W.json:cross_0884.sigma_cross')
    X4 = J('x4_states_for_main')
    for s in X4['states']: add('7 strips', 'X4 state sigma = %g: gap/sigma (two code-disjoint evaluators: max rel. diff %.1e)' % (s['parameters']['sigma'], max(s['rel_A_vs_B'].values())), s['A_x_indep']['gap_over_sigma'], 'x4_states_for_main.json:states[sigma=%g].A_x_indep.gap_over_sigma' % s['parameters']['sigma'])
    # ---- 8. N, Theorem W', Lemma X2
    W2 = J('w2_states'); add('8 theorem', 'max |N2/N - 1| (second-order form of N)', max(abs(v['N2_over_N'] - 1) for v in W2.values()), 'w2_states.json:N2_over_N')
    X1 = J('x1_states'); add('8 theorem', 'Theorem W\': states', len(X1), 'x1_states.json:len')
    add('8 theorem', 'Theorem W\': states where the bound exceeds the gap', sum(not v['bound_le_gap'] for v in X1.values()), 'x1_states.json:bound_le_gap')
    add('8 theorem', 'Theorem W\': max |eps|', max(abs(v['eps']) for v in X1.values()), 'x1_states.json:eps')
    add('8 theorem', 'Theorem W\': max eps_B/sqrt(sigma)', max(v['eps_B_absbound'] / np.sqrt(v['sigma']) for v in X1.values()), 'x1_states.json:eps_B_absbound/sqrt(sigma)')
    rb = [v['R3_over_sa'] / v['B_over_sa'] for k, v in X1.items() if (k.startswith('mix') or k.startswith('two') or k.startswith('stored')) and abs(v['B_over_sa']) > 1e-7]
    add('8 theorem', 'R3/B on mixtures and clusters: min', min(rb), 'x1_states.json'); add('8 theorem', 'R3/B on mixtures and clusters: max', max(rb), 'x1_states.json')
    L2 = J('x2_lemma'); add('8 theorem', 'Lemma X2: max ratio on random 1-d structures', L2['max_ratio_random'], 'x2_lemma.json:max_ratio_random')
    add('8 theorem', 'Lemma X2: random structures tested', len(L2['random']), 'x2_lemma.json:random')
    add('8 theorem', 'Lemma X2 transfer: max C on strip windows', max(t['C_needed'] for t in L2['transfer_strips']), 'x2_lemma.json:transfer_strips.C_needed max')
    # ---- 9. beyond the single strip
    for r in J('x3_verify'): add('9 beyond', '%s at sigma = %g: best/single strip - 1' % (r['family'], r['sigma']), r['x_indep_final'] / r['single_strip'] - 1, 'x3_verify.json:[%s,%g]' % (r['family'], r['sigma']))
    # ---- 10. stage Y3: R3 bound, X-A2 replaced by proved steps
    Y = J('y3_r3'); add('10 Y3', 'R3 bound holds on mixture/cluster states (count of %d)' % len(Y), sum(v['holds'] for v in Y.values()), 'y3_r3.json:holds')
    add('10 Y3', 'max |R3|/bound', max(v['ratio_R3_to_bound'] for v in Y.values()), 'y3_r3.json:ratio_R3_to_bound max')
    add('10 Y3', 'max eps_R3 bound (state quantities)', max(v['eps_R3_bound'] for v in Y.values()), 'y3_r3.json:eps_R3_bound max')
    A = J('y3_a2'); add('10 Y3', 'P# >= x^2/(1 + Dbar): max bound/P# over %d states' % A['chain_ii']['n'], A['chain_ii']['max_ratio'], 'y3_a2.json:chain_ii.max_ratio')
    add('10 Y3', 'min x^2/(1 + Dbar) in y/sigma <= 1/2', A['chain_ii']['region_min_bound'], 'y3_a2.json:chain_ii.region_min_bound')
    add('10 Y3', 'line bound D <= 8 xl + 2 sigma^4 E[s^4]/omega: max ratio (300 1-d densities)', A['line_iii']['max_ratio'], 'y3_a2.json:line_iii.max_ratio')
    add('10 Y3', 'observed max D/(1 + 8 xl)', A['line_iii']['max_D_over_1p8xl'], 'y3_a2.json:line_iii.max_D_over_1p8xl')
    add('10 Y3', "Lemma X2' (second spectral moment): max ratio", A['lemma_iv']['max_ratio'], 'y3_a2.json:lemma_iv.max_ratio')
    # ---- 11. stage Z: the conditional log-law theorem (Theorem Z)
    SZ = J('summary_Z'); tc = SZ['theorem_constants_strips']
    add('11 Z', 'Theorem Z on the strip family: K_Lambda max (enters c)', tc['K1_max_K_Lambda'], 'summary_Z.json:theorem_constants_strips.K1_max_K_Lambda')
    add('11 Z', 'Theorem Z on the strip family: K_J K_rho max (enters C)', tc['K2_max_KJ_Krho'], 'summary_Z.json:theorem_constants_strips.K2_max_KJ_Krho')
    add('11 Z', 'Theorem Z constant c = 1/(2 pi sqrt K_Lambda)', tc['c'], 'summary_Z.json:theorem_constants_strips.c')
    add('11 Z', 'Theorem Z constant C = ln(2 (pi^2 + 1) K_J K_rho / e)/2', tc['C'], 'summary_Z.json:theorem_constants_strips.C')
    add('11 Z', 'log-law coefficient A of the strips divided by c (looseness of the uniform constant)', tc['A_over_c'], 'summary_Z.json:theorem_constants_strips.A_over_c')
    for r in SZ['strips']:
        add('11 Z', 'strip sigma = %g: gap/sigma over the Theorem Z curve' % r['sigma'], r['gap_over_theorem'], 'summary_Z.json:strips[sigma=%g].gap_over_theorem' % r['sigma'])
    for r in SZ['strips']:
        if r['sigma'] in (1e-12, 1e-96):
            for q, nm in (('K_Lambda', 'K_Lambda'), ('xbar_w', 'xbar_w'), ('Xi', 'Xi'), ('bound_state', 'Theorem Z bound with state constants (gap/sigma)')):
                add('11 Z', 'strip sigma = %g: %s' % (r['sigma'], nm), r[q], 'summary_Z.json:strips[sigma=%g].%s' % (r['sigma'], q))
    for fam, nm in (('steps_x1', 'x1 states'), ('steps_stored', 'stored + adversarial states'), ('steps_strips', 'strips')):
        T = SZ[fam]; viol = sum(e.get(s, {}).get('violations', 0) for e in T.values() for s in ('S0', 'S1', 'S2', 'S3', 'S5', 'S6', 'S8', 'S56'))
        add('11 Z', 'step violations (steps 0-8) on the %s (%d states)' % (nm, sum(e['n'] for e in T.values())), viol, 'summary_Z.json:%s' % fam)
    v = SZ['vs_t1_chain']; add('11 Z', 'Lambda: max rel. difference to the stage-T code over %d states' % v['n'], v['max_rel_Lambda'], 'summary_Z.json:vs_t1_chain.max_rel_Lambda')
    add('11 Z', 'Lemma Z1 identity Lambda = 4 xbar_w(sqrt p) by FFT: max rel. deviation', SZ['identity']['max_rel_Lambda_4xbar_sqrtp'], 'summary_Z.json:identity')
    add('11 Z', 'lattice: max K_Lambda', SZ['lattice']['K_Lambda_range'][1], 'summary_Z.json:lattice.K_Lambda_range')
    for f, e in sorted(SZ['K_by_family'].items()):
        add('11 Z', '%s: K_Lambda range max (y/sigma <= 1/2)' % f, e['K_Lambda_max_y_le_half'], 'summary_Z.json:K_by_family[%s].K_Lambda_max_y_le_half' % f)
    # ---- 12. stage AA: Theorem Z' (K_J, K_rho removed), K_Lambda probed, eps_m-free factor
    SA = J('summary_AA'); st = SA['aa1_steps']
    add('12 AA', "AA1 steps (split Jensen, cut, kernel ratio, L2 Heisenberg, final): violations over %d evaluations" % st['final']['n'], sum(st[s]['violations'] for s in ('A_jensen', 'B_cut', 'C_kernel', 'D_heis_L2', 'final')), 'summary_AA.json:aa1_steps')
    add('12 AA', 'L2 Heisenberg (D): min ratio |d_w p|^2 8 E_Q[y_z^2]/|p|^2', st['D_heis_L2']['min_ratio'], 'summary_AA.json:aa1_steps.D_heis_L2.min_ratio')
    add('12 AA', 'max E_Q[y_z^2] over all states (H0 bound pi^2 + sigma^2/2)', SA['E_Q_max'], 'summary_AA.json:E_Q_max')
    add('12 AA', "Theorem Z' (state constants): min gap/bound over the states with the gap known", SA['gap_over_bound_state_min'][0], 'summary_AA.json:gap_over_bound_state_min')
    add('12 AA', "Theorem Z', t = 1/2: c sqrt(K1)", SA['constants']['t_half']['c_over_invsqrtK1'], 'summary_AA.json:constants.t_half')
    add('12 AA', "Theorem Z', t = 1/2: C", SA['constants']['t_half']['C'], 'summary_AA.json:constants.t_half')
    add('12 AA', 'eps_m-free factor kappa_u (sigma -> 0)', SA['constants']['kappa_uniform_sigma_to_0'], 'summary_AA.json:constants.kappa_uniform_sigma_to_0')
    add('12 AA', 'AA3: r <= sqrt(12 V_z) Y, min ratio', st['AA3_r']['min_ratio'], 'summary_AA.json:aa1_steps.AA3_r')
    add('12 AA', 'AA3: G >= (1 - eps) Y, min ratio', st['AA3_GY']['min_ratio'], 'summary_AA.json:aa1_steps.AA3_GY')
    for r in SA['product_family']:
        add('12 AA', 'product family sigma = %g: K_Lambda (y/sigma = %.2f, sqrt(P_w) >= %.3g)' % (r['sigma'], r['y_over_sigma'], r['sqrtPw_lower']), r['K_Lambda'], 'aa_product.json:[sigma=%g].K_Lambda' % r['sigma'])
    for k in ('KL', 'KL_gap'):
        b = SA['adv_best'][k]
        if b: add('12 AA', 'adversarial %s, best (sigma = %g): %s' % (k, b['sigma'], 'F_log' if k == 'F' else 'K_Lambda'), b['F_log'] if k == 'F' else b['K_Lambda'], 'summary_AA.json:adv_best.%s' % k)
    add('12 AA', 'stage-Z states with sqrt(P_w) <= 1 and y/sigma <= 1/2: max K_Lambda', SA['K_Lambda_max_small_gap_stored'], 'summary_AA.json:K_Lambda_max_small_gap_stored')
    # ---- 13. stage AB: H0 removed (Theorem Z'')
    SB = J('summary_AB')
    add('13 AB', "Theorem Z', t = 1/2: C as sigma -> 0", SB['t_half_C']['sigma0'], 'summary_AB.json:t_half_C.sigma0')
    add('13 AB', 'H0 removed, mixtures: max eps_pi (%d states)' % SB['H0_mixtures']['n'], SB['H0_mixtures']['max_eps_pi'], 'summary_AB.json:H0_mixtures.max_eps_pi')
    add('13 AB', 'H0 removed, mixtures: states with Var(X_z) > pi^2', SB['H0_mixtures']['n_Var_gt_pi2'], 'summary_AB.json:H0_mixtures.n_Var_gt_pi2')
    for k, v in SB['H0_strips'].items():
        if v['sigma'] in (0.01, 0.0001): add('13 AB', 'strip sigma = %g: log10 of the eps_pi bound' % v['sigma'], v['log10_eps_pi_bound'], 'summary_AB.json:H0_strips.%s' % k)
    # ---- 14. stage AC: Lemma AC1, directional de Bruijn
    SC = J('summary_AC')
    add('14 AC', 'Lemma AC1 E[q^2] <= 8 E[H_vw^2] + 2 J_ww/t: min ratio over %d states x 13 blurs' % SC['n_states'], SC['L1_min_ratio'][0], 'summary_AC.json:L1_min_ratio')
    add('14 AC', 'max t H_vv (bound 1)', SC['tHvv_max'], 'summary_AC.json:tHvv_max')
    add('14 AC', 'max sigma^4 E[q^2] (bound 14)', SC['q2_dimensionless_max'], 'summary_AC.json:q2_dimensionless_max')
    add('14 AC', 'directional de Bruijn dJ_ww/dt = -E[H_vw^2 + H_ww^2]: max rel. deviation (%d checks)' % SC['deBruijn_n'], SC['deBruijn_max_rel'][0], 'summary_AC.json:deBruijn_max_rel')
    r_lc, c_lc = lc_two_clusters()
    add('14 AC', 'log-convexity is not exact: two clusters (V2, sigma = 1e-3): R(sigma^2/4)/R(sigma^2)', r_lc, 'ac_heat_mix.json:twoclusters_V2_s0.001.rows[0.250, 1.000].R')
    add('14 AC', 'same state: smallest ratio R(t_i)/R(t_i+1) of consecutive blurs (t = sigma^2/4 -> 0.35 sigma^2)', c_lc, 'ac_heat_mix.json:twoclusters_V2_s0.001.rows[..].R')
    # ---- 15. stage AD: Gamma_2 identity
    SD = J('summary_AD')
    for k in ('G1', 'G2', 'G4'):
        add('15 AD', 'Gamma_2 identity %s: max rel. deviation (finite differences, 3 lattices x 3 blurs)' % k, SD['gamma2_identities'][k], 'summary_AD.json:gamma2_identities.%s' % k)
    # ---- 16. stage AE: valley lemma, 1-d and product-state theorem
    SE = J('summary_AE')
    for eta in ('0.05', '0.1', '0.2', '0.5'):
        add('16 AE', 'valley constant kappa_eta (Theorem AE3), eta = %s' % eta, SE['kappa_eta'][eta], 'summary_AE.json:kappa_eta.%s' % eta)
    add('16 AE', '1-d: largest N_eta/(tJ) found (searches K = 2-32), eta = 0.1', SE['oned']['max_N_eta_over_tJ'], 'summary_AE.json:oned.max_N_eta_over_tJ')
    add('16 AE', "1-d: largest t^2 R' found (searches K = 2-32)", SE['oned']['max_t2Rp'], 'summary_AE.json:oned.max_t2Rp')
    add('16 AE', "1-d lattices (81 points, spacing 1-8): largest t^2 R'", SE['oned']['lattice_1d']['max_t2Rp'], 'summary_AE.json:oned.lattice_1d.max_t2Rp')
    for k in ('1d_eta0.1_tau0.25', '1d_eta0.1_tau0.50'):
        for f in ('K', 'c0', 'B', 'c', 'C_sigma1'): add('16 AE', 'Theorem AE4 constants %s: %s' % (k, f), SE['constants'][k][f], 'summary_AE.json:constants.%s.%s' % (k, f))
    # ---- 17. stage AF: two-sided valley constant, strips without deep valleys
    SF = J('summary_AF')
    add('17 AF', 'two-sided valley constant kappa_0.1 (weight (c-1)^3)', SF['kappa_two_sided']['0.1'], 'summary_AF.json:kappa_two_sided.0.1')
    for q in ('t2R1', 'kappa_H', 'lambda_max'):
        add('17 AF', 'strip optima, t in {1/4, 1/2, 1} sigma^2: largest %s' % q, SF['strips_max'][q], 'summary_AF.json:strips_max.%s' % q)
    for k in ('1d_two_sided_eta0.1_tau0.25', '1d_two_sided_eta0.1_tau0.50'):
        for f in ('K', 'B', 'c', 'C_sigma1'): add('17 AF', 'constants %s: %s' % (k, f), SF['constants'][k][f], 'summary_AF.json:constants.%s.%s' % (k, f))
    # ---- 18. stage AG: strips on the whole window (Proposition AG1)
    SG = J('summary_AG')
    add('18 AG', 'strips: largest posterior excess lambda_max(-tH) over t in [sigma^2/4, sigma^2] and all y (8 optima, 25 blurs)', SG['strip_lambda_star'], 'summary_AG.json:strip_lambda_star')
    for k in ('quarter', 'half', 'one'):
        add('18 AG', 'strip sigma = 1e-2: lambda_max at t = %s sigma^2 (main line: %.3f)' % ({'quarter': '1/4', 'half': '1/2', 'one': '1'}[k], SG['main_line_check_sigma_0.01'][k][1]), SG['main_line_check_sigma_0.01'][k][0], 'summary_AG.json:main_line_check_sigma_0.01.%s' % k)
    for tau in ('0.25', '0.50'):
        for f in ('B', 'c', 'C_sigma1'): add('18 AG', 'Proposition AG1 on the strips (A = lambda*), tau = %s: %s' % (tau, f), SG['strip_theorem_tau%s' % tau][f], 'summary_AG.json:strip_theorem_tau%s.%s' % (tau, f))
    # ---- 19. (T2) on the stored states (stage AI)
    T2 = stored_T2_rows(); best = max(T2, key=lambda r: r[2] - r[3])
    add('19 C1a', 'stored state-blur rows (stage-AC heat-flow rows, stage-AD lattice ladders, stage-AF strip optima): count', len(T2), 'ad_qlc_lattice.json:rows (+ ac_heat_lattice/points/mix/strip.json, af_strip.json)')
    add('19 C1a', 'max over those rows of tR - ln(1/(t J_ww)) ((T2) with B\' = 1 needs A\' >= this on the stored states)', best[2] - best[3], '%s (t = %s)' % (best[0], best[1]))
    return E
