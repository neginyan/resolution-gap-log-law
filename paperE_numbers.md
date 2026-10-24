# Paper E — numbers and their sources

Every number below is computed by `paperE/code/paperE_data.py` from the files in `paperE/results/` only (source: `file.json:key`).
Symbols: pendulum kappa = 1; gap = (1 - R)/sigma with the cancellation-free formula 1 - R = (2 sigma^2 J_ww + N)/(sigma^2 (J_vv + J_ww)); a = sigma^2 J_vv;
P# = sigma^2 E[u^2 s_v^2] J_ww / a^2; p~ = omega-weighted W-marginal; Lambda, x as in stages S-T; strips: rho = N(v;0,Lv^2) N(w;0,Lw^2)(1 + c e^{-w^2/2 ell^2} cos(k0 v/sigma)).


## 1 mixtures

| quantity | value | source |
|---|---|---|
| min gap/sigma over Gaussian mixtures (K32 hourglass, sigma = 1e-2) | 0.8840 | `summary_T.json:T4_min.opt_P_sharp.gap` |
| min P# (same state) | 0.7914 | `summary_T.json:T4_min.opt_P_sharp.opt_P_sharp` |
| F of the same state | 0.8985 | `summary_T.json:T4_min.opt_P_sharp.F` |
| y/sigma of the same state | 0.2314 | `summary_T.json:T4_min.opt_P_sharp.y_over_sigma` |

## 2 chain

| quantity | value | source |
|---|---|---|
| states evaluated (stored + adversarial) | 377 | `t1_chain.json:len` |
| states with y/sigma <= 1/2 | 231 | `t1_chain.json:[y_over_sigma<=0.5]` |
| max Lambda in y/sigma <= 1/2 | 2.3795 | `t1_chain.json:[y<=0.5].Lambda max` |
| min x^2 J_ww/(a I(p~)) (bound_ptilde) in y/sigma <= 1/2 | 0.7619 | `t1_chain.json:[y<=0.5].bound_ptilde min` |
| max a I(p~)/J_ww in y/sigma <= 1/2 | 1.3074 | `t1_chain.json:[y<=0.5].aIpt_over_Jww max` |
| min x^2/Lambda (bound_Lambda) in y/sigma <= 1/2 | 0.4127 | `t1_chain.json:[y<=0.5].bound_Lambda min` |
| chain violations (bound_ptilde > P# or bound_Lambda > bound_ptilde), all states | 0 | `t1_chain.json:all` |

## 3 Var(phi)

| quantity | value | source |
|---|---|---|
| proved constant (sqrt(3/4) + 1.5 sqrt 2)^2 | 8.9242 | `formula (stage T1)` |
| largest Var(phi) found (adversarial 1-d mixtures) | 4.1804 | `t1_var1d.json:searches[mode=V].Var_phi max` |
| largest Var(phi) found (combs) | 4.4151 | `t1_var1d.json:comb.Var_phi max` |

## 4 lemma (x)

| quantity | value | source |
|---|---|---|
| states where |x - 1| <= sd_p~(mu) sqrt(I(p~)) fails (bound 1) | 0 | `t1_chain.json:x_minus_1, x_bound_ptilde` |
| max x_bound/sqrt(sigma) (bound 1, y/sigma <= 1/2) | 60.8014 | `t1_chain.json:[y<=0.5].x_bound_over_sqrt_sigma max` |
| states where |x - 1| <= new bound fails (bound 2, exact identity) | 0 | `u2_chain.json:states` |
| two clusters: max sd_p~(mu)/sigma (bound 1 is not uniform) | 59.9970 | `u2_chain.json:two_clusters.sd_ptilde_mu_over_sigma max` |
| two clusters: max new bound (bound 2 stays small) | 0.0383 | `u2_chain.json:two_clusters.x_bound_new max` |

## 5 families

| quantity | value | source |
|---|---|---|
| lattice: max rel. deviation of a I(p~)/J_ww from 4 k0^2 (c <= 0.1) | 3.984e-07 | `u1_lattice.json:rows[c<=0.1]` |
| lattice k0 = 1: a I(p~)/J_ww (4 k0^2 = 4) | 4.0000 | `u1_lattice.json:rows[k0=1,q0=0.3,c=0.1]` |
| lattice k0 = 2: a I(p~)/J_ww (4 k0^2 = 16) | 16.0000 | `u1_lattice.json:rows[k0=2,q0=0.3,c=0.1]` |
| lattice k0 = 3: a I(p~)/J_ww (4 k0^2 = 36) | 36.0000 | `u1_lattice.json:rows[k0=3,q0=0.3,c=0.1]` |
| normalised patches: max a I(p~)/J_ww | 11.6512 | `u1_patch.json:[sigma<0.5,q0<0.5] max` |
| localised ripples (y/sigma <= 1/2): min a I(p~)/J_ww | 3.8806 | `u3_ripple.json:ripple_local[Y0=0.5] min` |
| localised ripples: min gap/sigma (not counterexamples to the gap) | 6.8290 | `u3_ripple.json:ripple_local.gap_over_sigma min` |
| mixtures: max a I(p~)/J_ww (y/sigma <= 1/2) | 1.3074 | `t1_chain.json (= 2 chain)` |
| 1-d comb: max |D/(4 k0^2) - 1| | 0.0024 | `s1_gamma1d_comb.json:[spacing<=3]` |
| V-periodic strip k0 = 1.5: B2 at the largest M (limit 1/(4 k0^2) = 0.1111) | 0.1138 | `v1_vperiodic.json:rows[k0=1.5,lam=3,M=3000].B2` |

## 6 gap formula

| quantity | value | source |
|---|---|---|
| finite strip sigma = 0.01 (stage V optimum), gap/sigma | 2.4114 | `w_recheck_V.json:finite_strips[sigma=0.01].new_ibp` |
| finite strip sigma = 0.001 (stage V optimum), gap/sigma | 1.4887 | `w_recheck_V.json:finite_strips[sigma=0.001].new_ibp` |
| finite strip sigma = 1e-06 (stage V optimum), gap/sigma | 0.8237 | `w_recheck_V.json:finite_strips[sigma=1e-06].new_ibp` |
| stage-U ripples: max rel. difference old/new point fields | 6.439e-15 | `w_recheck_V.json:ripple_U` |

## 7 strips

| quantity | value | source |
|---|---|---|
| sigma -> 0 (sigma = 1e-80): min gap/sigma at k0 = 1.5 (1/(2k0) = 0.3333) | 0.3333 | `w1_klimit.json:rows[k0=1.5].opt_gap_over_sigma` |
| sigma -> 0 (sigma = 1e-80): min gap/sigma at k0 = 2 (1/(2k0) = 0.2500) | 0.2500 | `w1_klimit.json:rows[k0=2].opt_gap_over_sigma` |
| sigma -> 0 (sigma = 1e-80): min gap/sigma at k0 = 2.5 (1/(2k0) = 0.2000) | 0.2000 | `w1_klimit.json:rows[k0=2.5].opt_gap_over_sigma` |
| sigma -> 0 (sigma = 1e-80): min gap/sigma at k0 = 3 (1/(2k0) = 0.1667) | 0.1667 | `w1_klimit.json:rows[k0=3].opt_gap_over_sigma` |
| sigma -> 0 (sigma = 1e-80): min gap/sigma at k0 = 4 (1/(2k0) = 0.1250) | 0.1250 | `w1_klimit.json:rows[k0=4].opt_gap_over_sigma` |
| sigma -> 0 (sigma = 1e-80): min gap/sigma at k0 = 5 (1/(2k0) = 0.1000) | 0.1000 | `w1_klimit.json:rows[k0=5].opt_gap_over_sigma` |
| sigma -> 0 (sigma = 1e-80): min gap/sigma at k0 = 6 (1/(2k0) = 0.0833) | 0.0834 | `w1_klimit.json:rows[k0=6].opt_gap_over_sigma` |
| k0 = 7 at sigma = 1e-80 (too expensive) | 3.2772 | `w1_klimit.json:rows[k0=7]` |
| g_min(sigma = 0.01)/sigma | 1.8180 | `summary_W.json:gmin_best[0.01]` |
| g_min(sigma = 0.0001)/sigma (converged; the u_cgm search grid, capped at 8e6 points, gave 1.1484) | 1.1491 | `x4_states_for_main.json:states[sigma=0.0001].A_x_indep.gap_over_sigma` |
| g_min(sigma = 1e-05)/sigma | 0.9143 | `summary_W.json:gmin_best[1e-05]` |
| g_min(sigma = 1e-06)/sigma | 0.7403 | `summary_W.json:gmin_best[1e-06]` |
| g_min(sigma = 1e-12)/sigma | 0.3349 | `summary_W.json:gmin_best[1e-12]` |
| g_min(sigma = 1e-24)/sigma | 0.1794 | `summary_W.json:gmin_best[1e-24]` |
| g_min(sigma = 1e-48)/sigma | 0.1104 | `summary_W.json:gmin_best[1e-48]` |
| g_min(sigma = 1e-96)/sigma | 0.0728 | `summary_W.json:gmin_best[1e-96]` |
| log-law fit A (g = A/sqrt(ln(1/sigma) - B), sigma <= 1e-24) | 1.0241 | `summary_W.json:fit_opt_loglaw.A` |
| log-law fit B | 23.4034 | `summary_W.json:fit_opt_loglaw.B` |
| log-law fit max residual | 0.0115 | `summary_W.json:fit_opt_loglaw.max_rel_resid` |
| local exponent d ln g/d ln ln(1/sigma), 1e-06 -> 1e-12 | -1.1445 | `summary_W.json:local_exponent` |
| local exponent d ln g/d ln ln(1/sigma), 1e-12 -> 1e-24 | -0.9006 | `summary_W.json:local_exponent` |
| local exponent d ln g/d ln ln(1/sigma), 1e-24 -> 1e-48 | -0.7001 | `summary_W.json:local_exponent` |
| local exponent d ln g/d ln ln(1/sigma), 1e-48 -> 1e-96 | -0.6014 | `summary_W.json:local_exponent` |
| sigma below which strips beat 0.884 sigma | 7.116e-06 | `summary_W.json:cross_0884.sigma_cross` |
| X4 state sigma = 0.0001: gap/sigma (two code-disjoint evaluators: max rel. diff 2.9e-05) | 1.1491 | `x4_states_for_main.json:states[sigma=0.0001].A_x_indep.gap_over_sigma` |
| X4 state sigma = 1e-06: gap/sigma (two code-disjoint evaluators: max rel. diff 2.3e-05) | 0.7403 | `x4_states_for_main.json:states[sigma=1e-06].A_x_indep.gap_over_sigma` |
| X4 state sigma = 1e-12: gap/sigma (two code-disjoint evaluators: max rel. diff 4.3e-05) | 0.3349 | `x4_states_for_main.json:states[sigma=1e-12].A_x_indep.gap_over_sigma` |
| X4 state sigma = 1e-24: gap/sigma (two code-disjoint evaluators: max rel. diff 1.6e-04) | 0.1794 | `x4_states_for_main.json:states[sigma=1e-24].A_x_indep.gap_over_sigma` |

## 8 theorem

| quantity | value | source |
|---|---|---|
| max |N2/N - 1| (second-order form of N) | 0.0074 | `w2_states.json:N2_over_N` |
| Theorem W': states | 35 | `x1_states.json:len` |
| Theorem W': states where the bound exceeds the gap | 0 | `x1_states.json:bound_le_gap` |
| Theorem W': max |eps| | 0.0042 | `x1_states.json:eps` |
| Theorem W': max eps_B/sqrt(sigma) | 2.8916 | `x1_states.json:eps_B_absbound/sqrt(sigma)` |
| R3/B on mixtures and clusters: min | 0.4566 | `x1_states.json` |
| R3/B on mixtures and clusters: max | 0.5076 | `x1_states.json` |
| Lemma X2: max ratio on random 1-d structures | 0.9992 | `x2_lemma.json:max_ratio_random` |
| Lemma X2: random structures tested | 400 | `x2_lemma.json:random` |
| Lemma X2 transfer: max C on strip windows | 0.1427 | `x2_lemma.json:transfer_strips.C_needed max` |

## 9 beyond

| quantity | value | source |
|---|---|---|
| CH at sigma = 1e-12: best/single strip - 1 | 0.0011 | `x3_verify.json:[CH,1e-12]` |
| CH at sigma = 1e-24: best/single strip - 1 | 0.0008105 | `x3_verify.json:[CH,1e-24]` |
| H2 at sigma = 1e-12: best/single strip - 1 | 0.0018 | `x3_verify.json:[H2,1e-12]` |
| H2 at sigma = 1e-24: best/single strip - 1 | 0.0009682 | `x3_verify.json:[H2,1e-24]` |
| H3 at sigma = 1e-12: best/single strip - 1 | 0.0156 | `x3_verify.json:[H3,1e-12]` |
| H3 at sigma = 1e-24: best/single strip - 1 | 0.0058 | `x3_verify.json:[H3,1e-24]` |
| MW at sigma = 1e-12: best/single strip - 1 | 0.0007309 | `x3_verify.json:[MW,1e-12]` |
| MW at sigma = 1e-24: best/single strip - 1 | 0.000657 | `x3_verify.json:[MW,1e-24]` |

## 10 Y3

| quantity | value | source |
|---|---|---|
| R3 bound holds on mixture/cluster states (count of 13) | 13 | `y3_r3.json:holds` |
| max |R3|/bound | 0.5076 | `y3_r3.json:ratio_R3_to_bound max` |
| max eps_R3 bound (state quantities) | 0.0155 | `y3_r3.json:eps_R3_bound max` |
| P# >= x^2/(1 + Dbar): max bound/P# over 377 states | 0.5082 | `y3_a2.json:chain_ii.max_ratio` |
| min x^2/(1 + Dbar) in y/sigma <= 1/2 | 0.2258 | `y3_a2.json:chain_ii.region_min_bound` |
| line bound D <= 8 xl + 2 sigma^4 E[s^4]/omega: max ratio (300 1-d densities) | 0.4996 | `y3_a2.json:line_iii.max_ratio` |
| observed max D/(1 + 8 xl) | 0.4857 | `y3_a2.json:line_iii.max_D_over_1p8xl` |
| Lemma X2' (second spectral moment): max ratio | 0.9977 | `y3_a2.json:lemma_iv.max_ratio` |

## 11 Z

| quantity | value | source |
|---|---|---|
| Theorem Z on the strip family: K_Lambda max (enters c) | 0.9899 | `summary_Z.json:theorem_constants_strips.K1_max_K_Lambda` |
| Theorem Z on the strip family: K_J K_rho max (enters C) | 2.0631 | `summary_Z.json:theorem_constants_strips.K2_max_KJ_Krho` |
| Theorem Z constant c = 1/(2 pi sqrt K_Lambda) | 0.1600 | `summary_Z.json:theorem_constants_strips.c` |
| Theorem Z constant C = ln(2 (pi^2 + 1) K_J K_rho / e)/2 | 1.4017 | `summary_Z.json:theorem_constants_strips.C` |
| log-law coefficient A of the strips divided by c (looseness of the uniform constant) | 6.4019 | `summary_Z.json:theorem_constants_strips.A_over_c` |
| strip sigma = 0.01: gap/sigma over the Theorem Z curve | 27.8538 | `summary_Z.json:strips[sigma=0.01].gap_over_theorem` |
| strip sigma = 0.0001: gap/sigma over the Theorem Z curve | 23.4011 | `summary_Z.json:strips[sigma=0.0001].gap_over_theorem` |
| strip sigma = 1e-05: gap/sigma over the Theorem Z curve | 20.5404 | `summary_Z.json:strips[sigma=1e-05].gap_over_theorem` |
| strip sigma = 1e-06: gap/sigma over the Theorem Z curve | 18.0523 | `summary_Z.json:strips[sigma=1e-06].gap_over_theorem` |
| strip sigma = 1e-12: gap/sigma over the Theorem Z curve | 11.2793 | `summary_Z.json:strips[sigma=1e-12].gap_over_theorem` |
| strip sigma = 1e-24: gap/sigma over the Theorem Z curve | 8.4409 | `summary_Z.json:strips[sigma=1e-24].gap_over_theorem` |
| strip sigma = 1e-48: gap/sigma over the Theorem Z curve | 7.3023 | `summary_Z.json:strips[sigma=1e-48].gap_over_theorem` |
| strip sigma = 1e-96: gap/sigma over the Theorem Z curve | 6.7852 | `summary_Z.json:strips[sigma=1e-96].gap_over_theorem` |
| strip sigma = 1e-12: K_Lambda | 0.8494 | `summary_Z.json:strips[sigma=1e-12].K_Lambda` |
| strip sigma = 1e-12: xbar_w | 3.0776 | `summary_Z.json:strips[sigma=1e-12].xbar_w` |
| strip sigma = 1e-12: Xi | 4.7624 | `summary_Z.json:strips[sigma=1e-12].Xi` |
| strip sigma = 1e-12: Theorem Z bound with state constants (gap/sigma) | 0.1119 | `summary_Z.json:strips[sigma=1e-12].bound_state` |
| strip sigma = 1e-96: K_Lambda | 0.9899 | `summary_Z.json:strips[sigma=1e-96].K_Lambda` |
| strip sigma = 1e-96: xbar_w | 48.1720 | `summary_Z.json:strips[sigma=1e-96].xbar_w` |
| strip sigma = 1e-96: Xi | 52.5931 | `summary_Z.json:strips[sigma=1e-96].Xi` |
| strip sigma = 1e-96: Theorem Z bound with state constants (gap/sigma) | 0.0312 | `summary_Z.json:strips[sigma=1e-96].bound_state` |
| step violations (steps 0-8) on the x1 states (33 states) | 0 | `summary_Z.json:steps_x1` |
| step violations (steps 0-8) on the stored + adversarial states (377 states) | 0 | `summary_Z.json:steps_stored` |
| step violations (steps 0-8) on the strips (8 states) | 0 | `summary_Z.json:steps_strips` |
| Lambda: max rel. difference to the stage-T code over 377 states | 0.0009984 | `summary_Z.json:vs_t1_chain.max_rel_Lambda` |
| Lemma Z1 identity Lambda = 4 xbar_w(sqrt p) by FFT: max rel. deviation | 3.606e-05 | `summary_Z.json:identity` |
| lattice: max K_Lambda | 1.0387 | `summary_Z.json:lattice.K_Lambda_range` |
| grating mixture (adversarial): K_Lambda range max (y/sigma <= 1/2) | 0.6003 | `summary_Z.json:K_by_family[grating mixture (adversarial)].K_Lambda_max_y_le_half` |
| mixture (adversarial): K_Lambda range max (y/sigma <= 1/2) | 0.6896 | `summary_Z.json:K_by_family[mixture (adversarial)].K_Lambda_max_y_le_half` |
| mixture (stored): K_Lambda range max (y/sigma <= 1/2) | 0.9058 | `summary_Z.json:K_by_family[mixture (stored)].K_Lambda_max_y_le_half` |
| strip (g_min optimum): K_Lambda range max (y/sigma <= 1/2) | 0.9446 | `summary_Z.json:K_by_family[strip (g_min optimum)].K_Lambda_max_y_le_half` |

## 12 AA

| quantity | value | source |
|---|---|---|
| AA1 steps (split Jensen, cut, kernel ratio, L2 Heisenberg, final): violations over 830 evaluations | 0 | `summary_AA.json:aa1_steps` |
| L2 Heisenberg (D): min ratio |d_w p|^2 8 E_Q[y_z^2]/|p|^2 | 1.0000 | `summary_AA.json:aa1_steps.D_heis_L2.min_ratio` |
| max E_Q[y_z^2] over all states (H0 bound pi^2 + sigma^2/2) | 1.0369 | `summary_AA.json:E_Q_max` |
| Theorem Z' (state constants): min gap/bound over the states with the gap known | 5.8941 | `summary_AA.json:gap_over_bound_state_min` |
| Theorem Z', t = 1/2: c sqrt(K1) | 0.1125 | `summary_AA.json:constants.t_half` |
| Theorem Z', t = 1/2: C | 2.4023 | `summary_AA.json:constants.t_half` |
| eps_m-free factor kappa_u (sigma -> 0) | 0.1695 | `summary_AA.json:constants.kappa_uniform_sigma_to_0` |
| AA3: r <= sqrt(12 V_z) Y, min ratio | 1.4101 | `summary_AA.json:aa1_steps.AA3_r` |
| AA3: G >= (1 - eps) Y, min ratio | 2.4978 | `summary_AA.json:aa1_steps.AA3_GY` |
| product family sigma = 0.01: K_Lambda (y/sigma = 0.11, sqrt(P_w) >= 2.2) | 0.6971 | `aa_product.json:[sigma=0.01].K_Lambda` |
| product family sigma = 0.0001: K_Lambda (y/sigma = 0.19, sqrt(P_w) >= 29.7) | 1.6693 | `aa_product.json:[sigma=0.0001].K_Lambda` |
| product family sigma = 1e-06: K_Lambda (y/sigma = 0.27, sqrt(P_w) >= 360) | 5.2714 | `aa_product.json:[sigma=1e-06].K_Lambda` |
| product family sigma = 1e-09: K_Lambda (y/sigma = 0.38, sqrt(P_w) >= 1.71e+04) | 23.4969 | `aa_product.json:[sigma=1e-09].K_Lambda` |
| product family sigma = 1e-12: K_Lambda (y/sigma = 0.49, sqrt(P_w) >= 1.26e+06) | 42.7837 | `aa_product.json:[sigma=1e-12].K_Lambda` |
| adversarial KL, best (sigma = 0.01): K_Lambda | 3.0158 | `summary_AA.json:adv_best.KL` |
| adversarial KL_gap, best (sigma = 0.01): K_Lambda | 0.6822 | `summary_AA.json:adv_best.KL_gap` |
| stage-Z states with sqrt(P_w) <= 1 and y/sigma <= 1/2: max K_Lambda | 0.9899 | `summary_AA.json:K_Lambda_max_small_gap_stored` |

## 13 AB

| quantity | value | source |
|---|---|---|
| Theorem Z', t = 1/2: C as sigma -> 0 | 2.3776 | `summary_AB.json:t_half_C.sigma0` |
| H0 removed, mixtures: max eps_pi (407 states) | 0.2664 | `summary_AB.json:H0_mixtures.max_eps_pi` |
| H0 removed, mixtures: states with Var(X_z) > pi^2 | 0 | `summary_AB.json:H0_mixtures.n_Var_gt_pi2` |
| strip sigma = 0.01: log10 of the eps_pi bound | -3.5926 | `summary_AB.json:H0_strips.strip_gmin_0.01` |
| strip sigma = 0.0001: log10 of the eps_pi bound | -47.4713 | `summary_AB.json:H0_strips.strip_gmin_0.0001` |

## 14 AC

| quantity | value | source |
|---|---|---|
| Lemma AC1 E[q^2] <= 8 E[H_vw^2] + 2 J_ww/t: min ratio over 58 states x 13 blurs | 1.9818 | `summary_AC.json:L1_min_ratio` |
| max t H_vv (bound 1) | 1.0000 | `summary_AC.json:tHvv_max` |
| max sigma^4 E[q^2] (bound 14) | 2.3476 | `summary_AC.json:q2_dimensionless_max` |
| directional de Bruijn dJ_ww/dt = -E[H_vw^2 + H_ww^2]: max rel. deviation (46 checks) | 0.0011 | `summary_AC.json:deBruijn_max_rel` |
| log-convexity is not exact: two clusters (V2, sigma = 1e-3): R(sigma^2/4)/R(sigma^2) | 0.6985 | `ac_heat_mix.json:twoclusters_V2_s0.001.rows[0.250, 1.000].R` |
| same state: smallest ratio R(t_i)/R(t_i+1) of consecutive blurs (t = sigma^2/4 -> 0.35 sigma^2) | 0.8520 | `ac_heat_mix.json:twoclusters_V2_s0.001.rows[..].R` |

## 15 AD

| quantity | value | source |
|---|---|---|
| Gamma_2 identity G1: max rel. deviation (finite differences, 3 lattices x 3 blurs) | 0.0007509 | `summary_AD.json:gamma2_identities.G1` |
| Gamma_2 identity G2: max rel. deviation (finite differences, 3 lattices x 3 blurs) | 0.0003754 | `summary_AD.json:gamma2_identities.G2` |
| Gamma_2 identity G4: max rel. deviation (finite differences, 3 lattices x 3 blurs) | 0.0013 | `summary_AD.json:gamma2_identities.G4` |

## 16 AE

| quantity | value | source |
|---|---|---|
| valley constant kappa_eta (Theorem AE3), eta = 0.05 | 539.3970 | `summary_AE.json:kappa_eta.0.05` |
| valley constant kappa_eta (Theorem AE3), eta = 0.1 | 531.4474 | `summary_AE.json:kappa_eta.0.1` |
| valley constant kappa_eta (Theorem AE3), eta = 0.2 | 508.1990 | `summary_AE.json:kappa_eta.0.2` |
| valley constant kappa_eta (Theorem AE3), eta = 0.5 | 499.4715 | `summary_AE.json:kappa_eta.0.5` |
| 1-d: largest N_eta/(tJ) found (searches K = 2-32), eta = 0.1 | 2.2547 | `summary_AE.json:oned.max_N_eta_over_tJ` |
| 1-d: largest t^2 R' found (searches K = 2-32) | -0.3687 | `summary_AE.json:oned.max_t2Rp` |
| 1-d lattices (81 points, spacing 1-8): largest t^2 R' | -0.2415 | `summary_AE.json:oned.lattice_1d.max_t2Rp` |
| Theorem AE4 constants 1d_eta0.1_tau0.25: K | 1.0545 | `summary_AE.json:constants.1d_eta0.1_tau0.25.K` |
| Theorem AE4 constants 1d_eta0.1_tau0.25: c0 | 458.5556 | `summary_AE.json:constants.1d_eta0.1_tau0.25.c0` |
| Theorem AE4 constants 1d_eta0.1_tau0.25: B | 11.2480 | `summary_AE.json:constants.1d_eta0.1_tau0.25.B` |
| Theorem AE4 constants 1d_eta0.1_tau0.25: c | 0.0949 | `summary_AE.json:constants.1d_eta0.1_tau0.25.c` |
| Theorem AE4 constants 1d_eta0.1_tau0.25: C_sigma1 | 165.3928 | `summary_AE.json:constants.1d_eta0.1_tau0.25.C_sigma1` |
| Theorem AE4 constants 1d_eta0.1_tau0.50: K | 1.0310 | `summary_AE.json:constants.1d_eta0.1_tau0.50.K` |
| Theorem AE4 constants 1d_eta0.1_tau0.50: c0 | 207.3677 | `summary_AE.json:constants.1d_eta0.1_tau0.50.c0` |
| Theorem AE4 constants 1d_eta0.1_tau0.50: B | 16.4954 | `summary_AE.json:constants.1d_eta0.1_tau0.50.B` |
| Theorem AE4 constants 1d_eta0.1_tau0.50: c | 0.0784 | `summary_AE.json:constants.1d_eta0.1_tau0.50.c` |
| Theorem AE4 constants 1d_eta0.1_tau0.50: C_sigma1 | 52.2319 | `summary_AE.json:constants.1d_eta0.1_tau0.50.C_sigma1` |

## 17 AF

| quantity | value | source |
|---|---|---|
| two-sided valley constant kappa_0.1 (weight (c-1)^3) | 267.1559 | `summary_AF.json:kappa_two_sided.0.1` |
| strip optima, t in {1/4, 1/2, 1} sigma^2: largest t2R1 | -5.047e-09 | `summary_AF.json:strips_max.t2R1` |
| strip optima, t in {1/4, 1/2, 1} sigma^2: largest kappa_H | 0.0552 | `summary_AF.json:strips_max.kappa_H` |
| strip optima, t in {1/4, 1/2, 1} sigma^2: largest lambda_max | 1.8203 | `summary_AF.json:strips_max.lambda_max` |
| constants 1d_two_sided_eta0.1_tau0.25: K | 1.0545 | `summary_AF.json:constants.1d_two_sided_eta0.1_tau0.25.K` |
| constants 1d_two_sided_eta0.1_tau0.25: B | 11.2480 | `summary_AF.json:constants.1d_two_sided_eta0.1_tau0.25.B` |
| constants 1d_two_sided_eta0.1_tau0.25: c | 0.0949 | `summary_AF.json:constants.1d_two_sided_eta0.1_tau0.25.c` |
| constants 1d_two_sided_eta0.1_tau0.25: C_sigma1 | 84.2967 | `summary_AF.json:constants.1d_two_sided_eta0.1_tau0.25.C_sigma1` |
| constants 1d_two_sided_eta0.1_tau0.50: K | 1.0310 | `summary_AF.json:constants.1d_two_sided_eta0.1_tau0.50.K` |
| constants 1d_two_sided_eta0.1_tau0.50: B | 16.4954 | `summary_AF.json:constants.1d_two_sided_eta0.1_tau0.50.B` |
| constants 1d_two_sided_eta0.1_tau0.50: c | 0.0784 | `summary_AF.json:constants.1d_two_sided_eta0.1_tau0.50.c` |
| constants 1d_two_sided_eta0.1_tau0.50: C_sigma1 | 27.2248 | `summary_AF.json:constants.1d_two_sided_eta0.1_tau0.50.C_sigma1` |

## 18 AG

| quantity | value | source |
|---|---|---|
| strips: largest posterior excess lambda_max(-tH) over t in [sigma^2/4, sigma^2] and all y (8 optima, 25 blurs) | 1.8745 | `summary_AG.json:strip_lambda_star` |
| strip sigma = 1e-2: lambda_max at t = 1/4 sigma^2 (main line: 1.874) | 1.8745 | `summary_AG.json:main_line_check_sigma_0.01.quarter` |
| strip sigma = 1e-2: lambda_max at t = 1/2 sigma^2 (main line: 1.756) | 1.7564 | `summary_AG.json:main_line_check_sigma_0.01.half` |
| strip sigma = 1e-2: lambda_max at t = 1 sigma^2 (main line: 1.537) | 1.5370 | `summary_AG.json:main_line_check_sigma_0.01.one` |
| Proposition AG1 on the strips (A = lambda*), tau = 0.25: B | 23.4316 | `summary_AG.json:strip_theorem_tau0.25.B` |
| Proposition AG1 on the strips (A = lambda*), tau = 0.25: c | 0.0658 | `summary_AG.json:strip_theorem_tau0.25.c` |
| Proposition AG1 on the strips (A = lambda*), tau = 0.25: C_sigma1 | 2.2754 | `summary_AG.json:strip_theorem_tau0.25.C_sigma1` |
| Proposition AG1 on the strips (A = lambda*), tau = 0.50: B | 26.6268 | `summary_AG.json:strip_theorem_tau0.50.B` |
| Proposition AG1 on the strips (A = lambda*), tau = 0.50: c | 0.0617 | `summary_AG.json:strip_theorem_tau0.50.c` |
| Proposition AG1 on the strips (A = lambda*), tau = 0.50: C_sigma1 | 1.9237 | `summary_AG.json:strip_theorem_tau0.50.C_sigma1` |

## 19 C1a

| quantity | value | source |
|---|---|---|
| stored state-blur rows (stage-AC heat-flow rows, stage-AD lattice ladders, stage-AF strip optima): count | 1778 | `ad_qlc_lattice.json:rows (+ ac_heat_lattice/points/mix/strip.json, af_strip.json)` |
| max over those rows of tR - ln(1/(t J_ww)) ((T2) with B' = 1 needs A' >= this on the stored states) | 6.5440 | `ad_qlc_lattice.json:comb_D1.00_ang30 (t = 11.737651753019792)` |
