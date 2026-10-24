"""Stage X1: the remainder rho of Theorem W, term by term, on every state family.
Exact decomposition (all steps are integrations by parts; phi = 1 - cos m_z; den = sigma^2 (J_vv + J_ww); a = sigma^2 J_vv):
  2 sigma^2 J_ww + N = 2 sigma^2 E[s_w^2 (1 - phi/4)] + (sigma^2/2) E[s_v^2 phi] + B + R3
     B  = (sigma^2/2) E[s_v d_v phi - s_w d_w phi] = -sigma^4 E[s_z H_pz sin m_z]            (both forms computed: second method)
     R3 = N - N1,  N1 = E[sigma^2 H_pz phi]                                                     (posterior third and higher cumulants)
  AM-GM:  2 sigma^2 E[s_w^2 (1 - phi/4)] + (sigma^2/2) E[s_v^2 phi] >= sigma a sqrt(P_w),   P_w = 4 sigma^2 E[s_w^2 (1 - phi/4)] E[s_v^2 phi] / a^2
  =>  gap/sigma >= (a/den) sqrt(P_w) (1 - eps),   eps = -(B + R3) / (sigma a sqrt(P_w))   (the -E[s_w^2 phi] term is absorbed exactly: 1 - phi/4 >= 1/2).
Families: mixtures (K32 minimum, adversarial minima of stages S/T/U/V, a sample of the 361 stored states), localised ripples (stage U),
two clusters (stage U2), ripple strips (one-period limit and true V-envelope, incl. the g_min optima), multi-window states (stage X3 starts).
-> results/x1_states.json (restartable by key).  Python 3.11."""
import json, os, glob, time
import numpy as np
from u_cgm import mixture_terms, ripple_terms, nodes, trap_weights
from w2_tools import line_sums, line_sums_envelope
from h2_adversarial import unpack
from w_tools import terms_of, w_nodes

KX = ('svv', 'sww', 'Nibp', 'N1', 'phi_svv', 'phi_sww', 'ibp_rest', 'B_alt', 'B_abs', 'sww_w', 'u2svv', 'mz2')


def analyse(L, wn, sig, ww=None):
    ww = trap_weights(wn) if ww is None else ww; s2 = sig * sig
    T = {k: float(np.sum(L[k] * ww)) for k in L}; m = T['p']; E = {k: v / m for k, v in T.items()}
    a = s2 * E['svv']; den = s2 * (E['svv'] + E['sww']); gap = (2 * s2 * E['sww'] + E['Nibp']) / den / sig
    Pw = 4 * s2 * E['sww_w'] * E['phi_svv'] / a ** 2; B = s2 / 2 * E['ibp_rest']; B2 = -s2 * s2 * E['B_alt']; R3 = E['Nibp'] - E['N1']
    lhs = 2 * s2 * E['sww'] + E['Nibp']; rhs = 2 * s2 * E['sww_w'] + s2 / 2 * E['phi_svv'] + B + R3
    slack = 2 * s2 * E['sww_w'] + s2 / 2 * E['phi_svv'] - sig * a * np.sqrt(Pw)
    eps = -(B + R3) / (sig * a * np.sqrt(Pw)); bound = (a / den) * np.sqrt(Pw) * (1 - max(eps, 0.0))
    return {'gap_over_sigma': gap, 'a': a, 'den_factor': a / den, 'P_w': Pw, 'sqrtP_w': float(np.sqrt(Pw)), 'B_over_sa': B / (sig * a), 'B_alt_over_sa': B2 / (sig * a),
            'B_forms_rel': abs(B2 - B) / max(abs(B), abs(B2), 1e-300), 'R3_over_sa': R3 / (sig * a), 'slack_over_sa': slack / (sig * a), 'eps': eps,
            'eps_B_absbound': s2 * s2 * E['B_abs'] / (sig * a * np.sqrt(Pw)), 'identity_rel': abs(rhs / lhs - 1), 'bound': bound, 'bound_le_gap': bool(bound <= gap * (1 + 1e-9)),
            'A_term_over_sa': -s2 / 2 * E['phi_sww'] / (sig * a), 'sigma': sig}


def grid_for_terms(T, sig, h_v=0.25, n_w=700):
    ev = max(abs(np.real(t[1][0])) + 6 * np.sqrt(np.real(t[2][0][0]) + sig ** 2) for t in T); ew = max(abs(np.real(t[1][1])) + 6 * np.sqrt(np.real(t[2][1][1]) + sig ** 2) for t in T)
    vn = nodes(ev, h_v * sig, ev, h_v * sig); wn = nodes(ew, ew / n_w, ew, ew / n_w)
    while len(vn) * len(wn) > 8e6: vn = nodes(ev, (len(vn) * len(wn) / 8e6) ** 0.5 * (vn[1] - vn[0]), ev, 1)
    return vn, wn


def jobs():
    J = []
    for fn in ['results/t4_K32_s0.01.json'] + sorted(glob.glob('results/t4_K16_*.json') + glob.glob('results/s1_psharp*_K*_s*.json'))[:6]:
        o = json.load(open(fn))
        if 'x' not in o: continue
        J.append(('mix_' + os.path.basename(fn), 'mixture', o['K'], o['sigma'], np.array(o['x'])))
    from q2_uncert import all_states
    S = all_states()
    for i in range(0, len(S), 36):
        tag, ws, mus, Ss, sig = S[i]; J.append(('stored%d' % i, 'mixture (stored)', (ws, mus, Ss), sig, None))
    for r in json.load(open('results/u3_ripple.json'))['ripple_local']:
        J.append(('ripple_' + r['file'], 'localised ripple (stage U)', r, r['sigma'], None))
    from u2_chain import two_clusters
    for sg in (0.01, 0.001):
        for V0 in (2, 20, 60): J.append(('twoclusters_V%g_s%g' % (V0, sg), 'two clusters', V0, sg, None))
    for (k0, lam, M, sg) in [(2.5, 1.5, 11943, 1e-24), (4.0, 1.0, 8.9e12, 1e-80)]: J.append(('strip_vlimit_k%g' % k0, 'strip (one-period limit)', (k0, lam, M), sg, None))
    GI = json.load(open('results/w1_gmin_indep_opt.json'))
    for key in ('1e-06', '1e-12', '1e-24'):
        r = GI[key]; J.append(('strip_gmin_%s' % key, 'strip (true envelope, g_min optimum)', r, r['sigma'], None))
    return J


def run(job):
    key, fam, a1, sig, x = job
    if fam == 'mixture':
        ws, mus, Ss = unpack(x, a1, sig); T = mixture_terms(ws, mus, Ss); vn, wn = grid_for_terms(T, sig); L = line_sums(T, sig, vn, trap_weights(vn), wn); return analyse(L, wn, sig)
    if fam == 'mixture (stored)':
        T = mixture_terms(*a1); vn, wn = grid_for_terms(T, sig); L = line_sums(T, sig, vn, trap_weights(vn), wn); return analyse(L, wn, sig)
    if fam.startswith('localised ripple'):
        o = json.load(open('results/' + a1['file'])); par = dict(o['best']['par']); par['q0'] = a1['par_units']['q0_over_sqrt_sigma'] * np.sqrt(sig)
        T = ripple_terms(sig, par['Lv'], par['Lw'], par['c'], par['k0'], par['Rv'], par['ell'], par['q0'])
        hv = min(0.2 * sig, 2 * np.pi * sig / par['k0'] / 20); hw = 0.35 * min(par['ell'] / 40, (2 * np.pi * sig / par['q0']) / 24 if par['q0'] > 0 else 1e9, 0.5 * np.sqrt(sig))
        vn = nodes(6.5 * par['Lv'], hv, 6.5 * par['Lv'], hv); wn = nodes(6.5 * par['Lw'], hw, 6.5 * par['Lw'], hw)
        while len(vn) * len(wn) > 8.0e6: hw *= 1.1; wn = nodes(6.5 * par['Lw'], hw, 6.5 * par['Lw'], hw)
        L = line_sums(T, sig, vn, trap_weights(vn), wn); return analyse(L, wn, sig)
    if fam == 'two clusters':
        from u2_chain import two_clusters
        T = mixture_terms(*two_clusters(a1, sig)); vn, wn = grid_for_terms(T, sig); L = line_sums(T, sig, vn, trap_weights(vn), wn); return analyse(L, wn, sig)
    if fam == 'strip (one-period limit)':
        k0, lam, M = a1; bg = [(1.0, 0.0, lam * M)]; win = [(1.0, 0.0, lam, 1)]; T = terms_of(bg, win, k0, sig, lam / np.sqrt(sig) / 150); wn = w_nodes(bg, win, sig, 40, 400)
        per = 2 * np.pi * sig / k0; nv = 32; vn = (np.arange(nv) + 0.5) * per / nv - per / 2; L = line_sums(T, sig, vn, np.full(nv, per / nv), wn); return analyse(L, wn, sig)
    r = a1; bg = [(1.0, 0.0, r['lam'] * r['M'])]; win = [(r['c'], 0.0, r['lam'], 1)]; Lvs = r['Lv_over_ell'] * r['lam'] / np.sqrt(sig)
    T = terms_of(bg, win, r['k0'], sig, Lvs); wn = w_nodes(bg, win, sig, 40, 400)
    L = line_sums_envelope(T, sig, r['k0'], np.sqrt((Lvs * sig) ** 2 + sig ** 2), wn, n0=21); return analyse(L, wn, sig)


if __name__ == '__main__':
    import w2_tools; w2_tools.KEYS = tuple(sorted(set(w2_tools.KEYS) | set(KX)))
    fn = 'results/x1_states.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for job in jobs():
        if job[0] in out: continue
        t = time.time()
        try: r = run(job)
        except (MemoryError, ValueError, np.linalg.LinAlgError) as e: r = {'error': str(e)}
        r.update({'family': job[1], 'sigma': job[3]}); out[job[0]] = r
        json.dump(out, open(fn, 'w'), indent=1, default=float)
        print('%-45s %s (%.0fs)' % (job[0], ' '.join('%s %.4g' % (k, r[k]) for k in ('gap_over_sigma', 'sqrtP_w', 'eps', 'B_over_sa', 'R3_over_sa', 'identity_rel') if k in r), time.time() - t), flush=True)
