"""Stage Z3: every step of the conditional log-law chain (Theorem Z) on every state family.
Usage:  python z_chain.py x1|stored|strips|lattice     -> results/z_chain_<mode>.json  (restartable by key; one output file per process)
  x1      : the 33 normalisable states of stage X1 (mixtures, stored sample, localised ripples, two clusters, strips at the g_min optima);
            gap, eps, a/den from results/x1_states.json -> the whole chain incl. step 0 (Theorem W') and the final bound
  stored  : all 377 states of the stage-S/T/U chain (361 stored + 16 adversarial) -> steps 1-8 and the bound on sqrt(P_w);
            Lambda and P# are compared with results/t1_chain.json (stage-T code)
  strips  : ripple strips at the g_min optima, sigma = 1e-2 ... 1e-96 (z_strip two-scale + closed form; gap by x_indep);
            sigma = 1e-2 and 1e-4 also on a full 2-d grid with complex Gaussian terms (z_tools; code-disjoint check)
  lattice : periodic checkerboard of stage U1 (non-normalisable: only K_Lambda and the Jensen equality apply), own closed form.
Steps (left >= right):  S0 gap >= (a/den) sqrt(P_w)(1 - eps);  S1 P_w >= (2/pi^2) P#;  S2 sqrt(P#) >= sqrt(P_y)(1 - eps_m);  S3 P_y >= 1/Lambda;
  S4 K_Lambda = Lambda/(4 xbar_w) (X-A1);  S5 ln(|d_w p^(w)|^2/|d_w p|^2) >= xbar_w;  S6 J_ww >= 1/(2 V_z);  S8 K_rho >= e sigma^2 R_w;  S56 Xi >= xbar_w.
Python 3.11."""
import sys, os, json, glob, time
import numpy as np
from z_tools import grid_sums, pair_sums, moments, chain
from z_strip import strip_params, strip_sums, strip_closed, strip_terms

RM = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0); TOP = np.array([np.pi, 0.0])


def qp_terms(ws, mus, Ss):
    return [(complex(np.log(w_)), RM @ (np.asarray(m, float) - TOP), RM @ np.asarray(S, float) @ RM.T) for w_, m, S in zip(ws, mus, Ss)]


def ripple_local_terms(sig, Lv, Lw, c, k0, Rv, ell, q0):
    """rho = N(0, diag(Lv^2, Lw^2)) [1 + c exp(-v^2/2Rv^2 - w^2/2ell^2) cos(k v) cos(q w)] as complex Gaussian terms (stage-U state definition)."""
    Sb = np.diag([Lv * Lv, Lw * Lw]); S2 = np.linalg.inv(np.linalg.inv(Sb) + np.diag([1 / Rv ** 2, 1 / ell ** 2])); amp = c * np.sqrt(np.linalg.det(S2) / np.linalg.det(Sb))
    k, q = k0 / sig, q0 / sig; waves = [(1, 0), (-1, 0)] if q0 == 0 else [(a, b) for a in (1, -1) for b in (1, -1)]; sh = 1.0 / len(waves)
    T = [(0j, np.zeros(2, complex), Sb)]
    for a, b in waves:
        kv = np.array([a * k, b * q]); T.append((np.log(amp * sh) - 0.5 * kv @ S2 @ kv + 0j, 1j * (S2 @ kv), S2))
    return T


def axis_nodes(cs, sds, hs, ext=8.0):
    """union of local uniform node sets: spacing h_k over centre_k +- ext sd_k (sd = marginal sd, h from the conditional sd: tilted components)."""
    pts = [np.arange(c - ext * s, c + ext * s + h / 2, h) for c, s, h in zip(cs, sds, hs)]
    x = np.sort(np.concatenate(pts)); thr = 0.3 * min(hs); out = [x[0]]
    for xi in x[1:]:                                      # greedy thinning against the last KEPT node
        if xi - out[-1] > thr: out.append(xi)
    return np.array(out)


def mixture_grid(T, sig, n_per_sd=12):
    s2 = sig * sig; cv, cw, sv, sw, hv, hw = [], [], [], [], [], []
    for t in T:
        Tm = np.asarray(t[2], float) + s2 * np.eye(2); cv.append(float(np.real(t[1][0]))); cw.append(float(np.real(t[1][1])))
        sv.append(np.sqrt(Tm[0, 0])); sw.append(np.sqrt(Tm[1, 1])); dt = np.linalg.det(Tm)
        hv.append(np.sqrt(dt / Tm[1, 1]) / n_per_sd); hw.append(np.sqrt(dt / Tm[0, 0]) / n_per_sd)
    return axis_nodes(cv, sv, hv), axis_nodes(cw, sw, hw)


def closed_of(T, sig):
    s2 = sig * sig
    try: rho = pair_sums(T, np.zeros((2, 2)))                      # |rho|^2 is infinite for components of zero width (point-like needles)
    except np.linalg.LinAlgError: rho = (np.inf, np.inf, np.inf)
    if not np.isfinite(rho[0]) or rho[0] <= 0: rho = (np.inf, np.inf, np.inf)
    return {'full': pair_sums(T, s2 * np.eye(2)), 'wonly': pair_sums(T, np.diag([0.0, s2])), 'rho': rho}


def eval_terms(T, sig, vn, wn, **kw):
    E, L, m = grid_sums(T, sig, vn, wn); r = chain(sig, E, L, closed_of(T, sig), moments(T), **kw); r['grid_mass'] = m; r['n_points'] = len(vn) * len(wn); return r


def jobs_x1():
    from h2_adversarial import unpack
    from q2_uncert import all_states
    from u2_chain import two_clusters
    X = json.load(open('results/x1_states.json')); S = all_states(); J = []
    for key, r in X.items():
        if key.startswith('strip_vlimit'): continue
        J.append((key, r))
    return J, S, unpack, two_clusters


def run_x1(key, r, S, unpack, two_clusters):
    sig = r['sigma']; kw = dict(gap=r['gap_over_sigma'], den_factor=r['den_factor'], eps=r['eps'])
    if key.startswith('mix_'):
        o = json.load(open('results/' + key[4:])); T = qp_terms(*unpack(np.array(o['x']), o['K'], sig))
    elif key.startswith('stored'):
        tag, ws, mus, Ss, sg = S[int(key[6:])]; T = qp_terms(ws, mus, Ss)
    elif key.startswith('twoclusters'):
        V0 = float(key.split('_V')[1].split('_')[0]); T = qp_terms(*two_clusters(V0, sig))
    elif key.startswith('ripple_'):
        o = json.load(open('results/' + key[7:])); rr = [q for q in json.load(open('results/u3_ripple.json'))['ripple_local'] if q['file'] == key[7:]][0]
        par = dict(o['best']['par']); q0 = rr['par_units']['q0_over_sqrt_sigma'] * np.sqrt(sig)
        T = ripple_local_terms(sig, par['Lv'], par['Lw'], par['c'], par['k0'], par['Rv'], par['ell'], q0)
        per = 2 * np.pi * sig / par['k0']; hv = min(0.2 * sig, per / 24); hw = 0.35 * min(par['ell'] / 40, (2 * np.pi * sig / q0) / 24 if q0 > 0 else 1e9, 0.5 * np.sqrt(sig))
        vn = np.arange(-6.5 * par['Lv'], 6.5 * par['Lv'] + hv / 2, hv); wn = np.arange(-6.5 * par['Lw'], 6.5 * par['Lw'] + hw / 2, hw)
        out = eval_terms(T, sig, vn, wn, **kw); out['P_w_x1'] = r['P_w']; return out
    elif key.startswith('strip_gmin'):
        GI = json.load(open('results/w1_gmin_indep_opt.json'))[key.split('_')[-1]]
        P = strip_params(sig, GI['k0'], GI['lam'], GI['M'], GI['Lv_over_ell'], GI['c']); E, m = strip_sums(P); cl, vz = strip_closed(P)
        out = chain(sig, E, {'p2': np.nan, 'pw2': np.nan, 'pvw2': np.nan, 'pv2': np.nan}, cl, vz, **kw); out['P_w_x1'] = r['P_w']; return out
    vn, wn = mixture_grid(T, sig, 12); out = eval_terms(T, sig, vn, wn, **kw); out['P_w_x1'] = r['P_w']; return out


def jobs_stored():
    from h2_adversarial import unpack
    from q2_uncert import all_states
    S = all_states(); J = [('stored%d' % i, qp_terms(ws, mus, Ss), sig) for i, (tag, ws, mus, Ss, sig) in enumerate(S)]
    keys = {t['key'] for t in json.load(open('results/t1_chain.json'))}
    for k in sorted(keys):
        if k.startswith('stored'): continue
        o = json.load(open('results/' + k)); J.append((k, qp_terms(*unpack(np.array(o['x']), o['K'], o['sigma'])), o['sigma']))
    return J


def strip_jobs():
    J = []
    for s in ('0.01', '1e-05'):
        b = json.load(open('results/w1_gmin_grid_opt_s%s.json' % s))['best']; sig = float(s)
        J.append(('strip_gmin_%s' % s, sig, b['k0'], b['lam'], b['M'], b['Lv_over_sigma'] * sig / (b['lam'] * np.sqrt(sig)), b['c']))
    x = json.load(open('results/x4_states_for_main.json'))['states'][0]['parameters']
    J.append(('strip_gmin_0.0001', x['sigma'], x['k0'], x['lam'], x['M'], x['Lv'] / x['ell'], x['c']))
    GI = json.load(open('results/w1_gmin_indep_opt.json'))
    for k in ('1e-06', '1e-12', '1e-24', '1e-48', '1e-96'):
        r = GI[k]; J.append(('strip_gmin_%s' % k, r['sigma'], r['k0'], r['lam'], r['M'], r['Lv_over_ell'], r['c']))
    return J


def run_strip(job):
    from x_indep import multi_gap
    key, sig, k0, lam, M, eta, c = job; P = strip_params(sig, k0, lam, M, eta, c)
    G = multi_gap(sig, k0, [(c, 0.0, lam, 1, 0.0)], M, eta)
    X = json.load(open('results/x1_states.json')); eps = X[key]['eps'] if key in X else None
    E, m = strip_sums(P); cl, vz = strip_closed(P)
    out = chain(sig, E, {'p2': np.nan, 'pw2': np.nan, 'pvw2': np.nan, 'pv2': np.nan}, cl, vz, gap=G['gap_over_sigma'], den_factor=G['a'] / G['denominator'], eps=eps)
    out.update({'k0': k0, 'lam': lam, 'M': M, 'eta': eta, 'c': c, 'a_x_indep': G['a'], 'Jww_x_indep': G['J_ww'], 'Psharp_x_indep': G['P_sharp'], 'y_over_sigma_x_indep': G['y_over_sigma'],
                'eps_source': 'x1_states' if eps is not None else 'not computed: step 0 not checked; eps >= eps_eff', 'eps_eff': 1 - G['gap_over_sigma'] / ((G['a'] / G['denominator']) * np.sqrt(out['P_w']))})
    if eps is None: out.pop('S0')
    if sig >= 1e-4:                                                          # code-disjoint full-grid check
        T = strip_terms(P); per = 2 * np.pi * sig / k0
        vn = np.arange(-6.5 * P['Lv'], 6.5 * P['Lv'], per / 24); wn = np.arange(-7 * P['Lw'], 7 * P['Lw'], min(P['ell'] / 30, P['Lw'] / 60))
        g = eval_terms(T, sig, vn, wn)
        out['grid_check'] = {k: g[k] for k in ('a', 'J_ww', 'P_w', 'P_sharp', 'P_y', 'eps_m', 'Lambda', 'xbar_w', 'K_Lambda', 'K_J', 'R_w', 'K_rho', 'Xi', 'L2_rel_grid_vs_closed', 'n_points')}
        out['grid_check']['closed_rel_analytic_vs_pairs'] = max(abs(a_ / b_ - 1) for nm in ('full', 'wonly', 'rho') for a_, b_ in zip(cl[nm], closed_of(T, sig)[nm]))
        out['grid_check']['max_rel_twoscale_vs_grid'] = max(abs(out[k] / g[k] - 1) for k in ('a', 'J_ww', 'P_w', 'P_sharp', 'P_y', 'Lambda', 'K_Lambda', 'K_J'))
    return out


def lattice_rows():
    """p = 1 + A cos(k v) cos(q w) (sigma = 1), one period cell, own closed-form fields; K_Lambda = Lambda/(4 k0^2) since xbar_w = k0^2 exactly."""
    out = []
    for r in json.load(open('results/u1_lattice.json'))['rows']:
        k0, q0, c = r['k0'], r['q0'], r['c']; A = c * np.exp(-(k0 * k0 + q0 * q0) / 2); n = 256
        th = 2 * np.pi * np.arange(n) / n; a_, b_ = np.meshgrid(th, th, indexing='ij'); ca, sa, cb, sb = np.cos(a_), np.sin(a_), np.cos(b_), np.sin(b_)
        p = 1 + A * ca * cb; sv = -A * k0 * sa * cb / p; sw = -A * q0 * ca * sb / p; pvw = A * k0 * q0 * sa * sb
        Hvw = -(pvw / p - sv * sw); wt = p / p.sum()
        Jww = np.sum(wt * sw * sw); Lam = np.sum(wt * (sv * sw - 2 * Hvw) ** 2) / Jww
        out.append({'k0': k0, 'q0': q0, 'c': c, 'A': A, 'Lambda': float(Lam), 'Lambda_stageU': r['Lambda'], 'xbar_w': k0 * k0, 'K_Lambda': float(Lam / (4 * k0 * k0)),
                    'jensen': 'equality (single wavenumber in xi_v)', 'y_over_sigma': r['y_over_sigma']})
    return out


if __name__ == '__main__':
    mode = sys.argv[1]; fn = 'results/z_chain_%s.json' % mode; out = json.load(open(fn)) if os.path.exists(fn) else {}
    def save(): json.dump(out, open(fn, 'w'), indent=1, default=float)
    if mode == 'lattice':
        out = {'rows': lattice_rows()}; save(); print('lattice: K_Lambda range %.4f .. %.4f' % (min(r['K_Lambda'] for r in out['rows']), max(r['K_Lambda'] for r in out['rows']))); sys.exit()
    if mode == 'x1':
        J, S, unpack, tc = jobs_x1(); todo = [(k, (r, S, unpack, tc)) for k, r in J]; fun = lambda k, a: run_x1(k, *a)
    elif mode == 'stored':
        todo = [(k, (T, sg)) for k, T, sg in jobs_stored()]
        def fun(k, a):
            T, sg = a; vn, wn = mixture_grid(T, sg, 8); return eval_terms(T, sg, vn, wn)
    else:
        todo = [(j[0], j) for j in strip_jobs()]; fun = lambda k, a: run_strip(a)
    for k, a in todo:
        if k in out: continue
        t = time.time()
        try: r = fun(k, a)
        except (MemoryError, ValueError, np.linalg.LinAlgError, FloatingPointError) as e: r = {'error': str(e)}
        r['seconds'] = time.time() - t; out[k] = r; save()
        print('%-48s %s (%.0fs)' % (k, ' '.join('%s %.4g' % (q, r[q]) for q in ('P_w', 'P_y', 'Lambda', 'xbar_w', 'K_Lambda', 'K_J', 'Xi', 'sqrtPw_bound_state') if q in r), r['seconds']), flush=True)
