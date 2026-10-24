"""Stage AC: the Fisher-form smoothing lemma (sup F) along the heat flow.  Blur variance t (sigma^2 = t), state rho fixed.
Proved (stage AC):
  (L0) noise posterior: s = -E[Z|Y]/sigma, d_v d_w p/p = E[Z_v Z_w|Y]/sigma^2, H_vw = -Cov(Z_v, Z_w|Y)/sigma^2;  q := s_v s_w - 2 H_vw = (2E[Z_vZ_w|Y] - E[Z_v|Y]E[Z_w|Y])/sigma^2
  (L1) E[q^2] <= 8 E[H_vw^2] + 2 J_ww/t        (integration by parts, H_vv <= 1/t, Cauchy-Schwarz)   <=>  Lambda <= 2 + 8 t E[H_vw^2]/J_ww
  (L2) de Bruijn (directional):  dJ_ww/dt = -E[H_vw^2 + H_ww^2] =: -R(t) J_ww(t)
  (L3) if R is nonincreasing on [tau t, t] (log-convexity of J_ww along the heat flow, conjecture LC), then
       Lambda(t) <= 2 + 8/(1 - tau) * ln(1/(tau t J_ww(t)))        (J_ww(tau t) <= 1/(tau t))
This script evaluates, for each state and blur t = m sigma^2 (m on a grid), J_ww, E[H_vw^2], E[H_ww^2], E[q^2], R and the pieces of (L1);
it checks (L1), (L2) by finite differences and the monotonicity of R (LC).
States: mixtures of stage X1 (fields from p only, adaptive grid of stage Z), two-point configurations, the AB adversarial optima and the main-line
diamond 5x5 (grids), periodic lattices (exact cell averages, Poisson), ripple strips (two-scale, closed-form fields).
Usage: python ac_heat.py mix|points|lattice|strip -> results/ac_heat_<mode>.json (restartable by key).  Python 3.11."""
import sys, os, json, time
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from z_tools import fields, _prep

MS = (0.25, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
FD = 0.01


def sums_grid(T, t, vn, wn, rows=48):
    pre = _prep(T, np.sqrt(t))
    wv = np.zeros_like(vn); d = np.diff(vn); wv[1:] += d / 2; wv[:-1] += d / 2
    ww = np.zeros_like(wn); d = np.diff(wn); ww[1:] += d / 2; ww[:-1] += d / 2
    S = {k: 0.0 for k in ('m', 'J', 'Hvw2', 'Hww2', 'q2', 'sv2sw2', 'Hvv_sw2', 'svswHvw')}; hmax = -1e300
    for a in range(0, len(vn), rows):
        V, W = np.meshgrid(vn[a:a + rows], wn, indexing='ij'); x = np.stack([V.ravel(), W.ravel()], 1); dA = np.outer(wv[a:a + rows], ww).ravel()
        p, g, Hs = fields(pre, x); ok = p > 1e-300; p = np.where(ok, p, 1e-300)
        sv, sw = g[:, 0] / p, g[:, 1] / p; Hvv = -(Hs[:, 0, 0] / p - sv * sv); Hww = -(Hs[:, 1, 1] / p - sw * sw); Hvw = -(Hs[:, 0, 1] / p - sv * sw)
        q = sv * sw - 2 * Hvw; wt = p * dA * ok
        S['m'] += wt.sum(); S['J'] += np.sum(wt * sw * sw); S['Hvw2'] += np.sum(wt * Hvw ** 2); S['Hww2'] += np.sum(wt * Hww ** 2); S['q2'] += np.sum(wt * q * q)
        S['sv2sw2'] += np.sum(wt * sv * sv * sw * sw); S['Hvv_sw2'] += np.sum(wt * Hvv * sw * sw); S['svswHvw'] += np.sum(wt * sv * sw * Hvw)
        if ok.any(): hmax = max(hmax, float((Hvv[ok]).max() * t))
    m = S.pop('m'); E = {k: v / m for k, v in S.items()}; E['t_Hvv_max'] = hmax; E['mass'] = m; return E


def sums_lattice(B, t, N=128):
    B = np.asarray(B, float); Bi = np.linalg.inv(B); mm = np.fft.fftfreq(N, 1.0 / N); M1, M2 = np.meshgrid(mm, mm, indexing='ij')
    K = 2 * np.pi * np.einsum('ij,jab->iab', Bi.T, np.stack([M1, M2])); C = np.exp(-t * (K[0] ** 2 + K[1] ** 2) / 2) / abs(np.linalg.det(B))
    ifft = lambda X: np.real(np.fft.ifft2(X)) * N * N
    p = ifft(C); pv = ifft(1j * K[0] * C); pw = ifft(1j * K[1] * C); pvv = ifft(-K[0] ** 2 * C); pww = ifft(-K[1] ** 2 * C); pvw = ifft(-K[0] * K[1] * C)
    sv, sw = pv / p, pw / p; Hvv = -(pvv / p - sv * sv); Hww = -(pww / p - sw * sw); Hvw = -(pvw / p - sv * sw); q = sv * sw - 2 * Hvw; w = p / p.sum()
    E = {'J': np.sum(w * sw * sw), 'Hvw2': np.sum(w * Hvw ** 2), 'Hww2': np.sum(w * Hww ** 2), 'q2': np.sum(w * q * q), 'sv2sw2': np.sum(w * sv * sv * sw * sw),
         'Hvv_sw2': np.sum(w * Hvv * sw * sw), 'svswHvw': np.sum(w * sv * sw * Hvw), 't_Hvv_max': float(Hvv.max() * t), 'mass': 1.0, 'p_min_over_max': float(p.min() / p.max())}
    return {k: float(v) for k, v in E.items()}


def sums_strip(P, t, n_theta=64, n_slow=24, n_gl=12):
    """ripple strip with the physical state of P (ell, Lw, Lv, kappa) blurred with variance t (P['sig'] only fixes kappa = k0/sig)."""
    kap = P['k0'] / P['sig']; c, ell, Lw, Lv = P['c'], P['ell'], P['Lw'], P['Lv']
    Tv, Tw = Lv * Lv + t, Lw * Lw + t; Kv, Kw = Lv * Lv / Tv, Lw * Lw / Tw; sv2, sw2 = Lv * Lv * t / Tv, Lw * Lw * t / Tw
    G = c * np.exp(-0.5 * kap * kap * sv2); e2 = ell * ell + sw2; sc = np.sqrt(e2) / Kw; Rw = 9.0 * np.sqrt(Tw)
    edges = np.unique(np.concatenate([np.linspace(-Rw, Rw, 65), np.arange(-14 * sc, 14 * sc * (1 + 1e-9), sc / 3)])); edges = edges[(edges >= -Rw) & (edges <= Rw)]
    xg, wg = leggauss(n_gl); yw = ((edges[:-1, None] + edges[1:, None]) / 2 + (edges[1:, None] - edges[:-1, None]) / 2 * xg[None]).ravel()
    wyw = ((edges[1:, None] - edges[:-1, None]) / 2 * wg[None]).ravel() * np.exp(-yw * yw / (2 * Tw)) / np.sqrt(2 * np.pi * Tw)
    A = ell / np.sqrt(e2) * np.exp(-Kw * Kw * yw * yw / (2 * e2)); dA = -A * Kw * Kw * yw / e2; ddA = A * (Kw ** 4 * yw * yw / e2 ** 2 - Kw * Kw / e2)
    ts, wts = hermegauss(n_slow); wts = wts / wts.sum(); yvs = ts * np.sqrt(Tv); th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta; q0 = kap * Kv
    S = {k: 0.0 for k in ('m', 'J', 'Hvw2', 'Hww2', 'q2', 'sv2sw2', 'Hvv_sw2', 'svswHvw')}; hmax = -1e300
    for yv, wv in zip(yvs, wts):
        for tt in th:
            ct, st = np.cos(tt), np.sin(tt); F = G * A * ct; den = 1 + F
            Fv = -G * A * q0 * st; Fw = G * dA * ct; Fvv = -G * A * q0 * q0 * ct; Fww = G * ddA * ct; Fvw = -G * dA * q0 * st
            s_v = -yv / Tv + Fv / den; s_w = -yw / Tw + Fw / den
            Hvv = 1 / Tv - (Fvv * den - Fv * Fv) / den ** 2; Hww = 1 / Tw - (Fww * den - Fw * Fw) / den ** 2; Hvw = -(Fvw * den - Fv * Fw) / den ** 2
            q = s_v * s_w - 2 * Hvw; W = wv / n_theta * wyw * den
            S['m'] += W.sum(); S['J'] += np.sum(W * s_w * s_w); S['Hvw2'] += np.sum(W * Hvw ** 2); S['Hww2'] += np.sum(W * Hww ** 2); S['q2'] += np.sum(W * q * q)
            S['sv2sw2'] += np.sum(W * s_v * s_v * s_w * s_w); S['Hvv_sw2'] += np.sum(W * Hvv * s_w * s_w); S['svswHvw'] += np.sum(W * s_v * s_w * Hvw)
            hmax = max(hmax, float(np.max(Hvv) * t))
    m = S.pop('m'); E = {k: v / m for k, v in S.items()}; E['t_Hvv_max'] = hmax; E['mass'] = m; return E


def profile(fun, s2):
    """fun(t) -> sums.  Blur grid t = m s2, plus +-1% at m = 0.5 and 1 for the de Bruijn finite difference."""
    rows = {}
    for m in MS: rows['%.3f' % m] = fun(m * s2)
    for m in (0.5, 1.0):
        for e in (-FD, FD): rows['%.3f' % (m * (1 + e))] = fun(m * (1 + e) * s2)
    out = {'s2': s2, 'rows': rows}
    for key, E in rows.items():
        t = float(key) * s2; E['t'] = t; E['R'] = (E['Hvw2'] + E['Hww2']) / E['J']; E['Lambda'] = t * E['q2'] / E['J']; E['F'] = float(np.log(t * E['J']) + E['Lambda'] / 4)
        E['L1'] = [8 * E['Hvw2'] + 2 * E['J'] / t, E['q2']]; E['IBP_i'] = abs(E['sv2sw2'] - E['Hvv_sw2'] - 2 * E['svswHvw']) / E['sv2sw2']
    deb = {}
    for m in (0.5, 1.0):
        a, b = rows['%.3f' % (m * (1 - FD))], rows['%.3f' % (m * (1 + FD))]; c = rows['%.3f' % m]
        dJ = c['J'] * np.log(b['J'] / a['J']) / (b['t'] - a['t'])            # central difference of ln J (J ~ exponential in t for fine ripples)
        deb['%.1f' % m] = {'dJdt_fd': dJ, 'minus_E_H2': -(c['Hvw2'] + c['Hww2']), 'rel': abs(dJ / (-(c['Hvw2'] + c['Hww2'])) - 1)}
    out['deBruijn'] = deb
    Rs = [rows['%.3f' % m]['R'] * m for m in MS]                                  # R t (scale-free) -- LC needs R(t) nonincreasing; report R itself
    Rt = [rows['%.3f' % m]['R'] for m in MS]
    out['R_profile'] = Rt; out['LC_monotone'] = bool(all(Rt[i + 1] <= Rt[i] * (1 + 1e-6) for i in range(len(Rt) - 1)))
    out['LC_min_ratio'] = float(min(Rt[i] / Rt[i + 1] for i in range(len(Rt) - 1)))
    # (L3) at t = s2 with tau = 1/2 and tau = 1/4, and the average form actually needed
    c1 = rows['1.000']
    for tau in (0.25, 0.5):
        Jt = rows['%.3f' % tau]['J']; avgR = np.log(Jt / c1['J']) / ((1 - tau) * s2)
        out['L3_tau%.2f' % tau] = {'Lambda': c1['Lambda'], 'bound_LC': 2 + 8 / (1 - tau) * np.log(1 / (tau * s2 * c1['J'])), 'avg_R': avgR, 'R_at_1': c1['R'],
                                   'avg_condition': bool(avgR >= c1['R'] * (1 - 1e-9)), 'bound_from_avg': 2 + 8 * s2 * avgR if avgR >= c1['R'] else None}
    return out


def states_mix():
    from aa_chain import terms_x1
    from z_chain import mixture_grid
    Z = json.load(open('results/z_chain_x1.json')); J = []
    for k, z in Z.items():
        if k.startswith('strip') or k.startswith('ripple'): continue
        J.append((k, terms_x1(k, z['sigma']), z['sigma']))
    return J


def states_points():
    from ab_finite import points, build
    J = []
    for d, th in ((5.04, 39.8), (3.0, 45.0), (8.0, 45.0), (5.0, 10.0), (5.0, 80.0)):
        D = d * np.array([np.cos(np.radians(th)), np.sin(np.radians(th))]); J.append(('two_d%.2f_th%.0f' % (d, th), points([D / 2, -D / 2]), 1.0))
    b1, b2 = np.array([3.5, 3.34]), np.array([3.5, -3.34]); J.append(('diamond5', points([i * b1 + j * b2 for i in range(5) for j in range(5)]), 1.0))
    FI = json.load(open('results/ab_finite.json'))
    for k in ('adv_K3_start0', 'adv_K6_start1', 'adv_K9_start1'):
        J.append((k, build(FI[k]['x'], int(k[5])), 1.0))
    return J


if __name__ == '__main__':
    mode = sys.argv[1]; fn = 'results/ac_heat_%s.json' % mode; out = json.load(open(fn)) if os.path.exists(fn) else {}
    def save(): json.dump(out, open(fn, 'w'), indent=1, default=float)
    if mode in ('mix', 'points'):
        from z_chain import mixture_grid
        jobs = states_mix() if mode == 'mix' else states_points()
        for k, T, s in jobs:
            if k in out: continue
            t0 = time.time()
            def fun(t, T=T):
                vn, wn = mixture_grid(T, np.sqrt(t), 8 if mode == 'mix' else 10); return sums_grid(T, t, vn, wn)
            out[k] = profile(fun, s * s); out[k]['seconds'] = time.time() - t0; save()
            r = out[k]; print('%-40s LC %s (min ratio %.3f)  deBruijn rel %.1e  L1 min ratio %.2f  (%.0fs)' % (k, r['LC_monotone'], r['LC_min_ratio'], max(v['rel'] for v in r['deBruijn'].values()),
                  min(v['L1'][0] / v['L1'][1] for v in r['rows'].values()), r['seconds']), flush=True)
    elif mode == 'lattice':
        LA = json.load(open('results/ab_lattice.json')); LS = json.load(open('results/ab_lattice_scan.json'))
        bases = {'lattice_sup': np.array(LA['single_start0']['basis']).T, 'diamond_inf': np.array([[3.5, 3.5], [3.34, -3.34]]), 'square_aligned': np.array([[5.0, 0], [0, 5.0]])}
        for i, (D, ang) in enumerate(((2.0, 30.0), (1.5, 30.0), (1.2, 30.0), (1.0, 30.0), (0.85, 30.0), (0.75, 30.0), (1.0, 60.0), (0.8, 60.0), (1.0, 75.0), (0.8, 75.0), (1.0, 85.0), (0.8, 85.0), (0.7, 85.0), (0.8, 88.0))):
            e = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); ep = np.array([-e[1], e[0]])
            bases['comb_D%.2f_ang%.0f' % (D, ang)] = np.column_stack([0.25 * e, D * ep])   # dense along e (spacing 0.25), lines D apart: a tilted comb
        for k, B in bases.items():
            if k in out: continue
            t0 = time.time(); out[k] = profile(lambda t, B=B: sums_lattice(B, t, N=128 if not k.startswith('comb') else 192), 1.0); out[k]['basis'] = B.T.tolist(); out[k]['seconds'] = time.time() - t0; save()
            r = out[k]; c = r['rows']['1.000']; print('%-24s Lambda %.3f ln(1/J) %.3f F %.3f  LC %s  L1 %.2f' % (k, c['Lambda'], -np.log(c['J']), c['F'], r['LC_monotone'], c['L1'][0] / c['L1'][1]), flush=True)
    else:
        from z_strip import strip_params
        Z = json.load(open('results/z_chain_strips.json'))
        for k, z in Z.items():
            if k in out: continue
            P = strip_params(z['sigma'], z['k0'], z['lam'], z['M'], z['eta'], z['c']); t0 = time.time()
            with np.errstate(over='ignore'): out[k] = profile(lambda t, P=P: sums_strip(P, t), z['sigma'] ** 2)
            out[k]['seconds'] = time.time() - t0; save(); r = out[k]
            print('%-24s LC %s (min %.3f) deBruijn %.1e L1 %.2f' % (k, r['LC_monotone'], r['LC_min_ratio'], max(v['rel'] for v in r['deBruijn'].values()), min(v['L1'][0] / v['L1'][1] for v in r['rows'].values())), flush=True)
