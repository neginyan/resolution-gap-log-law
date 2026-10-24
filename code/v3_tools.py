"""Stage V (main-line requests 3-5): two independent computations of the TRUE gap for complex-Gaussian-mixture states (u_cgm terms).
Both use the cancellation-free form (pendulum, kappa = 1; m = y + sigma^2 s; z = x_q - pi; s_p = (s_v - s_w)/sqrt2, m_z = (m_v + m_w)/sqrt2):
    1 - R = ( 2 sigma^2 J_ww + N ) / ( sigma^2 (J_vv + J_ww) ),    N := E[ s_p (m_z - E[sin z | y]) ] = -E[ d_p D ],  D := E[z - sin z | y]
(from E[s.Ef] = sigma^2 (J_vv - J_ww) - N, which uses E[s_v v] = E[s_w w] = -1 exactly; checked against l_tools.diag in u_cgm).
  A  grid_sums : tensor grid (non-uniform allowed, trapezoid), streamed over V-rows -- constant memory, any grid size;
  A' grid_sums on a UNIFORM grid also returns the direct form of R (independent of the integration-by-parts rewriting);
  B  mc_sums   : Monte Carlo without any grid.  With N itself the gap is hopeless (the per-sample s_p (m_z - E[sin z|y]) has sd ~1e-2 against a
                 mean ~1e-8: s_v ~ 1e5 times z^3 ~ 1e-6); it uses N = -E[d_p D] (one more exact integration by parts), which is smooth.: x ~ rho exactly (rejection from the background Gaussian; rho = N_b (1 + c W cos(k v) [cos(q w)])),
                 y = x + sigma Z, exact s, H and posterior moments at the sample points; standard errors from 20 batches.
Both also return j, Z, J_ww (-> P, d, F, y/sigma).  Python 3.11 compatible."""
import numpy as np
from u_cgm import trap_weights

SQ2 = np.sqrt(2.0)


def _pre(terms, sig):
    """terms: (log_coef, mean, S) as in u_cgm (complex mean allowed), or (log_coef, real centre mu, S, kappa) = a Gaussian times the plane wave
    e^{i kappa.(x - mu)} in native form.  The native form is evaluated without cancellation:
        N(y; mu + i S kappa, T) e^{-kappa'S kappa/2} = N(y; mu, T) exp( i (K kappa).(y - mu) - kappa'C kappa / 2 ),
        posterior mean m' = mu + K (y - mu) + i C kappa,  K = I - sigma^2 T^-1,  C = sigma^2 (I - sigma^2 T^-1)
    (with the complex-mean form, +k^2 S/2 and -k^2 S/2 cancel in floating point once k^2 S ~ 1e16: wide strips)."""
    s2 = sig * sig; pre = []
    Sref = np.asarray(terms[0][2], float)
    for t in terms:
        if len(t) == 4: lc, mu, S, kap = t; mu = np.asarray(mu, float); kap = np.asarray(kap, float)
        else: lc, m, S = t; m = np.asarray(m, complex); mu = m.real; kap = None
        T = S + s2 * np.eye(2); Ti = np.linalg.inv(T); K = np.eye(2) - s2 * Ti; C = s2 * (np.eye(2) - s2 * Ti)   # = S T^-1, S - S T^-1 S (cancellation-free)
        varz = (C[0, 0] + 2 * C[0, 1] + C[1, 1]) / 2
        if kap is None:                          # complex mean m = mu + i b  ->  equivalent native data with  i S kappa = i b
            b = np.asarray(m, complex).imag; kap = np.linalg.solve(S, b) if np.any(b != 0) else np.zeros(2)
            lc = lc + 0.5 * kap @ S @ kap         # the old log_coef contained -kappa'S kappa/2 explicitly
        c0 = lc - np.log(2 * np.pi * np.sqrt(np.linalg.det(T)))
        pre.append((c0, mu, Ti, K, varz, K @ kap, C @ kap, float(kap @ C @ kap), Sref, np.asarray(S, float)))
    return pre


def ripple_terms_k(sig, Lv, Lw, c, k0, Rv, ell, q0=0.0):
    """u_cgm.ripple_terms in the native plane-wave form (log_coef, centre 0, S, kappa) -- no k^2 S/2 in the coefficient."""
    Sb = np.diag([Lv ** 2, Lw ** 2]); S2 = np.linalg.inv(np.linalg.inv(Sb) + np.diag([1 / Rv ** 2, 1 / ell ** 2]))
    amp = c * np.sqrt(np.linalg.det(S2) / np.linalg.det(Sb)); k, q = k0 / sig, q0 / sig
    terms = [(0.0, np.zeros(2), Sb, np.zeros(2))]
    waves = [(a, 0.0) for a in (1, -1)] if q0 == 0 else [(a, b) for a in (1, -1) for b in (1, -1)]
    share = 0.5 if q0 == 0 else 0.25
    for a, b in waves: terms.append((np.log(amp * share), np.zeros(2), S2, np.array([a * k, b * q])))
    return terms


def _zms(m):
    """m - sin m for complex m, without cancellation (series for |m| < 0.5)."""
    m2 = m * m; ser = m * m2 / 6 * (1 - m2 / 20 * (1 - m2 / 42 * (1 - m2 / 72 * (1 - m2 / 110 * (1 - m2 / 156)))))
    return np.where(np.abs(m) < 0.5, ser, m - np.sin(m))


def point_fields(pre, sig, x):
    """exact p, s, H_vv and posterior quantities at points x (n x 2, (v, w) relative to the top), written in DIFFERENCES between terms:
    g_t = grad log N_t = g_ref + delta_t (ref = the first term), delta_t = -(T_t^-1 - T_ref^-1) x + (T_t^-1 mu_t - T_ref^-1 mu_ref) + i K_t kappa_t,
    (T_t^-1 - T_ref^-1) = T_t^-1 (S_ref - S_t) T_ref^-1 exactly.  Then s = g_ref + sum pi_t delta_t, the posterior covariance of the score uses
    delta_t - mean delta, and d_p D = sum pi_t [ f'_t d_p m'_z,t + f_t (delta_t,p - sum pi_u delta_u,p) ].  Terms sharing an envelope (a ripple
    filling the strip) then never form the huge common score -v/T_v ~ 1/sigma^2 x (strip width) inside a cancellation.
    'N'    = s_p (m_z - E[sin z|y])       (oscillating; only for checks)
    'Nibp' = -d_p D, D := E[z - sin z|y]  (cancellation-free; E[N] = E[Nibp] by integration by parts in the p-direction)."""
    s2 = sig * sig; n = len(x); ez = np.array([1.0, 1.0]) / SQ2; ep = np.array([1.0, -1.0]) / SQ2
    c0r, mur, Tir = pre[0][0], pre[0][1], pre[0][2]; dr = x - mur; gref = -(dr @ Tir.T)
    logs = []; dels = []; fts = []; dfs = []; fps = []; dms = []; ezs = []; eps = []; Tvv = []; Tww = []; Tvw = []; gts = []
    for c0, mu, Ti, K, varz, Kk, Ck, qd, Sref_, St_ in pre:
        d = x - mu; Tdr = d @ Ti.T
        logs.append(c0 - 0.5 * np.einsum('ni,ni->n', d, Tdr) + 1j * (d @ Kk) - 0.5 * qd)
        dT = Ti @ (Sref_ - St_) @ Tir                              # T_t^-1 - T_ref^-1, exact
        dels.append(-(x @ dT.T) + (Ti @ mu - Tir @ mur)[None, :] + 1j * Kk[None, :])
        mp = mu + d @ K.T + 1j * Ck[None, :]; mzt = (mp[:, 0] + mp[:, 1]) / SQ2; em1 = -np.expm1(-varz / 2); e = np.exp(-varz / 2)
        fts.append(_zms(mzt) + np.sin(mzt) * em1); fps.append(2 * np.sin(mzt / 2) ** 2 + np.cos(mzt) * em1); dms.append(-s2 * (ez @ Ti @ ep))      # = ez.K.ep exactly (ez.ep = 0); stage W fix: ez@K@ep with K ~ I is pure rounding (~2e-17) once sigma^2/S << 1e-16
        ezs.append(np.sin(mzt) * e); eps.append((mp[:, 0] - mp[:, 1]) / SQ2); Tvv.append(Ti[0, 0]); Tww.append(Ti[1, 1]); Tvw.append(Ti[0, 1])
        # stage W: F_t - F_ref without cancellation.  m'_t - m'_ref = s2 Re(delta_t) + i (C_t kappa_t - C_ref kappa_ref);  varz_t - varz_ref = -s2^2 ez.dT.ez
        if not dfs: m0, v0, Ck0 = mzt, varz, Ck
        dz = (s2 * ((-(x @ dT.T) + (Ti @ mu - Tir @ mur)[None, :]) @ ez) + 1j * ((Ck - Ck0) @ ez)) if dfs else np.zeros(n, complex)
        dvz = -s2 * (s2 * (ez @ dT @ ez)); mb = m0 + dz / 2; e1t = -np.expm1(-(v0 + dvz) / 2)          # dvz bracketed: sigma^4 underflows below sigma ~ 1e-77
        dfs.append(2 * dz * np.sin(mb / 2) ** 2 + 2 * np.cos(mb) * _zms(dz / 2)
                   + 2 * np.cos(mb) * np.sin(dz / 2) * e1t + np.sin(m0) * np.exp(-v0 / 2) * (-np.expm1(-dvz / 2)))
    L = np.stack(logs, 1); mx = L.real.max(1, keepdims=True); W_ = np.exp(L - mx); p_rel = W_.sum(1); pi = W_ / p_rel[:, None]
    p = (p_rel * np.exp(mx[:, 0])).real
    De = np.stack(dels, 1)                                          # n x T x 2
    dbar = np.einsum('nt,ntj->nj', pi, De); s = (gref + dbar).real
    Dc = De - dbar[:, None, :]
    Hvv = (np.einsum('nt,t->n', pi, np.array(Tvv)) - np.einsum('nt,nt->n', pi, Dc[:, :, 0] ** 2)).real
    F_ = np.stack(fts, 1); Fp = np.stack(fps, 1); Dm = np.array(dms)
    D = np.einsum('nt,nt->n', pi, F_).real
    dF = np.stack(dfs, 1)                                           # F_t - F_ref (sum_t pi_t Dc_t = 0, so F_ref drops out exactly)
    dD_p = (np.einsum('nt,nt,t->n', pi, Fp, Dm) + np.einsum('nt,nt,nt->n', pi, dF, Dc @ ep)).real
    Esz = np.einsum('nt,nt->n', pi, np.stack(ezs, 1)).real; Exp = np.einsum('nt,nt->n', pi, np.stack(eps, 1)).real
    sv, sw = s[:, 0], s[:, 1]; mz = ((x[:, 0] + s2 * sv) + (x[:, 1] + s2 * sw)) / SQ2; sp = (sv - sw) / SQ2; sq = (sv + sw) / SQ2
    u = x[:, 0] + x[:, 1] + s2 * (sv + sw); wf = np.clip(s2 * np.maximum(Hvv, 0), 0, 1)
    # stage W2 diagnostics: H_ww, H_vw, the posterior Var_z = sigma^2 (1 - sigma^2 H_zz), and the second-order form of N
    Hww = (np.einsum('nt,t->n', pi, np.array(Tww)) - np.einsum('nt,nt->n', pi, Dc[:, :, 1] ** 2)).real
    Hvw = (np.einsum('nt,t->n', pi, np.array(Tvw)) - np.einsum('nt,nt->n', pi, Dc[:, :, 0] * Dc[:, :, 1])).real
    Vz = s2 * (1 - s2 * (Hvv + 2 * Hvw + Hww) / 2); mz2 = mz * mz
    extra = {'aHvv': s2 * Hvv, 'aHww': s2 * Hww, 'mz2': mz2, 'Varz': Vz, 'Napx': s2 * (Hvv - Hww) * (mz2 + Vz) / 4,
             'Napx_vv': s2 * Hvv * (mz2 + Vz) / 4, 'Napx_ww': -s2 * Hww * (mz2 + Vz) / 4}
    # the AM-GM form (stage W2): phi = 1 - cos m_z;  N1 = E[sigma^2 H_pz phi] = (sigma^2/2) E[(H_vv - H_ww) phi]
    #   = (sigma^2/2) (E[s_v^2 phi] - E[s_w^2 phi]) + (sigma^2/2) E[s_v d_v phi - s_w d_w phi]   (integration by parts, exact)
    phi = 2 * np.sin(mz / 2) ** 2; smz = np.sin(mz)
    dvphi = smz * (1 - s2 * (Hvv + Hvw)) / SQ2; dwphi = smz * (1 - s2 * (Hww + Hvw)) / SQ2
    extra.update({'N1': s2 * (Hvv - Hww) * phi / 2, 'phi_svv': sv * sv * phi, 'phi_sww': sw * sw * phi, 'ibp_rest': sv * dvphi - sw * dwphi})
    # stage X1: B = (s2/2) E[s_v d_v phi - s_w d_w phi] = -s2^2 E[s_z H_pz sin m_z] (two more integrations by parts); its absolute-value bound
    Hpz = (Hvv - Hww) / 2; Bx = sq * Hpz * smz
    extra.update({'B_alt': Bx, 'B_abs': np.abs(Bx), 'sww_w': sw * sw * (1 - phi / 4)})
    return p, {**extra, 'svv': sv * sv, 'sww': sw * sw, 'N': sp * (mz - Esz), 'Nibp': -dD_p, 'j': wf, 'Z': wf * u * u, 'sEf': sq * Exp + sp * Esz, 'u2svv': u * u * sv * sv}


def _finish(S, sig, use='N'):
    if use not in S and 'Nibp' in S: use = 'Nibp'
    s2 = sig * sig; Jvv, Jww, N, j, Z = S['svv'], S['sww'], S[use], S['j'], S['Z']
    omr = (2 * s2 * Jww + N) / (s2 * (Jvv + Jww)); y = sig * Jww / j; P = Z * Jww / j ** 2; d2 = Z / (j * sig)
    Ps = (s2 * S['u2svv'] * Jww / (s2 * Jvv) ** 2) if 'u2svv' in S else float('nan')
    return {'gap_over_sigma': omr / sig, 'R': 1 - omr, 'P_sharp': Ps, 'J_vv': Jvv, 'J_ww': Jww, 'N': N, 'j': j, 'Z': Z, 'y_over_sigma': y, 'P': P, 'F': d2 / 8 + 2 * y, 'd': np.sqrt(d2)}


def grid_sums(terms, sig, vn, wn, rows=64, wv=None):
    pre = _pre(terms, sig); wv, ww = (trap_weights(vn) if wv is None else wv), trap_weights(wn); acc = {k: 0.0 for k in ('svv', 'sww', 'N', 'Nibp', 'j', 'Z', 'sEf', 'u2svv')}; mass = 0.0
    for a in range(0, len(vn), rows):
        V, W = np.meshgrid(vn[a:a + rows], wn, indexing='ij'); x = np.stack([V.ravel(), W.ravel()], 1); dA = np.outer(wv[a:a + rows], ww).ravel()
        p, q = point_fields(pre, sig, x); pw = p * dA; mass += pw.sum()
        for k in acc: acc[k] += float(np.sum(pw * q[k]))
    S = {k: v / mass for k, v in acc.items()}; out = _finish(S, sig); out['grid_mass'] = mass; out['n_points'] = len(vn) * len(wn)
    out['gap_over_sigma_ibp'] = _finish(S, sig, 'Nibp')['gap_over_sigma']; out['N_ibp'] = S['Nibp']
    out['gap_over_sigma_direct'] = (1 - S['sEf'] / (sig * sig * (S['svv'] + S['sww']))) / sig          # the direct form (needs a uniform grid)
    return out


def sample_rho(n, rng, Lv, Lw, c, k, Rv, ell, q=0.0):
    """x ~ rho = N(0, diag(Lv^2, Lw^2)) (1 + c exp(-v^2/2Rv^2 - w^2/2 ell^2) cos(k v) cos(q w)) exactly (rejection, acceptance >= 1/(1+c))."""
    out = []; got = 0
    while got < n:
        m = int(1.3 * (n - got) * (1 + c)) + 16
        v = rng.normal(0, Lv, m); w = rng.normal(0, Lw, m)
        g = 1 + c * np.exp(-v * v / (2 * Rv * Rv) - w * w / (2 * ell * ell)) * np.cos(k * v) * np.cos(q * w)
        keep = rng.random(m) * (1 + c) < g; out.append(np.stack([v[keep], w[keep]], 1)); got += int(keep.sum())
    return np.concatenate(out)[:n]


def mc_sums(terms, sig, par, n=4_000_000, batches=20, seed=0, chunk=200_000):
    """par: dict Lv, Lw, c, k0, Rv, ell, q0 (absolute lengths, as in u_cgm.ripple_terms)."""
    rng = np.random.default_rng(seed); pre = _pre(terms, sig); res = []
    k = par['k0'] / sig; q = par.get('q0', 0.0) / sig
    for b in range(batches):
        acc = {kk: 0.0 for kk in ('svv', 'sww', 'Nibp', 'j', 'Z')}; m = 0
        for a in range(0, n // batches, chunk):
            nn = min(chunk, n // batches - a)
            x = sample_rho(nn, rng, par['Lv'], par['Lw'], par['c'], k, par['Rv'], par['ell'], q) + sig * rng.normal(size=(nn, 2))
            _, qd = point_fields(pre, sig, x)
            for kk in acc: acc[kk] += float(np.sum(qd[kk]))
            m += nn
        res.append(_finish({kk: v / m for kk, v in acc.items()}, sig, 'Nibp'))
    keys = ('gap_over_sigma', 'P', 'F', 'y_over_sigma', 'd')
    tot = {kk: float(np.mean([r[kk] for r in res])) for kk in keys}
    tot.update({kk + '_se': float(np.std([r[kk] for r in res], ddof=1) / np.sqrt(batches)) for kk in keys}); tot['n_samples'] = n
    return tot


def gap_vlimit(k0, lam, M, sig, c=1.0, nv=64, nw_per_ell=40, Lv_sig=1.0e6, coarse=None, Rv_factor=10.0):
    """TRUE gap (and P, F, y/sigma) in the V-periodic limit of the ripple strip: window ell = lam sqrt(sigma), background Lw = M ell,
    strip width Lv = Lv_sig * sigma -> infinity in sigma units (background V-focus sigma^2/Lv^2 -> 0) while Lv/ell -> 0 is implied by
    evaluating ONE V-period around v = 0 (the v-extent of u and z drops out).  Periodic trapezoid in v, uniform grid in w."""
    from u_cgm import nodes
    s = np.sqrt(sig); ell = lam * s; Lw = M * ell; Lv = Lv_sig * sig; per = 2 * np.pi * sig / k0
    T = ripple_terms_k(sig, Lv, Lw, c, k0, Rv_factor * Lv, ell, 0.0)
    vn = (np.arange(nv) + 0.5) * per / nv - per / 2; wv = np.full(nv, per / nv)
    if coarse is None: wn = nodes(6.5 * Lw, ell / nw_per_ell, 6.5 * Lw, ell / nw_per_ell)
    else: wn = nodes(min(10 * ell, 6.5 * Lw), ell / nw_per_ell, 6.5 * Lw, max(Lw / coarse, ell / nw_per_ell))   # fine on the window, coarse on the background
    out = grid_sums(T, sig, vn, wn, rows=16, wv=wv); out['gap_over_sigma'] = out['gap_over_sigma_ibp']       # only the smooth form is valid on one period
    return out


def rho_density(x, Lv, Lw, c, k, Rv, ell, q=0.0):
    """rho(x) = N(0, diag(Lv^2, Lw^2)) (1 + c exp(-v^2/2Rv^2 - w^2/2ell^2) cos(k v) cos(q w)), x in (v, w)."""
    v, w = x[:, 0], x[:, 1]
    nb = np.exp(-v * v / (2 * Lv * Lv) - w * w / (2 * Lw * Lw)) / (2 * np.pi * Lv * Lw)
    return nb * (1 + c * np.exp(-v * v / (2 * Rv * Rv) - w * w / (2 * ell * ell)) * np.cos(k * v) * np.cos(q * w))


def mc_sums_is(terms, sig, par, n=8_000_000, batches=20, seed=0, chunk=200_000, beta=0.5):
    """Monte Carlo with importance sampling (no grid): x ~ q = (1 - beta) rho + beta g, g = the window Gaussian N(0, S2) of the ripple,
    weight rho(x)/q(x) (exact, unbiased); y = x + sigma Z.  Needed when the ripple window carries a tiny fraction (~1/M) of the mass.
    The gap uses the cancellation-free N = -E[d_p D] (as gap_vlimit)."""
    rng = np.random.default_rng(seed); pre = _pre(terms, sig); res = []
    k = par['k0'] / sig; q = par.get('q0', 0.0) / sig
    Sb = np.diag([par['Lv'] ** 2, par['Lw'] ** 2]); S2 = np.linalg.inv(np.linalg.inv(Sb) + np.diag([1 / par['Rv'] ** 2, 1 / par['ell'] ** 2]))
    L2 = np.linalg.cholesky(S2); det2 = np.linalg.det(S2)
    for b in range(batches):
        acc = {kk: 0.0 for kk in ('svv', 'sww', 'Nibp', 'j', 'Z', 'u2svv')}; wsum = 0.0
        for a in range(0, n // batches, chunk):
            nn = min(chunk, n // batches - a); nr = rng.binomial(nn, 1 - beta)
            x = np.concatenate([sample_rho(nr, rng, par['Lv'], par['Lw'], par['c'], k, par['Rv'], par['ell'], q), rng.normal(size=(nn - nr, 2)) @ L2.T])
            rho = rho_density(x, par['Lv'], par['Lw'], par['c'], k, par['Rv'], par['ell'], q)
            g = np.exp(-0.5 * np.einsum('ni,ij,nj->n', x, np.linalg.inv(S2), x)) / (2 * np.pi * np.sqrt(det2))
            wt = rho / ((1 - beta) * rho + beta * g)
            y = x + sig * rng.normal(size=(nn, 2)); _, qd = point_fields(pre, sig, y)
            for kk in acc: acc[kk] += float(np.sum(wt * qd[kk]))
            wsum += float(np.sum(wt))
        res.append(_finish({kk: v / wsum for kk, v in acc.items()}, sig, 'Nibp'))
    keys = ('gap_over_sigma', 'P', 'F', 'y_over_sigma', 'd', 'P_sharp')
    tot = {kk: float(np.mean([r[kk] for r in res])) for kk in keys}
    tot.update({kk + '_se': float(np.std([r[kk] for r in res], ddof=1) / np.sqrt(batches)) for kk in keys}); tot['n_samples'] = n
    return tot


def mc_delta_sums(sig, par, n=8_000_000, batches=20, seed=0, chunk=200_000, beta=0.5, bg_grid_n=(801, 801)):
    """Second, grid-free computation of the ripple's effect (for large background length M, where plain MC drowns in background noise):
       E_p[h] = E_{p_bg}[h_bg] + E_q[ (rho/q) h(y) - (rho_bg/q) h_bg(y) ],   same (x, Z) in both terms (common random numbers),
    q = (1 - beta) rho + beta g (g = window Gaussian), y = x + sigma Z.  h = the summands of the gap / P / F (cancellation-free N).
    E_{p_bg}[h_bg] for the pure Gaussian background on a separable trapezoid grid (smooth, no ripple).  Returns the combined sums."""
    rng = np.random.default_rng(seed); k = par['k0'] / sig; q = par.get('q0', 0.0) / sig
    full = _pre(ripple_terms_k(sig, par['Lv'], par['Lw'], par['c'], par['k0'], par['Rv'], par['ell'], par.get('q0', 0.0)), sig)
    bgt = [(0.0, np.zeros(2), np.diag([par['Lv'] ** 2, par['Lw'] ** 2]), np.zeros(2))]; bg = _pre(bgt, sig)
    from u_cgm import nodes
    Tv, Tw = np.sqrt(par['Lv'] ** 2 + sig ** 2), np.sqrt(par['Lw'] ** 2 + sig ** 2)
    vn = np.linspace(-9 * Tv, 9 * Tv, bg_grid_n[0]); wn = np.linspace(-9 * Tw, 9 * Tw, bg_grid_n[1])
    keys = ('svv', 'sww', 'Nibp', 'j', 'Z', 'u2svv')
    G = grid_sums(bgt, sig, vn, wn); Sbg = {'svv': G['J_vv'], 'sww': G['J_ww'], 'Nibp': G['N_ibp'], 'j': G['j'], 'Z': G['Z']}
    # u2svv of the background from the same grid
    V_, W_ = np.meshgrid(vn, wn, indexing='ij'); x = np.stack([V_.ravel(), W_.ravel()], 1); pb, qb = point_fields(bg, sig, x)
    from u_cgm import trap_weights
    dA = np.outer(trap_weights(vn), trap_weights(wn)).ravel(); pw = pb * dA; Sbg['u2svv'] = float(np.sum(pw * qb['u2svv']) / np.sum(pw))
    Sg = np.diag([par['Lv'] ** 2, par['Lw'] ** 2]); S2 = np.linalg.inv(np.linalg.inv(Sg) + np.diag([1 / par['Rv'] ** 2, 1 / par['ell'] ** 2]))
    L2 = np.linalg.cholesky(S2); S2i = np.linalg.inv(S2); det2 = np.linalg.det(S2); res = []
    for b in range(batches):
        acc = {kk: 0.0 for kk in keys}; m = 0
        for a in range(0, n // batches, chunk):
            nn = min(chunk, n // batches - a); nr = rng.binomial(nn, 1 - beta)
            xs = np.concatenate([sample_rho(nr, rng, par['Lv'], par['Lw'], par['c'], k, par['Rv'], par['ell'], q), rng.normal(size=(nn - nr, 2)) @ L2.T])
            rho = rho_density(xs, par['Lv'], par['Lw'], par['c'], k, par['Rv'], par['ell'], q); rb = rho_density(xs, par['Lv'], par['Lw'], 0.0, k, par['Rv'], par['ell'], q)
            g = np.exp(-0.5 * np.einsum('ni,ij,nj->n', xs, S2i, xs)) / (2 * np.pi * np.sqrt(det2)); qq = (1 - beta) * rho + beta * g
            y = xs + sig * rng.normal(size=(nn, 2)); _, hf = point_fields(full, sig, y); _, hb = point_fields(bg, sig, y)
            for kk in keys: acc[kk] += float(np.sum(rho / qq * hf[kk] - rb / qq * hb[kk]))
            m += nn
        S = {kk: Sbg[kk] + acc[kk] / m for kk in keys}; res.append(_finish(S, sig))
    ks = ('gap_over_sigma', 'P', 'F', 'y_over_sigma', 'P_sharp')
    tot = {kk: float(np.mean([r[kk] for r in res])) for kk in ks}
    tot.update({kk + '_se': float(np.std([r[kk] for r in res], ddof=1) / np.sqrt(batches)) for kk in ks}); tot['n_samples'] = n
    return tot


def gap_vlimit_split(k0, lam, M, sig, c=1.0, nv=64, nw_per_ell=40, Lv_sig=1.0e10, coarse=400, win=7.0):
    """gap_vlimit with every sum split into the ripple window (|w| < win*ell) and the background outside it.
    Returns the window / background parts of the numerator 2 sigma^2 J_ww + N and of the denominator sigma^2 (J_vv + J_ww) (both as
    contributions to the gap: part / (sigma * total denominator)), so that the gap = sum of the four contributions' numerators / denominator."""
    from u_cgm import nodes
    s = np.sqrt(sig); ell = lam * s; Lw = M * ell; Lv = Lv_sig * sig; per = 2 * np.pi * sig / k0
    T = ripple_terms_k(sig, Lv, Lw, c, k0, 1e30, ell, 0.0)
    vn = (np.arange(nv) + 0.5) * per / nv - per / 2; wv = np.full(nv, per / nv)
    wn = nodes(min(10 * ell, 6.5 * Lw), ell / nw_per_ell, 6.5 * Lw, max(Lw / coarse, ell / nw_per_ell))
    pre = _pre(T, sig); from u_cgm import trap_weights
    ww = trap_weights(wn); inside = np.abs(wn) < win * ell
    acc = {r: {k: 0.0 for k in ('svv', 'sww', 'Nibp', 'mass')} for r in ('win', 'bg')}
    for a in range(0, len(vn), 16):
        V, W = np.meshgrid(vn[a:a + 16], wn, indexing='ij'); x = np.stack([V.ravel(), W.ravel()], 1); dA = np.outer(wv[a:a + 16], ww).ravel()
        p, q = point_fields(pre, sig, x); pw = p * dA; ins = np.tile(inside, len(vn[a:a + 16]))
        for r, m in (('win', ins), ('bg', ~ins)):
            acc[r]['mass'] += float(pw[m].sum())
            for k in ('svv', 'sww', 'Nibp'): acc[r][k] += float(np.sum(pw[m] * q[k][m]))
    mt = acc['win']['mass'] + acc['bg']['mass']; s2 = sig * sig
    den = s2 * (acc['win']['svv'] + acc['bg']['svv'] + acc['win']['sww'] + acc['bg']['sww']) / mt
    out = {}
    for r in ('win', 'bg'):
        out['gapnum_' + r] = (2 * s2 * acc[r]['sww'] + acc[r]['Nibp']) / mt / (sig * den)
        out['mass_' + r] = acc[r]['mass'] / mt; out['den_share_' + r] = s2 * (acc[r]['svv'] + acc[r]['sww']) / mt / den
        out['gap_from_N_' + r] = acc[r]['Nibp'] / mt / (sig * den); out['gap_from_Jww_' + r] = 2 * s2 * acc[r]['sww'] / mt / (sig * den)
    out['gap_over_sigma'] = out['gapnum_win'] + out['gapnum_bg']
    return out
