"""Stage Q, task Q2: the uncertainty relation behind (B), proved in unweighted form, and the one factor that remains.

Notation (pendulum, V/W frame, stage J/M): p = rho * N(0, sigma^2 I), s = grad log p, H = -grad^2 log p, C(y) = Cov(X | Y = y),
m(y) = E[X | y] = y + sigma^2 s,  grad m = C/sigma^2 = I - sigma^2 H  (so H <= I/sigma^2 because C >= 0),  J = E[H] = E[s s^T].
w = sigma^2 (H_vv)_+ in [0, 1],  u = sqrt2 (m_q - pi),  j = E[w],  Z = E[w u^2],  P = Z J_ww / j^2  (= d^2 y/sigma).

LEMMA (proved here; Stein/Cramer-Rao for the posterior mean).  For unit vectors e, f and any constant c:
    E[(e.m - c)^2] * J_ff  >=  ( E[e^T C f]/sigma^2 )^2  =  ( e.f - sigma^2 e^T J f )^2 .
  Proof: E[(e.m - c) f.s] = -E[f . grad(e.m)] = -E[e^T C f]/sigma^2 (integration by parts, p s = grad p, p decays);
  Cauchy-Schwarz with E[(f.s)^2] = J_ff;  E[C] = sigma^2 I - sigma^4 J.   With e = q-direction (V + W)/sqrt2, f = W, c = pi:
    P_unw := E[u^2] J_ww  >=  ( 1 - sigma^2 (J_ww + J_vw) )^2                        (U)
  i.e. the UNWEIGHTED focus spread times the W-information is bounded below by a number that tends to 1 when the W-information is
  small on the scale 1/sigma^2.  It fails only when sigma^2 (J_ww + J_vw) -> 1, i.e. when the state is as sharp along W as the blur
  allows -- exactly the N2 counterexample (y/sigma ~ 100 at sigma = 1e-2, sigma^2 J_ww ~ j).
  In the region y/sigma <= Y0:  sigma^2 J_ww = sigma j (y/sigma) <= sigma Y0, and |sigma^2 J_vw| <= sqrt(sigma^2 J_vv) sqrt(sigma^2 J_ww)
  <= sqrt(sigma Y0)  (H <= I/sigma^2 gives sigma^2 J_vv <= 1 only if H_vv >= 0 on average... used numerically, reported).
LINE-AVERAGED VERSION (proved): with omega(v) = E[w | V = v] (constant along W-lines, E[omega] = j):
    E[omega u^2] J_ww  >=  E[omega u^2] E[omega s_w^2]  >=  ( E[omega (c_vw + c_ww)] )^2,   c = C/sigma^2 = I - sigma^2 H,
  so  P_omega := E[omega u^2] J_ww / j^2 >= B_omega := (E[omega (c_vw + c_ww)]/j)^2  (no slope term),  and  P = theta P_omega with
  theta = E[w u^2]/E[omega u^2]: (B) follows from theta >= theta_0 and B_omega >= b_0.
IDENTITY (exact):  P = lambda_w * P_unw,  lambda_w := E[w u^2] / ( j^2 E[u^2] )   ("does the focus weight see the far points?").
  Hence  P >= lambda_w (1 - sigma^2 (J_ww + J_vw))^2  -- rigorous; (B) follows from a lower bound lambda_w >= lambda_0 in the region
  y/sigma <= Y0.  If w were uncorrelated with u^2, lambda_w = 1/j >= 1.  lambda_w is the remaining (unproved) factor.
For every state: P, P_unw, the lemma bound (U) and the plain W-version Var(m_w) J_ww >= (1 - sigma^2 J_ww)^2, lambda_w, j,
sigma^2 J_ww, sigma^2 J_vw, kappa_corr = E[w u^2]/(E[w] E[u^2]) (= j lambda_w).   -> results/q2_uncert.json"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from i_tools import grid, V, W


def quantities(ws, mus, Ss, sig, per_sd=6, max_points=2_500_000):
    Y, dA, n = grid(ws, mus, Ss, sig, per_sd=per_sd, max_points=int(min(max_points, 2.4e7 / len(ws)))); s2 = sig * sig
    T = [S + s2 * np.eye(2) for S in Ss]; Ti = [np.linalg.inv(t) for t in T]
    lg = [np.log(w) - 0.5 * np.log(np.linalg.det(t)) - np.log(2 * np.pi) for w, t in zip(ws, T)]
    logc = np.stack([l - 0.5 * np.einsum('ni,ij,nj->n', Y - m, ti, Y - m) for l, m, ti in zip(lg, mus, Ti)], 1)
    mx = logc.max(1); e = np.exp(logc - mx[:, None]); p = e.sum(1) * np.exp(mx); pi = e / e.sum(1, keepdims=True); del e, logc
    uk = np.stack([-(Y - m) @ ti for m, ti in zip(mus, Ti)], 1); s = np.einsum('nk,nkd->nd', pi, uk)
    H = np.einsum('nk,kij->nij', pi, np.array(Ti)) - (np.einsum('nk,nki,nkj->nij', pi, uk, uk) - np.einsum('ni,nj->nij', s, s)); del uk
    pw = p * dA; pw = pw / pw.sum(); m = Y + s2 * s
    R = np.stack([V, W], 1); Hf = np.einsum('ia,nij,jb->nab', R, H, R); sv, sw = s @ V, s @ W
    w = np.clip(s2 * np.maximum(Hf[:, 0, 0], 0), 0, 1); u = np.sqrt(2) * (m[:, 0] - np.pi); mw = m @ W
    E = lambda a: float(np.sum(pw * a))
    j = E(w); Jww = E(sw * sw); Jvw = E(sv * sw); Jww_H = E(Hf[:, 1, 1]); Eu2 = E(u * u); Z = E(w * u * u)
    var_mw = E(mw * mw) - E(mw) ** 2
    # line-averaged focus weight omega(v) = E[w | V = v] (p-weighted average along each W-line; grid axis 1 = W): d_w omega = 0, so the
    # weighted Stein identity has no slope term:  E[omega u^2] J_ww >= E[omega u^2] E[omega s_w^2] >= ( E[omega (c_vw + c_ww)] )^2
    P2 = pw.reshape(n); W2 = w.reshape(n); om = (np.sum(P2 * W2, 1) / np.maximum(np.sum(P2, 1), 1e-300))[:, None] * np.ones(n[1]); om = om.ravel()
    c_vw = -s2 * Hf[:, 0, 1]; c_ww = 1 - s2 * Hf[:, 1, 1]          # C/sigma^2 = I - sigma^2 H in the V/W frame
    Zom = E(om * u * u); Bom = E(om * (c_vw + c_ww)) ** 2; Eoms = E(om * sw * sw)
    return {'sigma': sig, 'j': j, 'Z': Z, 'Eu2': Eu2, 'Jww': Jww, 'Jww_from_H': Jww_H, 'Jvw': Jvw, 's2Jww': s2 * Jww, 's2Jvw': s2 * Jvw,
            'P': Z * Jww / j ** 2, 'P_unw': Eu2 * Jww, 'bound_U': (1 - s2 * (Jww + Jvw)) ** 2, 'varmw_Jww': var_mw * Jww, 'bound_W': (1 - s2 * Jww) ** 2,
            'lambda_w': Z / (j * j * Eu2), 'kappa_corr': Z / (j * Eu2),
            'Z_omega': Zom, 'lemma_omega_lhs': Zom * Eoms, 'lemma_omega_rhs': Bom, 'P_omega': Zom * Jww / j ** 2, 'B_omega_over_j2': Bom / j ** 2, 'theta': Z / Zom,
            'E_omega_check': E(om) / j, 'y_over_sigma': sig * Jww / j, 'd': float(np.sqrt(Z / (j * sig))), 'n_grid': len(Y)}


def run(st):
    tag, ws, mus, Ss, sig = st
    try: q = quantities(ws, mus, Ss, sig)
    except (MemoryError, ValueError, np.linalg.LinAlgError) as e: return {'tag': tag, 'sigma': sig, 'error': str(e)}
    q['tag'] = tag; return q


def all_states():
    from o2_branch import all_states as A
    from h2_adversarial import unpack
    S = A()
    for fn in ['p3_rhomax', 'q2_fmin', 'q2_theta']:
        if os.path.exists(f'results/{fn}.json'):
            for o in json.load(open(f'results/{fn}.json')):
                if o.get('feasible_found', True) and 'x' in o:
                    ws, mus, Ss = unpack(np.array(o['x']), o['K'], o['sigma']); S.append((f"{fn}_K{o['K']}_s{o['sigma']:g}_{o.get('D', '')}_{o.get('Y', '')}", ws, mus, Ss, o['sigma']))
    return S


if __name__ == '__main__':
    S = all_states(); print(len(S), 'states', flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as pool: out = pool.map(run, S, chunksize=4)
    json.dump(out, open('results/q2_uncert.json', 'w'), indent=1, default=float)
    ok = [o for o in out if 'error' not in o]
    print('errors', len(out) - len(ok))
    print('lemma U violations', sum(o['P_unw'] < o['bound_U'] * (1 - 1e-9) for o in ok), ' W-version violations', sum(o['varmw_Jww'] < o['bound_W'] * (1 - 1e-9) for o in ok))
    print('omega lemma violations', sum(o['lemma_omega_lhs'] < o['lemma_omega_rhs'] * (1 - 1e-9) for o in ok), ' E[omega]/j max dev', max(abs(o['E_omega_check'] - 1) for o in ok))
    print('identity P = lambda P_unw max rel', max(abs(o['P'] - o['lambda_w'] * o['P_unw']) / o['P'] for o in ok))
    sm = [o for o in ok if o['y_over_sigma'] <= 0.5]
    print('y/sigma <= 0.5:', len(sm), ' min P', min(o['P'] for o in sm), ' min P_unw', min(o['P_unw'] for o in sm), ' min bound_U', min(o['bound_U'] for o in sm),
          ' min lambda_w', min(o['lambda_w'] for o in sm), ' min theta', min(o['theta'] for o in sm), ' min B_omega/j2', min(o['B_omega_over_j2'] for o in sm),
          ' min P_omega', min(o['P_omega'] for o in sm), ' max s2Jww', max(o['s2Jww'] for o in sm), ' max |s2Jvw|', max(abs(o['s2Jvw']) for o in sm))
