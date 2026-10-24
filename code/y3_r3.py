"""Stage Y3: a state-quantity bound for the posterior remainder R3 of Theorem W', checked on Gaussian-mixture states (stored and adversarial
mixtures, two clusters).  Written directly from the mixture formulas (numpy only; the posterior of a Gaussian mixture is a Gaussian mixture).
Bound (proof in the stage-Y report):  with g(z) = z - sin z, Taylor to third order with Lagrange remainder (|g'''| <= 1), Cauchy-Schwarz and
Tweedie (third posterior cumulant kappa_pzz = -sigma^6 d_p H_zz):
    |R3| <= (sigma^4/2) E[ |sin m_z| |d_p H_zz| ]  +  (sigma^2/6) E[ s_hat_p M6_hat ],
    s_hat_p = sd(x_p | y)/sigma,  M6_hat = sqrt(E[(x_z - m_z)^6 | y]) / sigma^3   (= sqrt(15) for a Gaussian posterior of variance sigma^2).
In units of the AM-GM term:  eps_R3 <= bound / (sigma a sqrt(P_w)).  -> results/y3_r3.json.  Python 3.11."""
import json, os, glob
import numpy as np

SQ2 = np.sqrt(2.0); EZ = np.array([1.0, 1.0]) / SQ2; EP = np.array([1.0, -1.0]) / SQ2
RM = np.array([[1, 1], [1, -1]]) / SQ2; TOP = np.array([np.pi, 0.0])


def to_vw(ws, mus, Ss):
    return [(w, RM @ (np.asarray(m) - TOP), RM @ np.asarray(S) @ RM.T) for w, m, S in zip(ws, mus, Ss)]


def fields(comps, sig, y):
    s2 = sig * sig; n = len(y); logs = []; pre = []
    for w, mu, S in comps:
        T = S + s2 * np.eye(2); Ti = np.linalg.inv(T); d = y - mu
        logs.append(np.log(w) - np.log(2 * np.pi * np.sqrt(np.linalg.det(T))) - 0.5 * np.einsum('ni,ij,nj->n', d, Ti, d)); pre.append((mu, Ti, S @ Ti, S - S @ Ti @ S, d))
    Lg = np.stack(logs, 1); mx = Lg.max(1, keepdims=True); W = np.exp(Lg - mx); pi = W / W.sum(1, keepdims=True)
    g = np.stack([-(d @ Ti.T) for mu, Ti, K, C, d in pre], 1)                         # grad log N_t
    s = np.einsum('nt,ntj->nj', pi, g)
    Hm = sum(pi[:, t, None, None] * (pre[t][1][None] - np.einsum('ni,nj->nij', g[:, t], g[:, t])) for t in range(len(pre))) + np.einsum('ni,nj->nij', s, s)
    mt = np.stack([mu + d @ K.T for mu, Ti, K, C, d in pre], 1); m = np.einsum('nt,ntj->nj', pi, mt)
    mz = m @ EZ; mp = m @ EP
    dz = mt @ EZ - mz[:, None]; dp = mt @ EP - mp[:, None]
    cz = np.array([EZ @ C @ EZ for mu, Ti, K, C, d in pre]); cp = np.array([EP @ C @ EP for mu, Ti, K, C, d in pre])
    E6 = np.einsum('nt,nt->n', pi, dz ** 6 + 15 * dz ** 4 * cz + 45 * dz ** 2 * cz ** 2 + 15 * cz ** 3)
    Vp = np.einsum('nt,nt->n', pi, cp[None] + dp ** 2)
    Hzz = np.einsum('i,nij,j->n', EZ, Hm, EZ)
    return pi, s, Hm, mz, Hzz, np.sqrt(Vp), np.sqrt(E6)


def bound_terms(comps, sig, ng=500):
    ev = max(abs(mu[0]) + 6 * np.sqrt(S[0, 0] + sig ** 2) for w, mu, S in comps); ew = max(abs(mu[1]) + 6 * np.sqrt(S[1, 1] + sig ** 2) for w, mu, S in comps)
    vn = np.linspace(-ev, ev, ng); wn = np.linspace(-ew, ew, ng); V, W = np.meshgrid(vn, wn, indexing='ij'); y = np.stack([V.ravel(), W.ravel()], 1)
    acc = np.zeros(4); h = 1e-3 * sig
    for a in range(0, len(y), 50000):
        yy = y[a:a + 50000]
        pi, s, Hm, mz, Hzz, sdp, sq6 = fields(comps, sig, yy)
        _, _, _, _, Hp, _, _ = fields(comps, sig, yy + h * EP); _, _, _, _, Hm_, _, _ = fields(comps, sig, yy - h * EP); dpH = (Hp - Hm_) / (2 * h)
        p = np.zeros(len(yy))
        for w, mu, S in comps:
            T = S + sig * sig * np.eye(2); d = yy - mu; p += w * np.exp(-0.5 * np.einsum('ni,ij,nj->n', d, np.linalg.inv(T), d)) / (2 * np.pi * np.sqrt(np.linalg.det(T)))
        acc += [np.sum(p), np.sum(p * np.abs(np.sin(mz)) * np.abs(dpH)), np.sum(p * (sdp / sig) * (sq6 / sig ** 3)), np.sum(p * s[:, 0] ** 2)]
    E = acc[1:] / acc[0]
    return {'E_sin_dpHzz': E[0], 'E_shat_M6': E[1], 'a_check': sig * sig * E[2], 'term1': sig ** 4 / 2 * E[0], 'term2': sig ** 2 / 6 * E[1]}


if __name__ == '__main__':
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from h2_adversarial import unpack
    from u2_chain import two_clusters
    X1 = json.load(open('results/x1_states.json')); out = {}
    jobs = []
    for fn in ['results/t4_K32_s0.01.json'] + sorted(glob.glob('results/s1_psharp_K*_s*.json')):
        o = json.load(open(fn)); jobs.append(('mix_' + os.path.basename(fn), to_vw(*unpack(np.array(o['x']), o['K'], o['sigma'])), o['sigma']))
    for sg in (0.01, 0.001):
        for V0 in (2, 20, 60): jobs.append(('twoclusters_V%g_s%g' % (V0, sg), to_vw(*two_clusters(V0, sg)), sg))
    for key, comps, sg in jobs:
        if key not in X1: continue
        b = bound_terms(comps, sg); x = X1[key]; sa = sg * x['a']
        R3 = x['R3_over_sa'] * sa; bnd = b['term1'] + b['term2']
        r = {'sigma': sg, **b, 'R3': R3, 'bound': bnd, 'holds': bool(abs(R3) <= bnd), 'ratio_R3_to_bound': abs(R3) / bnd,
             'eps_R3_actual': abs(R3) / (sa * x['sqrtP_w']), 'eps_R3_bound': bnd / (sa * x['sqrtP_w']), 'a_rel_check': abs(b['a_check'] / x['a'] - 1),
             'sigma_over_a': sg / x['a']}
        out[key] = r; print('%-36s |R3| %.3e  bound %.3e (%.2f)  eps_R3 %.2e <= %.2e  M6 %.2f  a-check %.1e' % (key, abs(R3), bnd, r['ratio_R3_to_bound'], r['eps_R3_actual'], r['eps_R3_bound'], b['E_shat_M6'], r['a_rel_check']), flush=True)
    json.dump(out, open('results/y3_r3.json', 'w'), indent=1)
