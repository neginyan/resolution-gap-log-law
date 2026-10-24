"""Stage AA1: the K2-free bound of xbar_w (aa_tools) on every state of stage Z (x1 33, stored 377, strips 8), t = 1/2 and t = t*(sigma).
Also the new final bound of Theorem Z' (K_J, K_rho removed) with K_Lambda and P_y from results/z_chain_*.json.
-> results/aa_chain.json.  Python 3.11."""
import json, sys
import numpy as np
from aa_tools import pair_l2, strip_l2, xbar_chain, best_t
from z_chain import qp_terms, ripple_local_terms
from z_strip import strip_params

UNIF = lambda s: 1 / (1 + np.sqrt(2) / np.pi * np.sqrt(12 * (np.pi ** 2 + s * s)))   # AA3: eps_m-free factor


def terms_x1(key, sig):
    from h2_adversarial import unpack
    from q2_uncert import all_states
    from u2_chain import two_clusters
    if key.startswith('mix_'):
        o = json.load(open('results/' + key[4:])); return qp_terms(*unpack(np.array(o['x']), o['K'], sig))
    if key.startswith('stored'):
        tag, ws, mus, Ss, sg = all_states()[int(key[6:])]; return qp_terms(ws, mus, Ss)
    if key.startswith('twoclusters'):
        return qp_terms(*two_clusters(float(key.split('_V')[1].split('_')[0]), sig))
    if key.startswith('ripple_'):
        o = json.load(open('results/' + key[7:])); rr = [q for q in json.load(open('results/u3_ripple.json'))['ripple_local'] if q['file'] == key[7:]][0]
        par = dict(o['best']['par']); return ripple_local_terms(sig, par['Lv'], par['Lw'], par['c'], par['k0'], par['Rv'], par['ell'], rr['par_units']['q0_over_sqrt_sigma'] * np.sqrt(sig))
    return None


def finish(r, z, sig, t):
    """Theorem Z' bound:  gap/sigma >= (a/den)(1 - eps) kappa sqrt2/pi sqrt(P_y),  P_y >= 1/Lambda >= 1/(4 K_Lambda X_t)."""
    out = dict(r)
    out['S_lambda'] = [4 * z['K_Lambda'] * r['X_t_state'], z['Lambda']]                       # Lambda = 4 K_Lambda xbar_w <= 4 K_Lambda X_t
    kap_state = max(1 - z['eps_m'], UNIF(sig)); out['kappa_state'] = kap_state; out['kappa_uniform'] = UNIF(sig)
    sq = np.sqrt(2) / np.pi / np.sqrt(4 * z['K_Lambda'] * r['X_t_state'])
    out['sqrtPw_bound_state'] = kap_state * sq; out['sqrtPw_bound_uniform_kappa'] = UNIF(sig) * np.sqrt(2) / np.pi / np.sqrt(4 * z['K_Lambda'] * r['X_t_uniform'])
    out['sqrtPw'] = float(np.sqrt(z['P_w'])); out['K_Lambda'] = z['K_Lambda']; out['y_over_sigma'] = z['y_over_sigma']
    # AA3: Minkowski remainder r = eps_m sqrt(P_y) <= sqrt(12 V_z) Y,  Y = sigma J_ww / a  (moments E[zeta^4] <= 3 and Cramer-Rao sigma/a <= 2 V_z Y)
    Y = sig * z['J_ww'] / z['a']; out['Y'] = Y; out['AA3_r'] = [np.sqrt(12 * z['V_z']) * Y, z['eps_m'] * np.sqrt(z['P_y'])]
    out['AA3_r_uniform'] = [np.sqrt(12 * (np.pi ** 2 + sig * sig)) * Y, z['eps_m'] * np.sqrt(z['P_y'])]
    if 'gap_over_sigma' in z:
        e = z['eps'] if z.get('eps') is not None else 0.0; f = z['den_factor'] * (1 - max(e, 0.0))
        out['gap_over_sigma'] = z['gap_over_sigma']; out['bound_state'] = f * out['sqrtPw_bound_state']; out['bound_uniform'] = f * out['sqrtPw_bound_uniform_kappa']
        out['eps_known'] = z.get('eps') is not None
        G = z['gap_over_sigma'] / z['den_factor']; out['AA3_GY'] = [G, (1 - max(e, 0.0)) * Y]            # G >= (1 - eps) Y  (Theorem W' before AM-GM)
    return out


if __name__ == '__main__':
    Z = {'x1': json.load(open('results/z_chain_x1.json')), 'stored': json.load(open('results/z_chain_stored.json')), 'strips': json.load(open('results/z_chain_strips.json'))}
    out = {}
    for key, z in Z['x1'].items():
        sig = z['sigma']; T = terms_x1(key, sig)
        if T is None: continue                                                          # strips: done below
        for t in (0.5, best_t(sig)[0]):
            r = xbar_chain(lambda bv, bw, w: pair_l2(T, np.diag([bv, bw]), w), sig, t); out['x1:%s:t%.3f' % (key, t)] = finish(r, z, sig, t)
    from jobs_stored_aa import stored_terms
    for key, T, sig in stored_terms():
        z = Z['stored'][key]
        for t in (0.5, best_t(sig)[0]):
            r = xbar_chain(lambda bv, bw, w: pair_l2(T, np.diag([bv, bw]), w), sig, t); out['st:%s:t%.3f' % (key, t)] = finish(r, z, sig, t)
    for key, z in Z['strips'].items():
        sig = z['sigma']; P = strip_params(sig, z['k0'], z['lam'], z['M'], z['eta'], z['c'])
        for t in (0.5, best_t(sig)[0]):
            r = xbar_chain(lambda bv, bw, w: strip_l2(P, bv, bw, w), sig, t); out['sp:%s:t%.3f' % (key, t)] = finish(r, z, sig, t)
    json.dump(out, open('results/aa_chain.json', 'w'), indent=1, default=float)
    print(len(out), 'rows')
