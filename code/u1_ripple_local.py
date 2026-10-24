"""Stage U1: lemma (p~) on LOCALISED ripple states (the scale of all adversarial states: V within ~60 sigma, W within ~8 sqrt(sigma)).
rho = N(0, diag(Lv^2, Lw^2)) [1 + c exp(-v^2/(2 Rv^2) - w^2/(2 ell^2)) cos(k0 v/sigma) cos(q0 w/sigma)]  at the top (u_cgm.ripple_terms).
Maximise a I(p~)/J_ww over (k0, c, Lv, Lw, Rv, ell, q0) in the box  Lv <= 60 sigma, Lw <= 8 sqrt(sigma), Rv <= Lv, ell <= Lw,  with y/sigma <= Y0
held exactly (only feasible evaluations count), grid-mass guard |mass - 1| <= 1e-4.  Exact fields (u_cgm.evaluate), true gap recorded.
Optimised values carry 'opt' (loose).  Checkpointed.  usage: python3 u1_ripple_local.py <sigma> <Y0> [iters] [seed]   Python 3.11 compatible."""
import json, os, sys
import numpy as np
from scipy.optimize import minimize
from u_cgm import evaluate, ripple_terms, nodes
from t_tools import t_terms

CK = 'results/u1_ripple_local_ckpt'; os.makedirs(CK, exist_ok=True)


def unpackz(z, sig):
    s = np.sqrt(sig); sg = lambda t: 1 / (1 + np.exp(-t))
    k0 = 0.3 + 3.2 * sg(z[0]); c = sg(z[1]); Lv = sig * (2 + 58 * sg(z[2])); Lw = s * (0.3 + 7.7 * sg(z[3]))
    Rv = Lv * (0.05 + 0.95 * sg(z[4])); ell = Lw * (0.05 + 0.95 * sg(z[5])); q0 = 2.0 * np.sqrt(sig) * sg(z[6]) if len(z) > 6 else 0.0
    return dict(k0=k0, c=c, Lv=Lv, Lw=Lw, Rv=Rv, ell=ell, q0=q0)


def run(par, sig, fine=1.0):
    T = ripple_terms(sig, par['Lv'], par['Lw'], par['c'], par['k0'], par['Rv'], par['ell'], par['q0'])
    hv = min(0.25 * sig, 2 * np.pi * sig / par['k0'] / 16) / fine
    hw = min(par['ell'] / 40, (2 * np.pi * sig / par['q0']) / 24 if par['q0'] > 0 else 1e9, 0.5 * np.sqrt(sig)) / fine
    hv = max(hv, 0.3 * sig / fine) if par['k0'] < 2.5 else hv
    vn = nodes(6.5 * par['Lv'], hv, 6.5 * par['Lv'], hv); wn = nodes(6.5 * par['Lw'], hw, 6.5 * par['Lw'], hw)
    while len(vn) * len(wn) > 1.2e6 * fine ** 2:                                     # <= ~1.5 GB
        hw *= 1.15; wn = nodes(6.5 * par['Lw'], hw, 6.5 * par['Lw'], hw)
    f, dg = evaluate(T, sig, vn, wn); t = t_terms(f)
    dg.update({k: float(t[k]) for k in ('aIpt_over_Jww', 'Lambda', 'a', 'P_sharp', 'bound_ptilde', 'x_minus_1', 'K_V', 'sd_ptilde_mu_over_sigma')})
    return dg


def job(sig, Y0, iters=300, seed=0):
    key = 's%g_Y%g_seed%d' % (sig, Y0, seed); ck = os.path.join(CK, key + '.json')
    st = json.load(open(ck)) if os.path.exists(ck) else {'n': 0, 'best': None, 'z_last': None}
    rng = np.random.default_rng(1000 + seed)
    def obj(z):
        par = unpackz(z, sig)
        try: dg = run(par, sig)
        except (MemoryError, ValueError, np.linalg.LinAlgError): return 1e3
        st['n'] += 1; st['z_last'] = list(map(float, z))
        if st['n'] % 10 == 0: json.dump(st, open(ck, 'w'), default=float)
        if not np.isfinite(dg['aIpt_over_Jww']) or abs(dg['grid_mass'] - 1) > 1e-4: return 1e3
        if dg['y_over_sigma'] > Y0: return 1e2 + dg['y_over_sigma']
        if st['best'] is None or dg['aIpt_over_Jww'] > st['best']['aIpt_over_Jww']:
            st['best'] = dict(dg, par=par, z=list(map(float, z)))
        return -dg['aIpt_over_Jww']
    lg = lambda t: float(np.log(t / (1 - t)))
    manual = np.array([lg((1.0 - 0.3) / 3.2), lg(0.9), lg((58 - 2) / 58 * 0.98), lg((7.5 - 0.3) / 7.7), lg(0.9), lg(0.9), lg(0.5)])   # k0 1, c .9, Lv ~57, Lw 7.5, Rv/Lv, ell/Lw .9, q0 = sqrt(sigma)
    z0 = np.array(st['best']['z']) if st['best'] else manual + (rng.normal(0, 0.3, 7) if seed > 0 else 0.0)
    minimize(obj, z0, method='Nelder-Mead', options={'maxfev': iters, 'xatol': 1e-3, 'fatol': 1e-4, 'initial_simplex': z0 + np.vstack([np.zeros(7), np.eye(7) * 1.0])})
    json.dump(st, open(ck, 'w'), default=float)
    b = st['best']
    if b:
        fine = run(b['par'], sig, fine=1.5)
        b = dict(b); b['opt_aIpt_over_Jww'] = b.pop('aIpt_over_Jww'); b['aIpt_over_Jww_finegrid'] = fine['aIpt_over_Jww']
        b['y_over_sigma_finegrid'] = fine['y_over_sigma']; b['gap_finegrid'] = fine['gap_over_sigma']; b['feasible_on_fine_grid'] = bool(fine['y_over_sigma'] <= Y0)
        b['par_units'] = {'k0': b['par']['k0'], 'c': b['par']['c'], 'Lv_over_sigma': b['par']['Lv'] / sig, 'Lw_over_sqrt_sigma': b['par']['Lw'] / np.sqrt(sig),
                          'Rv_over_sigma': b['par']['Rv'] / sig, 'ell_over_sqrt_sigma': b['par']['ell'] / np.sqrt(sig), 'q0_over_sqrt_sigma': b['par']['q0'] / np.sqrt(sig)}
    out = 'results/u1_ripple_local_%s.json' % key
    json.dump({'sigma': sig, 'Y0': Y0, 'seed': seed, 'n_evals': st['n'], 'best': b}, open(out, 'w'), indent=1, default=float)
    print(key, json.dumps({k: b[k] for k in ('opt_aIpt_over_Jww', 'aIpt_over_Jww_finegrid', 'y_over_sigma', 'gap_over_sigma', 'gap_finegrid', 'P_sharp', 'bound_ptilde', 'par_units')} if b else None, default=float), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]; job(float(a[0]), float(a[1]), int(a[2]) if len(a) > 2 else 300, int(a[3]) if len(a) > 3 else 0)
