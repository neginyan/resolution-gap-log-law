"""Stage Y3: assumption X-A2 (local P vs global P) replaced by proved steps, with numerical checks.
Proved chain (stage-Y report):
  (i)   P_w >= (2/pi^2) P#  when |m_z| <= pi                     (1 - cos m >= 2 m^2/pi^2 on [-pi, pi], and 1 - phi/4 >= 1/2)
  (ii)  P# >= x^2 J_ww/(a I(p~)) >= x^2/(1 + Dbar)               (stage T chain + stage U: a I(p~) <= (1 + Dbar) J_ww)
  (iii) on every line: D = Var(phi|line)/omega <= 8 xl + 2 sigma^4 E[s^4|line]/omega,  xl := sigma^4 E[(p''/p)^2|line]/omega,
        phi = sigma^2 (s^2 - 2 p''/p) (the line field of stage U), sigma^4 E[s^4] <= 3 (Jensen: sigma s = -E[Z|y])
  (iv)  Lemma X2' (second spectral moment): sigma^2 ||g'||^2 <= x2 e^{-x2} ||f||^2,  x2 := sigma^2 ||g''||^2 / ||g'||^2 (Jensen with the convex e^x/x)
So the local quantity that controls the global P is the line dispersion D, and D is controlled by the second spectral moment of the line,
whose focus Lemma X2' suppresses exponentially; the remaining assumption is the flat envelope (X-A1), which identifies xl with x2.
Numerics: (a) (ii) on the stored/adversarial states (t1_chain + u1_chain joined by key); (b) (iii) and (iv) on random 1-d blurred densities.
-> results/y3_a2.json.  Python 3.11."""
import json
import numpy as np

out = {}
T = {o['key']: o for o in json.load(open('results/t1_chain.json')) if 'error' not in o}
U = {o['key']: o for o in json.load(open('results/u1_chain.json')) if 'error' not in o}
rows = []
for k in sorted(set(T) & set(U)):
    t, u = T[k], U[k]; x = 1 + t['x_minus_1']; b = x * x / (1 + u['Dbar'])
    rows.append({'key': k, 'P_sharp': t['P_sharp'], 'x': x, 'Dbar': u['Dbar'], 'bound_x2_1pD': b, 'ratio': b / t['P_sharp'], 'y_over_sigma': t['y_over_sigma']})
out['chain_ii'] = {'n': len(rows), 'max_ratio': max(r['ratio'] for r in rows), 'n_violations': sum(r['ratio'] > 1 + 1e-9 for r in rows),
                   'region_min_bound': min(r['bound_x2_1pD'] for r in rows if r['y_over_sigma'] <= 0.5), 'region_max_Dbar': max(r['Dbar'] for r in rows if r['y_over_sigma'] <= 0.5)}
# (b) 1-d: blurred densities p = rho * phi_sigma (sigma = 1) on a periodic grid
rng = np.random.default_rng(11); n = 2 ** 14; Lbox = 300.0; h = Lbox / n; v = (np.arange(n) - n / 2) * h; xi = 2 * np.pi * np.fft.fftfreq(n, d=h)
res = []; lem = []
for t in range(300):
    kind = t % 3
    env = np.exp(-v ** 2 / (2 * rng.uniform(10, 40) ** 2))
    if kind == 0: mod = 1 + rng.uniform(0.05, 1.0) * np.cos(rng.uniform(0.3, 4) * v + rng.uniform(0, 6.3)) * np.exp(-v ** 2 / (2 * rng.uniform(3, 30) ** 2))
    elif kind == 1:
        mod = np.ones(n)
        for _ in range(rng.integers(1, 4)): mod += rng.uniform(0, 0.3) * np.cos(rng.uniform(0.3, 4) * v + rng.uniform(0, 6.3))
        mod = np.maximum(mod, 0.0)
    else:
        c = rng.uniform(-20, 20, rng.integers(2, 7)); wd = rng.uniform(0.2, 5, len(c)); mod = sum(np.exp(-(v - ci) ** 2 / (2 * wi ** 2)) for ci, wi in zip(c, wd)) + 1e-3
    rho = env * mod; rho /= rho.sum() * h
    ph = np.fft.fft(rho); p = np.real(np.fft.ifft(ph * np.exp(-0.5 * xi ** 2))); p1 = np.real(np.fft.ifft(1j * xi * ph * np.exp(-0.5 * xi ** 2))); p2 = np.real(np.fft.ifft(-xi ** 2 * ph * np.exp(-0.5 * xi ** 2)))
    m = p > 1e-12 * p.max(); wq = p[m] * h; wq /= wq.sum(); s = (p1 / p)[m]; q = (p2 / p)[m]
    phi = s * s - 2 * q; om = np.sum(wq * phi); varphi = np.sum(wq * (phi - om) ** 2)
    D = varphi / om; xl = np.sum(wq * q * q) / om; s4 = np.sum(wq * s ** 4)
    res.append({'kind': kind, 'omega': float(om), 'ibp_check': float(abs(np.sum(wq * s * s) / om - 1)), 'D': float(D), 'xl': float(xl), 's4_term': float(2 * s4 / om), 'sigma4_Es4': float(s4),
                'bound_iii': float(8 * xl + 2 * s4 / om), 'ratio_iii': float(D / (8 * xl + 2 * s4 / om))})
    f = rho - np.real(np.fft.ifft(ph * (np.abs(xi) < 0.1)))            # the structure (envelope removed by a low-pass), any real f works for (iv)
    F_ = np.fft.fft(f) * h; G = F_ * np.exp(-0.5 * xi ** 2); n0 = np.sum(np.abs(F_) ** 2); n1 = np.sum(xi ** 2 * np.abs(G) ** 2); n2 = np.sum(xi ** 4 * np.abs(G) ** 2)
    x2 = n2 / n1; lem.append({'x2': float(x2), 'ratio_iv': float(n1 / (x2 * np.exp(-x2) * n0))})
out['line_iii'] = {'n': len(res), 'max_ratio': max(r['ratio_iii'] for r in res), 'max_sigma4_Es4': max(r['sigma4_Es4'] for r in res), 'max_ibp_check': max(r['ibp_check'] for r in res),
                   'max_D_over_1p8xl': max(r['D'] / (1 + 8 * r['xl']) for r in res), 'rows': res}
out['lemma_iv'] = {'n': len(lem), 'max_ratio': max(r['ratio_iv'] for r in lem), 'rows': lem}
json.dump(out, open('results/y3_a2.json', 'w'), indent=1)
print('(ii) P# >= x^2/(1+Dbar): %d states, max ratio %.4f, violations %d; region: min bound %.3f, max Dbar %.2f' % (len(rows), out['chain_ii']['max_ratio'], out['chain_ii']['n_violations'], out['chain_ii']['region_min_bound'], out['chain_ii']['region_max_Dbar']))
print('(iii) D <= 8 xl + 2 s^4/omega on %d 1-d densities: max ratio %.4f; max sigma^4 E[s^4] %.3f (<= 3); ibp check %.1e; max D/(1+8 xl) %.3f' % (len(res), out['line_iii']['max_ratio'], out['line_iii']['max_sigma4_Es4'], out['line_iii']['max_ibp_check'], out['line_iii']['max_D_over_1p8xl']))
print('(iv) Lemma X2\': max ratio %.4f (<= 1)' % out['lemma_iv']['max_ratio'])
