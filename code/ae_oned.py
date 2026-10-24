"""Stage AE: one-dimensional checks of the valley lemma and of Theorem AE3, and 1-d adversarial searches.
State: K point masses (positions x_i, weights w_i) blurred with variance t; exact posterior on a fine y-grid (independent of the 2-d codes).
  c(y) = Var(X|y), s = E[X|y] - y (t = 1 units), H = (1 - c)/t, J = E[s^2] = E[H].
Checks: Lemma AE1 pointwise (second difference of ln p vs ln cosh(u sqrt(2c)) - u^2, several u), the inequality (*) pointwise,
N_eta := E[(c/t - 1)_+^3 1{c/t >= 1 + eta}] <= kappa_eta t J, and the 1-d theorem t^2 R' <= eta tR + kappa_eta.
Searches (Nelder-Mead): maximise N_eta/(tJ), kappa_H = -t^2 E[H^3]/J and t^2 R' for K = 2, 4, 8, 16, 32.
-> results/ae_oned.json (restartable).  Python 3.11."""
import os, json, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def fields(x, w, t, y):
    """posterior quantities at the points y (1-d)."""
    L = np.log(w)[None, :] - (y[:, None] - x[None, :]) ** 2 / (2 * t); lz = logsumexp(L, axis=1); P = np.exp(L - lz[:, None])
    m = P @ x; c = P @ (x * x) - m * m; c = np.maximum(c, 0.0); lp = lz - 0.5 * np.log(2 * np.pi * t)
    return lp, (m - y) / t, c


def sums(x, w, t, eta=0.1, h=None):
    x = np.asarray(x, float); w = np.asarray(w, float) / np.sum(w); sd = np.sqrt(t)
    h = 0.02 * sd if h is None else h; y = np.arange(x.min() - 9 * sd, x.max() + 9 * sd, h)
    lp, s, c = fields(x, w, t, y); p = np.exp(lp); p = p / (p.sum() * h); W = p * h
    H = (1 - c / t) / t; J = np.sum(W * s * s); EH = np.sum(W * H); E2 = np.sum(W * H * H); E3 = np.sum(W * H ** 3)
    g = np.clip(c / t - 1, 0, None); N = np.sum(W * g ** 3 * (c / t >= 1 + eta)); Nsh = np.sum(W * g ** 3 * (c / t < 1 + eta))
    return {'J': J, 'EH': EH, 'R': E2 / J, 'tR': t * E2 / J, 'kappa_H': -t * t * E3 / J, 'N_eta_over_tJ': N / (t * J), 'shallow_over_eta_t2EH2': Nsh / (eta * t * t * E2) if E2 > 0 else 0.0,
            'mass': float(np.sum(W)), 'y': y, 'lp': lp, 's': s, 'c': c, 'W': W}


def t2Rp(x, w, t, eta=0.1, hh=2e-3):
    a, b, m = sums(x, w, t * (1 - hh), eta), sums(x, w, t * (1 + hh), eta), sums(x, w, t, eta)
    return t * t * (b['R'] - a['R']) / (2 * t * hh), m


def lemma_checks(x, w, t, us=(0.2, 0.5, 1.0, 2.0, 4.0)):
    """Lemma AE1 and (*) at every grid point: min over points of (lhs - rhs) (>= 0 expected)."""
    x = np.asarray(x, float); w = np.asarray(w, float) / np.sum(w); sd = np.sqrt(t); h = 0.01 * sd
    y = np.arange(x.min() - 7 * sd, x.max() + 7 * sd, h); lp, s, c = fields(x, w, t, y); worst1 = np.inf; worst2 = np.inf
    p = np.exp(lp); cs = np.concatenate([[0], np.cumsum(p * s * s) * h])           # cumulative integral of p s^2
    for u0 in us:
        u = u0 * sd; k = int(round(u / h)); ip = np.arange(k, len(y) - k)
        if len(ip) == 0: continue
        lhs = lp[ip + k] + lp[ip - k] - 2 * lp[ip]; arg = (k * h) * np.sqrt(2 * c[ip] / t) / sd
        rhs = np.where(arg < 300, np.log(np.cosh(np.minimum(arg, 300))), arg - np.log(2)) - (k * h) ** 2 / t
        worst1 = min(worst1, float(np.min(lhs - rhs)))
        D = rhs; ok = D > 1e-3
        if ok.any():
            seg = cs[ip + k + 1] - cs[ip - k]                                          # int_{y0-u}^{y0+u} p s^2 (Riemann)
            bound = (k * h) / (4 * np.expm1(np.minimum(D, 1400) / 4) ** 2) * seg * t   # t: p s^2 has units 1/t
            r = p[ip][ok] / np.maximum(bound[ok], 1e-300); worst2 = min(worst2, float(np.min(1 / r)) if r.size else np.inf)
    return worst1, worst2


def unpack(z, K):
    x = np.cumsum(np.concatenate([[0.0], np.exp(z[:K - 1])])); w = np.exp(z[K - 1:2 * K - 1] - np.max(z[K - 1:2 * K - 1])); return x, w


if __name__ == '__main__':
    fn = 'results/ae_oned.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    C = json.load(open('results/ae_const.json')); kap = C['kappa_eta']['0.1']['kappa']
    rng = np.random.default_rng(3)
    if 'lemmas' not in out:
        L1, L2 = [], []
        for i in range(60):
            K = int(rng.integers(2, 12)); x = np.sort(rng.normal(0, rng.uniform(1, 6), K)); w = rng.dirichlet(np.ones(K) * rng.uniform(0.2, 2))
            a, b = lemma_checks(x, w, 1.0); L1.append(a); L2.append(b)
        out['lemmas'] = {'n': 60, 'lemma1_min_margin': min(L1), 'star_min_ratio': min(L2)}
        json.dump(out, open(fn, 'w'), indent=1); print(out['lemmas'], flush=True)
    for what in ('N_eta_over_tJ', 'kappa_H', 't2Rp'):
        for K in (2, 4, 8, 16, 32):
            key = '%s_K%d' % (what, K)
            if key in out: continue
            t0 = time.time(); best = None
            for st in range(3 if K <= 8 else 2):
                z0 = np.concatenate([np.log(rng.uniform(0.5, 4.5, K - 1)), rng.normal(0, 1, K)])
                def f(z):
                    x, w = unpack(z, K)
                    if x[-1] > 400: return 1e3
                    try:
                        if what == 't2Rp': v, _ = t2Rp(x, w, 1.0); return -v
                        return -sums(x, w, 1.0)[what]
                    except Exception: return 1e3
                r = minimize(f, z0, method='Nelder-Mead', options={'maxfev': 600 if K <= 8 else 400, 'xatol': 1e-6, 'fatol': 1e-9})
                if best is None or r.fun < best.fun: best = r
            x, w = unpack(best.x, K); S = sums(x, w, 1.0); tr, _ = t2Rp(x, w, 1.0)
            out[key] = {'value': float(-best.fun), 'x': list(map(float, x)), 'w': list(map(float, w / w.sum())), 'tR': S['tR'], 'kappa_H': S['kappa_H'], 'N_eta_over_tJ': S['N_eta_over_tJ'],
                        't2Rp': tr, 'theorem_1d_slack': 0.1 * S['tR'] + kap - tr, 'seconds': time.time() - t0}
            json.dump(out, open(fn, 'w'), indent=1); print(key, {k: (round(v, 5) if isinstance(v, float) else '') for k, v in out[key].items() if not isinstance(v, list)}, flush=True)
    # periodic-like 1-d lattice (many equal points) for several spacings
    if 'lattice_1d' not in out:
        rows = []
        for D in np.linspace(1.0, 8.0, 29):
            x = D * np.arange(81.0); w = np.ones(81); tr, S = t2Rp(x, w, 1.0); rows.append({'D': float(D), 't2Rp': tr, 'kappa_H': S['kappa_H'], 'N_eta_over_tJ': S['N_eta_over_tJ'], 'tR': S['tR']})
        out['lattice_1d'] = rows; json.dump(out, open(fn, 'w'), indent=1); print('1-d lattice max t2Rp %.4f, max kappa_H %.4f, max N/tJ %.4f' % (max(r['t2Rp'] for r in rows), max(r['kappa_H'] for r in rows), max(r['N_eta_over_tJ'] for r in rows)))
