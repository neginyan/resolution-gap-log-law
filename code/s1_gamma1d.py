"""Stage S1, the 1-d 'line shape' constant.
For a 1-d measure mu, Y = X + Z (X ~ mu, Z ~ N(0, 1): sigma = 1 without loss of generality, the quantities are scale free), define
    phi(y) := 2 - 2 E[Z^2 | y] + E[Z | y]^2     ( = sigma^2 (2 H - s^2) ),     omega := E[phi] = E[ E[Z|y]^2 ] = sigma^2 I   (in (0, 1]).
The chain of s_tools needs the dispersion  D(mu) := Var(phi) / E[phi]  ( = omega * Gamma ) to be bounded on every line.
Gaussian mu = N(m, t): D = 2 omega <= 2.  Here: adversarial MAXIMISATION of D over K-component Gaussian mixtures (point masses as the
width -> 0 limit), K = 2, 3, 4, 8; random restarts + adaptive hill climbing; exact posterior moments on a fine y grid.
Usage: python s1_gamma1d.py <steps> <restarts> <K list>   -> results/s1_gamma1d.json"""
import numpy as np, json, os, sys


def unpack(x, K):
    w = np.exp(x[:K] - x[:K].max()); w /= w.sum(); m = x[K:2 * K]; t = np.exp(x[2 * K:3 * K]); return w, m, t


def disp(w, m, t, ny=40001):
    lo, hi = m.min() - 9 * np.sqrt(t.max() + 1), m.max() + 9 * np.sqrt(t.max() + 1); y = np.linspace(lo, hi, ny); dy = y[1] - y[0]
    v = t + 1; lc = np.log(w)[None] - 0.5 * np.log(2 * np.pi * v)[None] - 0.5 * (y[:, None] - m[None]) ** 2 / v[None]
    mx = lc.max(1); e = np.exp(lc - mx[:, None]); p = e.sum(1) * np.exp(mx); pi = e / e.sum(1, keepdims=True)
    ez_k = (y[:, None] - m[None]) / v[None]; ez = (pi * ez_k).sum(1); ez2 = (pi * (ez_k ** 2 + (t / v)[None])).sum(1)
    pw = p * dy; mass = pw.sum(); pw /= mass
    phi = 2 - 2 * ez2 + ez ** 2; om = float(np.sum(pw * ez * ez)); Ephi = float(np.sum(pw * phi)); Vphi = float(np.sum(pw * phi ** 2) - Ephi ** 2)
    return {'omega': om, 'E_phi': Ephi, 'Var_phi': Vphi, 'D': Vphi / Ephi, 'Gamma': Vphi / Ephi ** 2, 'mass': float(mass), 'identity': (Ephi - om) / om}


def obj(x, K):
    if np.any(np.abs(x[:K]) > 12) or np.any(x[2 * K:] < -12) or np.any(x[2 * K:] > 8) or np.ptp(x[K:2 * K]) > 60: return -1.0, None
    d = disp(*unpack(x, K), ny=8001)
    return (d['D'] if np.isfinite(d['D']) else -1.0), d


def search(K, steps, restarts, seed):
    rng = np.random.default_rng(seed); best = None; allr = []
    for rs in range(restarts):
        x = np.concatenate([rng.normal(0, 1, K), rng.normal(0, 3, K), rng.uniform(-8, 3, K)]); fx, _ = obj(x, K); step = 0.5; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.4); fy, _ = obj(y, K)
            if fy > fx: x, fx = y, fy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 2.0)); acc = 0
        allr.append(fx)
        if best is None or fx > best[0]: best = (fx, x.copy())
    d = disp(*unpack(best[1], K), ny=40001); d2 = disp(*unpack(best[1], K), ny=160001)
    w, m, t = unpack(best[1], K)
    return {'K': K, 'opt_D': d['D'], 'opt_D_fine': d2['D'], 'opt_Gamma': d['Gamma'], 'omega': d['omega'], 'identity': d['identity'], 'restarts': allr,
            'w': w.tolist(), 'm': (m - m.mean()).tolist(), 't': t.tolist(), 'x': best[1].tolist()}


if __name__ == '__main__':
    steps, restarts = int(sys.argv[1]), int(sys.argv[2]); Ks = [int(v) for v in sys.argv[3].split(',')]
    fn = 'results/s1_gamma1d.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    # reference: single Gaussians (D = 2 omega), two point masses at distance L
    ref = {'gauss': [disp(np.array([1.0]), np.array([0.0]), np.array([tt])) for tt in [1e-4, 0.1, 1, 10, 100]],
           'two_points': [dict(disp(np.array([0.5, 0.5]), np.array([-L / 2, L / 2]), np.array([1e-5, 1e-5])), L=L) for L in [1, 2, 3, 4, 5, 6, 8, 12]]}
    for K in Ks:
        if any(o.get('K') == K for o in out): continue
        o = search(K, steps, restarts, 500 + K); out.append(o); print(K, o['opt_D'], o['opt_D_fine'], o['omega'], flush=True)
        json.dump(out, open(fn, 'w'), indent=1, default=float)
    json.dump(out, open(fn, 'w'), indent=1, default=float); json.dump(ref, open('results/s1_gamma1d_ref.json', 'w'), indent=1, default=float)


def periodic(Dl, lam, ny=4001, nmax=40):
    """infinite comb of point masses (spacing Dl, in units of sigma) of relative height lam on a flat background (the
    infinitely-wide-background limit, computed exactly over one period).  Fourier argument: as lam -> 0 or for dominant combs,
    D -> 4 k0^2, k0 = 2 pi / Dl, so D has no universal bound (a 'grating' finer than ~ 2.5 sigma)."""
    y = np.linspace(0, Dl, ny, endpoint=False); n = np.arange(-nmax, nmax + 1)
    z = y[:, None] - n[None] * Dl; g = lam * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    p = 1 + g.sum(1); pi = g / p[:, None]; pb = 1 / p
    ez = (pi * z).sum(1); ez2 = (pi * z * z).sum(1) + pb; phi = 2 - 2 * ez2 + ez ** 2; w = p / p.sum()
    om = float((w * ez * ez).sum()); Ef = float((w * phi).sum()); V = float((w * phi * phi).sum() - Ef ** 2)
    return {'spacing_over_sigma': Dl, 'lam': lam, 'D': V / Ef, 'omega': om, 'E_phi': Ef, 'four_k0_sq': 4 * (2 * np.pi / Dl) ** 2}


def comb_table():
    out = []
    for Dl in [6, 5, 4, 3, 2.5, 2, 1.5]:
        best = max((periodic(Dl, l) for l in np.exp(np.linspace(-6, 6, 49))), key=lambda r: r['D']); out.append(best)
    return out
