"""Independent checks of paper E, written separately from the study code in code/ (no imports from it, no stored results).

  1. Gap of the strip optima (Table II): the closed-form two-scale evaluator strip_gap.py reproduces g_min(sigma)
     at sigma = 1e-4 ... 1e-96 from the optimal parameters alone.
  2. Numerical result 3: the strips have no deep valleys. Largest eigenvalue of -tH on a 2-d Fourier grid of the
     sigma = 1e-2 strip optimum at t = sigma^2/4, sigma^2/2, sigma^2.
  3. Lemma 5 (smoothing-uncertainty in 1-d) on random structures.
  4. Lemma 11 (a valley is a dip) on random one-dimensional Gaussian mixtures.
Exit code 0 iff every check passes.  Run time: about one minute on one core.
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strip_gap as S

ok = True
def report(name, passed, detail):
    global ok
    ok &= bool(passed)
    print(('[PASS] ' if passed else '[FAIL] ') + name + '  ' + detail)

# --- 1. strip optima: (sigma, k0, lambda, M, c, L_v, published g_min) -----------------------------------------
STRIPS = [
    (1e-4, 1.0776040913961458, 2.568927048360967, 5.843292237991376, 0.9999962543713626, 0.009826802272113236, 1.149),
    (1e-6, 1.2274723008157367, 2.1260787682706574, 19.00331868912701, 0.9999999936597486, 0.0007078256023545683, 0.740),
    (1e-12, 1.8479474435694963, 1.543676118113122, 554.9685553693851, 0.9999999970672002, 3.1779586644318827e-07, 0.335),
    (1e-24, 2.9999410048171837, 1.1723430808691142, 488764.42415849684, 0.9999999929468937, 1.4216579634831477e-13, 0.179),
    (1e-48, 4.663473586141622, 0.9316671817787303, 436034261864.8868, 0.9999999999999998, 7.145893889696438e-26, 0.110),
    (1e-96, 6.960118489440912, 0.7604602230229209, 3.883961199338749e+23, 0.9997217372476832, 3.815003067880804e-50, 0.073),
]
for sig, k0, lam, M, c, Lv, gpub in STRIPS:
    g = S.evaluate(sig, k0, lam, M, c, Lv)['gap_over_sigma']
    report('strip g_min at sigma = %g' % sig, abs(g - gpub) <= 0.0006, 'independent %.6f, Table II %.3f' % (g, gpub))

# --- 2. strips have no deep valleys (sigma = 1e-2 optimum) ------------------------------------------------------
sig, k0, lam, M, c, eta = 0.01, 0.9835318720940052, 3.501498554511305, 1.528504639428711, 0.9999999999896707, 0.32429422466066626
ell = lam * math.sqrt(sig); Lw = M * ell; Lv = eta * ell; kap = k0 / sig
hv, hw = sig / 12, sig / 2
v = np.arange(-7 * Lv, 7 * Lv, hv); w = np.arange(-7 * Lw, 7 * Lw, hw); V, W = np.meshgrid(v, w, indexing='ij')
rho = np.exp(-V ** 2 / (2 * Lv ** 2) - W ** 2 / (2 * Lw ** 2)) * (1 + c * np.exp(-W ** 2 / (2 * ell ** 2)) * np.cos(kap * V))
kv = 2 * np.pi * np.fft.fftfreq(len(v), hv); kw = 2 * np.pi * np.fft.fftfreq(len(w), hw); KV, KW = np.meshgrid(kv, kw, indexing='ij')
Rh = np.fft.fft2(rho); del rho
lam_max = {}
for m in (0.25, 0.5, 1.0):
    t = m * sig ** 2; F = Rh * np.exp(-t * (KV ** 2 + KW ** 2) / 2)
    d = lambda mult: np.real(np.fft.ifft2(F * mult))
    p = d(1.0); pv, pw = d(1j * KV), d(1j * KW); pvv, pww, pvw = d(-KV ** 2), d(-KW ** 2), d(-KV * KW)
    mask = p > 1e-8 * p.max(); sv, sw = pv / p, pw / p
    Hvv = -(pvv / p - sv ** 2); Hww = -(pww / p - sw ** 2); Hvw = -(pvw / p - sv * sw)
    tr = Hvv + Hww; det = Hvv * Hww - Hvw ** 2; lmin = tr / 2 - np.sqrt(np.maximum((tr / 2) ** 2 - det, 0))
    lam_max[m] = float((-t * lmin)[mask].max())
ref = {0.25: 1.8745, 0.5: 1.7564, 1.0: 1.5370}
for m in ref:
    report('largest eigenvalue of -tH, sigma = 1e-2, t = %.2f sigma^2' % m, abs(lam_max[m] - ref[m]) <= 0.005,
           '%.4f (paper %.4f)' % (lam_max[m], ref[m]))
report('no deep valleys: all values below 1.88', max(lam_max.values()) < 1.88, 'max %.4f' % max(lam_max.values()))

# --- 3. Lemma 5 on random structures -----------------------------------------------------------------------------
rng = np.random.default_rng(5)
n = 4096; L = 200.0; x = np.linspace(-L / 2, L / 2, n, endpoint=False); xi = 2 * np.pi * np.fft.fftfreq(n, x[1] - x[0])
worst = 0.0
for _ in range(200):
    f = np.zeros(n)
    for _ in range(rng.integers(1, 6)):
        f += rng.normal() * np.exp(-(x - rng.uniform(-20, 20)) ** 2 / (2 * rng.uniform(0.05, 3) ** 2)) * np.cos(rng.uniform(0, 8) * x + rng.uniform(0, 6.3))
    sigma = rng.uniform(0.1, 2.0)
    fh = np.fft.fft(f); uh = fh * np.exp(-sigma ** 2 * xi ** 2 / 2)
    nf = np.sum(np.abs(fh) ** 2); nu = np.sum(np.abs(uh) ** 2); nd = np.sum(xi ** 2 * np.abs(uh) ** 2)
    xb = sigma ** 2 * nd / nu
    worst = max(worst, sigma ** 2 * nd / (xb * np.exp(-xb) * nf))
report('Lemma 5 on 200 random structures', worst <= 1 + 1e-9, 'largest ratio %.6f (must be <= 1)' % worst)

# --- 4. Lemma 11 on random 1-d Gaussian mixtures -------------------------------------------------------------
y = np.linspace(-30, 30, 24001); dy = y[1] - y[0]
def mix(xs, ws, vs):
    lg = np.array([math.log(a) - 0.5 * math.log(2 * math.pi * (1 + s)) - (y - m) ** 2 / (2 * (1 + s)) for m, a, s in zip(xs, ws, vs)])
    Mx = lg.max(0); e = np.exp(lg - Mx); lp = Mx + np.log(e.sum(0)); r = e / e.sum(0)
    mu = np.array([(m + s * y) / (1 + s) for m, s in zip(xs, vs)]); var = np.array([s / (1 + s) + 0 * y for s in vs])
    pm = (r * mu).sum(0); cvar = (r * (var + mu ** 2)).sum(0) - pm ** 2
    return lp, cvar
worst = 1e9
for _ in range(150):
    K = int(rng.integers(2, 8)); xs = rng.normal(0, rng.uniform(0.5, 5), K); ws = rng.uniform(0.05, 1, K)
    vs = np.exp(rng.uniform(-6, 1, K)) * (rng.random(K) < 0.6)
    lp, cv = mix(xs, ws, vs)
    for k in (50, 200, 800):
        u = k * dy; i = slice(k, len(y) - k)
        lhs = lp[2 * k:] + lp[:-2 * k] - 2 * lp[i]; rhs = np.log(np.cosh(u * np.sqrt(2 * np.maximum(cv[i], 0)))) - u * u
        msk = lp[i] > -60; worst = min(worst, float((lhs - rhs)[msk].min()))
report('Lemma 11 on 150 random 1-d mixtures', worst >= -1e-10, 'smallest margin %.2e (must be >= 0 up to rounding)' % worst)

print('ALL PASSED' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
