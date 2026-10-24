"""Stage X2: the smoothing-uncertainty lemma, 1-d (on a line).
Lemma X2 (proved, L^2 form).  For any real f in L^2(R) and g = f * phi_sigma (Gaussian blur, variance sigma^2), let
    F := sigma^2 ||g'||^2  (the focus carried by f after the blur),   xbar := sigma^2 ||g'||^2 / ||g||^2  (= 1/(4 P_loc): squared mean wavenumber after blur).
Then  F <= xbar e^{-xbar} ||f||^2.
Proof: Parseval with x = sigma^2 xi^2 and the probability measure nu ~ |f^(xi)|^2 dxi: ||g||^2 = A ||f||^2, A = int e^{-x} dnu,
F = ||f||^2 int x e^{-x} dnu.  With mu = e^{-x} nu / A, 1/A = E_mu[e^x] >= e^{E_mu x} = e^{xbar} (Jensen), so A <= e^{-xbar} and F = xbar A ||f||^2.  QED
(equality iff |f^| is concentrated on one |xi|: a single wavenumber.)
Numerics here: (1) random 1-d structures (sums of Gaussian-windowed cosines with random wavenumbers, chirps, steps), FFT on a fine grid: the ratio
F / (xbar e^{-xbar} ||f||^2) must be <= 1 (= 1 for a pure wavenumber);  (2) the transfer to the strip window (weighted, global quantities): the
measured a_win/mu_win against xbar e^{-xbar} with xbar = 1/(4 P#_win) (results/w2_states.json, x1 states).  -> results/x2_lemma.json.  Python 3.11."""
import json
import numpy as np

rng = np.random.default_rng(2026)


def lemma_ratio(f, h, sig):
    n = len(f); xi = 2 * np.pi * np.fft.fftfreq(n, d=h); F_ = np.fft.fft(f) * h
    G = F_ * np.exp(-0.5 * (sig * xi) ** 2); nf = np.sum(np.abs(F_) ** 2); ng = np.sum(np.abs(G) ** 2); Fd = sig ** 2 * np.sum(xi ** 2 * np.abs(G) ** 2)
    xbar = Fd / ng; return Fd / (xbar * np.exp(-xbar) * nf), xbar, Fd / nf


out = {'random': [], 'pure': []}
sig = 1.0; L = 400.0; n = 2 ** 16; h = L / n; v = (np.arange(n) - n / 2) * h
for k0 in (0.5, 1.0, 2.0, 3.0, 5.0):
    f = np.exp(-v ** 2 / (2 * 40.0 ** 2)) * np.cos(k0 * v / sig); r, xb, Fn = lemma_ratio(f, h, sig); out['pure'].append({'k0': k0, 'ratio': r, 'xbar': xb})
for t in range(400):
    kind = t % 4; f = np.zeros(n)
    if kind == 0:      # several windowed cosines
        for _ in range(rng.integers(1, 6)):
            f += rng.normal() * np.exp(-(v - rng.uniform(-80, 80)) ** 2 / (2 * rng.uniform(2, 40) ** 2)) * np.cos(rng.uniform(0, 6) * v / sig + rng.uniform(0, 6.3))
    elif kind == 1:    # chirp
        a0, a1 = rng.uniform(0, 4), rng.uniform(-0.03, 0.03); f = np.exp(-v ** 2 / (2 * 50.0 ** 2)) * np.cos((a0 + a1 * v) * v / sig)
    elif kind == 2:    # steps and kinks (broad spectrum)
        f = np.sign(np.sin(rng.uniform(0.05, 2) * v)) * np.exp(-v ** 2 / (2 * 60.0 ** 2))
    else:              # white noise in a window, band-limited above K
        K = rng.uniform(0.5, 4); z = np.fft.fft(rng.normal(size=n) * np.exp(-v ** 2 / (2 * 50.0 ** 2))); xi = 2 * np.pi * np.fft.fftfreq(n, d=h)
        z[np.abs(sig * xi) < K] = 0; f = np.real(np.fft.ifft(z))
    r, xb, Fn = lemma_ratio(f, h, sig); out['random'].append({'kind': int(kind), 'ratio': float(r), 'xbar': float(xb), 'F_over_norm': float(Fn)})
out['max_ratio_random'] = max(r['ratio'] for r in out['random']); out['pure_ratio_dev'] = max(abs(r['ratio'] - 1) for r in out['pure'])
# transfer to the strip windows (global, Fisher-weighted quantities of stage W2)
W2 = json.load(open('results/w2_states.json')); tr = []
for k, s in W2.items():
    if not k.startswith('strip') or not np.isfinite(s.get('sqrtPsharp_win', np.nan)): continue
    xb = 1 / (4 * s['sqrtPsharp_win'] ** 2); tr.append({'state': k, 'a_win_per_mass': s['a_win_per_mass'], 'xbar': xb, 'bound_shape': xb * np.exp(-xb),
                                                        'C_needed': s['a_win_per_mass'] / (xb * np.exp(-xb))})
out['transfer_strips'] = tr
json.dump(out, open('results/x2_lemma.json', 'w'), indent=1, default=float)
print('max ratio (random, must be <= 1): %.6f;  pure-wavenumber ratio - 1: %.1e' % (out['max_ratio_random'], out['pure_ratio_dev']))
for t in tr: print(t)
