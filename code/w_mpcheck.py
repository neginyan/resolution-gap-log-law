"""Stage W: an independent high-precision evaluation (Python decimal, 320 digits; no mpmath in this workspace) of the point fields, used to check
the cancellation-free implementation in v3_tools.point_fields.  Everything is computed the naive way, with no differences and no rewriting:
  g_t(y) = e^{lc} / (2 pi sqrt det T) exp(-d'T^-1 d/2 + i (K kappa).d - kappa'C kappa/2),  d = y - mu,  K = S T^-1,  C = sigma^2 (I - sigma^2 T^-1)
  p = Re sum g_t,  D = Re sum g_t F_t / sum g_t,  F_t = m_z - sin(m_z) e^{-Var_z/2},  m = mu + K d + i C kappa  (complex posterior mean of a term)
  s = grad p / p and d_p D by central differences (step 1e-25 sigma; truncation ~1e-50 relative).
Python 3.11."""
from decimal import Decimal as Dd, getcontext, localcontext
getcontext().prec = 320; getcontext().Emin = -999999999999999999; getcontext().Emax = 999999999999999999
_PI = None


def pi():
    global _PI
    if _PI is None:
        with localcontext() as c:
            c.prec += 10; a, b, t, p = Dd(1), Dd(1) / Dd(2).sqrt(), Dd(1) / 4, Dd(1)
            for _ in range(12): an = (a + b) / 2; b = (a * b).sqrt(); t -= p * (a - an) ** 2; a = an; p *= 2
            _PI = (a + b) ** 2 / (4 * t)
    return +_PI


def dexp(x):
    if x < -10 ** 6: return Dd(0)
    return x.exp()


def _sincos(x):
    tp = 2 * pi(); x = x - tp * (x / tp).to_integral_value()
    with localcontext() as c:
        c.prec += 10; s = Dd(0); cc = Dd(0); term = Dd(1); n = 0
        while True:
            if n % 4 == 0: cc += term
            elif n % 4 == 1: s += term
            elif n % 4 == 2: cc -= term
            else: s -= term
            n += 1; term = term * x / n
            if abs(term) < Dd(10) ** (-c.prec - 5) and n > 5: break
    return +s, +cc


class Cx:
    __slots__ = ('r', 'i')
    def __init__(s, r, i=Dd(0)): s.r = Dd(r); s.i = Dd(i)
    def __add__(s, o): o = o if isinstance(o, Cx) else Cx(o); return Cx(s.r + o.r, s.i + o.i)
    __radd__ = __add__
    def __sub__(s, o): o = o if isinstance(o, Cx) else Cx(o); return Cx(s.r - o.r, s.i - o.i)
    def __mul__(s, o): o = o if isinstance(o, Cx) else Cx(o); return Cx(s.r * o.r - s.i * o.i, s.r * o.i + s.i * o.r)
    __rmul__ = __mul__
    def __truediv__(s, o):
        o = o if isinstance(o, Cx) else Cx(o); n = o.r * o.r + o.i * o.i; return Cx((s.r * o.r + s.i * o.i) / n, (s.i * o.r - s.r * o.i) / n)


def cexp(z):
    e = dexp(z.r)
    if e == 0: return Cx(0)
    sn, cs = _sincos(z.i); return Cx(e * cs, e * sn)


def csin(z):
    sn, cs = _sincos(z.r); ep = dexp(z.i); em = dexp(-z.i); return Cx(sn * (ep + em) / 2, cs * (ep - em) / 2)


def _inv2(T):
    det = T[0][0] * T[1][1] - T[0][1] * T[1][0]; return [[T[1][1] / det, -T[0][1] / det], [-T[1][0] / det, T[0][0] / det]], det


def prepare(terms, sig):
    sg = Dd(repr(float(sig))); s2 = sg * sg; pre = []
    for lc, mu, S, kap in terms:
        S = [[Dd(repr(float(S[a][b]))) for b in range(2)] for a in range(2)]
        T = [[S[a][b] + (s2 if a == b else 0) for b in range(2)] for a in range(2)]; Ti, det = _inv2(T)
        K = [[sum(S[a][c] * Ti[c][b] for c in range(2)) for b in range(2)] for a in range(2)]
        C = [[s2 * ((1 if a == b else 0) - s2 * Ti[a][b]) for b in range(2)] for a in range(2)]
        kp = [Dd(repr(float(kap[0]))), Dd(repr(float(kap[1])))]; mu = [Dd(repr(float(mu[0]))), Dd(repr(float(mu[1])))]
        Kk = [K[a][0] * kp[0] + K[a][1] * kp[1] for a in range(2)]; Ck = [C[a][0] * kp[0] + C[a][1] * kp[1] for a in range(2)]
        q = sum(kp[a] * C[a][b] * kp[b] for a in range(2) for b in range(2)); vz = (C[0][0] + 2 * C[0][1] + C[1][1]) / 2
        c0 = Dd(repr(float(lc))) - (2 * pi() * det.sqrt()).ln()
        pre.append((c0, mu, Ti, K, Kk, Ck, q, vz))
    return pre


def pD(pre, y):
    r2 = Dd(2).sqrt(); P = Cx(0); PF = Cx(0)
    for c0, mu, Ti, K, Kk, Ck, q, vz in pre:
        d = [y[0] - mu[0], y[1] - mu[1]]
        quad = sum(d[a] * Ti[a][b] * d[b] for a in range(2) for b in range(2))
        g = cexp(Cx(c0 - quad / 2 - q / 2, Kk[0] * d[0] + Kk[1] * d[1]))
        m = [Cx(mu[a] + K[a][0] * d[0] + K[a][1] * d[1], Ck[a]) for a in range(2)]; mz = (m[0] + m[1]) * Cx(1 / r2)
        F = mz - csin(mz) * Cx(dexp(-vz / 2)); P = P + g; PF = PF + g * F
    return P.r, (PF / P).r


def fields(terms, sig, pts):
    pre = prepare(terms, sig); out = []; sg = Dd(repr(float(sig))); h = sg * Dd('1e-25'); r2 = Dd(2).sqrt()
    for x in pts:
        y = [Dd(repr(float(x[0]))), Dd(repr(float(x[1])))]
        p, D = pD(pre, y)
        pv = (pD(pre, [y[0] + h, y[1]])[0] - pD(pre, [y[0] - h, y[1]])[0]) / (2 * h)
        pw = (pD(pre, [y[0], y[1] + h])[0] - pD(pre, [y[0], y[1] - h])[0]) / (2 * h)
        dpD = (pD(pre, [y[0] + h / r2, y[1] - h / r2])[1] - pD(pre, [y[0] - h / r2, y[1] + h / r2])[1]) / (2 * h)
        out.append({'p': float(p), 'svv': float((pv / p) ** 2), 'sww': float((pw / p) ** 2), 'Nibp': float(-dpD)})
    return out
