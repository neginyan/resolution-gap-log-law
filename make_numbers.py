"""Paper E numbers: paperE/paperE_numbers.md (every number with its source file and key) and paperE/results/paperE_numbers.json.
Built ONLY from paperE/results.  Run:  python paperE/make_numbers.py   (Python 3.11)"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
import paperE_data as D

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(v):
    if isinstance(v, (bool, str)): return str(v)
    if v == int(v) and abs(v) < 1e6: return '%d' % v
    return '%.4g' % v if (abs(v) < 1e-3 or abs(v) >= 1e4) else '%.4f' % v


if __name__ == '__main__':
    E = D.numbers()
    L = ['# Paper E — numbers and their sources', '',
         'Every number below is computed by `paperE/code/paperE_data.py` from the files in `paperE/results/` only (source: `file.json:key`).',
         'Symbols: pendulum kappa = 1; gap = (1 - R)/sigma with the cancellation-free formula 1 - R = (2 sigma^2 J_ww + N)/(sigma^2 (J_vv + J_ww)); a = sigma^2 J_vv;',
         'P# = sigma^2 E[u^2 s_v^2] J_ww / a^2; p~ = omega-weighted W-marginal; Lambda, x as in stages S-T; strips: rho = N(v;0,Lv^2) N(w;0,Lw^2)(1 + c e^{-w^2/2 ell^2} cos(k0 v/sigma)).', '']
    sec = None
    for e in E:
        if e['sec'] != sec:
            sec = e['sec']; L += ['', '## ' + sec, '', '| quantity | value | source |', '|---|---|---|']
        L.append('| %s | %s | `%s` |' % (e['name'], fmt(e['value']), e['src']))
    open(os.path.join(HERE, 'paperE_numbers.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(E, open(os.path.join(HERE, 'results', 'paperE_numbers.json'), 'w'), indent=1)
    print('%d numbers' % len(E))
