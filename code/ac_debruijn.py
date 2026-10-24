"""Stage AC: the directional de Bruijn identity dJ_ww/dt = -E[H_vw^2 + H_ww^2] checked by a central difference of ln J on ONE fixed fine grid
(the grids of ac_heat depend on t, which adds discretisation noise to a 1% finite difference).  Mixtures of stage X1 and the AB configurations.
-> results/ac_debruijn.json.  Python 3.11."""
import json
import numpy as np
from ac_heat import sums_grid, states_mix, states_points
from z_chain import mixture_grid

if __name__ == '__main__':
    out = {}
    for k, T, s in states_mix()[::2] + states_points():
        t = s * s; vn, wn = mixture_grid(T, np.sqrt(0.97 * t), 12)
        if len(vn) * len(wn) > 6e6: continue
        E = [sums_grid(T, m * t, vn, wn) for m in (0.99, 1.0, 1.01)]
        d = E[1]['J'] * np.log(E[2]['J'] / E[0]['J']) / (0.02 * t); rhs = -(E[1]['Hvw2'] + E[1]['Hww2'])
        out[k] = {'dJdt_fd': d, 'minus_E_H2': rhs, 'rel': abs(d / rhs - 1), 'tR': t * (-rhs) / E[1]['J'], 'n_points': len(vn) * len(wn)}
        print('%-40s rel %.1e  tR %.3g' % (k, out[k]['rel'], out[k]['tR']), flush=True)
    json.dump(out, open('results/ac_debruijn.json', 'w'), indent=1)
