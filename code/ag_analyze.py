"""Stage AG (first milestone): summary (results/summary_AG.json) and figure (figures/stageAG.png/.pdf).  Python 3.11."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ae_const import qlcw

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

if __name__ == '__main__':
    L = json.load(open('results/ag_strip_lambda.json')); C = json.load(open('results/ag_counter.json'))
    C2 = json.load(open('results/ag_counter2.json')) if os.path.exists('results/ag_counter2.json') else {}
    AL = json.load(open('results/ag_align.json')) if os.path.exists('results/ag_align.json') else {}
    S = json.load(open('results/af_strip.json'))
    out = {'strip_lambda': {k: {'max': v['max_1e-8'], 'max_all': v['max_all'], 'monotone_decreasing': all(v['rows'][i + 1]['lam_max_all'] <= v['rows'][i]['lam_max_all'] + 1e-12 for i in range(len(v['rows']) - 1)),
                                'at_quarter': v['rows'][0]['lam_max_1e-8'], 'at_half': v['rows'][8]['lam_max_1e-8'], 'at_one': v['rows'][-1]['lam_max_1e-8']} for k, v in L.items()}}
    lam_star = max(v['max_all'] for v in out['strip_lambda'].values()); out['strip_lambda_star'] = lam_star
    out['main_line_check_sigma_0.01'] = {'quarter': [L['strip_gmin_0.01']['rows'][0]['lam_max_1e-8'], 1.874], 'half': [L['strip_gmin_0.01']['rows'][8]['lam_max_1e-8'], 1.756], 'one': [L['strip_gmin_0.01']['rows'][-1]['lam_max_1e-8'], 1.537]}
    # Proposition AG1 on the strips: t^2 R' <= lam* tR on [sigma^2/4, sigma^2]; check against the stored kappa_H / t^2R'
    out['strip_kappa_H_over_tR_max'] = max(v['kappa_H'] / v['tR'] for v in S.values())
    out['strip_t2R1_over_tR_max'] = max(v['t2R1'] / v['tR'] for v in S.values())
    for tau in (0.25, 0.5):
        out['strip_theorem_tau%.2f' % tau] = qlcw(lam_star, 0.0, tau)
    out['counter'] = {k: {q: v[q] for q in ('Y3', 'Y1', 'X', 'kappa_H', 't2R1', 'tJ')} for k, v in C.items() if k.startswith('opt_')}
    out['counter_size'] = {k: [{q: r[q] for q in ('scale', 'Y3', 'Y1', 'X', 't2R1', 'tJ')} for r in v] for k, v in C.items() if k.startswith('size_')}
    out['scissors_theta'] = [{q: r[q] for q in ('theta', 'Y3', 'X', 't2R1')} for r in C.get('scissors_theta', [])]
    out['counter2'] = {k: {q: v[q] for q in ('Y3', 'Y1', 'X', 't2R1', 'tJ')} | {'size_Y3': [r['Y3'] for r in v['size_scan']]} for k, v in C2.items()}
    allY = [v['Y3'] for v in out['counter'].values()] + [r['Y3'] for v in out['counter_size'].values() for r in v] + [v['Y3'] for v in out['counter2'].values()] + [y for v in out['counter2'].values() for y in v['size_Y3']]
    out['counter_Y3_max'] = max(allY)
    out['alignment'] = AL
    json.dump(out, open('results/summary_AG.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('strip_lambda', 'counter_size')}, indent=1, default=float))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2)); a = ax[0]
    for i, (k, v) in enumerate(L.items()):
        a.plot([r['m'] for r in v['rows']], [r['lam_max_all'] for r in v['rows']], '-', color=OI[i % 8], label=k.replace('strip_gmin_', '$\\sigma$ = '))
    a.axhline(3, color=OI[6], ls='--', lw=0.9, label='A = 3 (deep-valley threshold)'); a.set_xlabel('t/$\\sigma^2$'); a.set_ylabel('max$_y$ $\\lambda_{max}(-tH)$')
    a.set_title('strips: no deep valleys on the whole window'); a.legend(frameon=False, fontsize=6)
    b = ax[1]
    for i, (k, v) in enumerate(out['counter_size'].items()):
        b.plot([r['scale'] for r in v], [r['Y3'] for r in v], 'o-', color=OI[(i + 3) % 8], label=k.replace('size_', ''))
    b.set_xscale('log'); b.set_xlabel('size scale'); b.set_ylabel('$Y/(tJ_{ww})$ (A = 3, $a_0$ = 1/2)'); b.set_title('attempts to break Y: no growth with size'); b.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig('figures/stageAG.png', dpi=160); fig.savefig('figures/stageAG.pdf'); plt.close(fig)
