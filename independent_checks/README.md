# Independent checks

These checks were written separately from the study code in `code/`. They import nothing from it and read no stored results.

| file | content |
|---|---|
| `strip_gap.py` | a closed-form two-scale evaluator of the ripple strips (complex-Gaussian terms; the nonlinear term in posterior-covariance form, with no cancellation) |
| `check_paperE.py` | 12 checks: the strip minima of Table II at σ = 1e-4 … 1e-96 from the optimal parameters alone; the absence of deep valleys on the σ = 1e-2 strip (Fourier grid); Lemma 5 on 200 random structures; Lemma 11 on 150 random one-dimensional mixtures |

Run with `python independent_checks/check_paperE.py` (about ten seconds; exit code 0 iff all checks pass).
