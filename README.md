# Is the resolution gap of the arrow of time uniform?

Code and data for the paper **"Is the resolution gap of the arrow of time uniform?"** by Taishi Namba.

- Paper: [paper DOI]
- Code (this repository, archived): [code DOI]

## What the paper shows

The paper studies the resolution gap of the coarse-grained arrow of time near the hyperbolic top of the pendulum and asks whether it
admits a lower bound gap ≥ cσ with a constant c independent of the blur σ.

- **Proposition 1 and Theorem 2.** A cancellation-free formula for the gap and a direct lower bound with an explicit remainder.
- **Lemma 3 and Corollary 4.** A line-wise Heisenberg inequality, and its consequence gap ≳ 1/√x̄ for every state, where x̄ measures the
  fineness of the state at the scale σ.
- **Lemmas 5 and 6, Theorem 7.** Smoothing–uncertainty in 1-d and with a split blur, giving the log law
  gap/σ ≥ c/√(ln(1/σ) + C) under the single checkable condition K_Λ ≤ K₁ and the assumptions (H1), (H2).
- **Proposition 8 and Conjecture 1.** Proposition 8: a target inequality on Λ implies the log law. Conjecture 1: the log law for all
  states satisfying (H1) and (H2). The stronger form (T2), tR ≤ A′ + B′ ln(1/(tJ_ww)), is what the theorems prove; it does not hold for
  all states with universal constants.
- **Lemma 9.** The Fisher form of the uncertainty step, the directional de Bruijn identity and the Γ₂ identity along the heat flow.
- **Lemma 10, Lemma 11, Theorem 12.** Integration along the heat flow; valleys of the blurred density are dips; and valleys are paid for
  by Fisher information (two-sided constant κ_0.1 = 267).
- **Theorem 13 and Proposition 14.** The log law without any constant to check for product states in any orientation, including states
  concentrated on a line, and for states without deep valleys on the blur window.
- **Numerical results 1–3.** Ripple strips have gap/σ → 1/(2k₀) as σ → 0, so no σ-uniform constant exists; their minimum gap follows
  1.02/√(ln(1/σ) − 23.4) down to σ = 1e-96; and the strips have no deep valleys on the whole blur window (largest posterior excess
  Λ* = 1.8745), so Proposition 14 applies to them.

The correspondence between the numbering of the paper and the rows of `claims.md` is given at the top of `claims.md`.
Every claim, with its status (proof / assumption / numerical / conjecture), is listed in `claims.md`.

## Contents

| path | content |
|---|---|
| `verification/run_all_E.py` | 65 checks of the claims from the stored results, plus a few recomputations with the packaged code (about 2 minutes on one core) |
| `make_numbers.py` | writes `paperE_numbers.md` (every number of the paper with its source `file.json:key`) and `results/paperE_numbers.json` |
| `paperE_numbers.md` | the numbers of the paper and where each comes from |
| `claims.md` | every claim with its status, the stage that produced it and the check that tests it |
| `results/*.json` | the stored results read by the numbers and the checks (and nothing else) |
| `code/paperE_data.py` | the single place where numbers are read from `results/` |
| `code/*.py` | the scripts that produced the results and every local module they import (list: `code/MODULES.txt`) |

## Reproduce

Python 3.11 or later.

```
pip install -r requirements.txt     # numpy, scipy, matplotlib
python make_numbers.py              # ~1 s; rebuilds paperE_numbers.md from results/
python verification/run_all_E.py    # ~2 min; exit code 0 iff all 65 checks pass
```

The verification can be run from any directory. It also recomputes a few quantities from scratch (the gap minimum of a stored mixture,
the de Bruijn and Γ₂ identities on a lattice, the valley lemma on a 1-d state, the strip window λ_max on a second grid).

The scripts in `code/` that produced the stored results were run in the layout of the original study (results written to `results/`);
several of them are long searches (hours on one core). They are included for transparency; the checks above do not rerun them.

Tested with numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2.

## Acknowledgments

The research was carried out with the help of the AI assistants Google Gemini, OpenAI ChatGPT and Anthropic Claude (derivations,
code, numerical experiments and cross-checks). The author directed the research and made all decisions.

## License

MIT (see `LICENSE`).

## Citation

Taishi Namba, *Is the resolution gap of the arrow of time uniform?*, [paper DOI]. Code: [code DOI].
