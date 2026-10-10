# Quantum trace estimates for Frobenius

Roger Cooperman and A.E. Petrunina V.V. — corrected validation package.

Seven scripts test finite instances of the corresponding results in the corrected white paper. They do not prove the theorems or Conjecture 1.

| Script | Checks |
|---|---|
| v1 | Exhaustive point counts, exact Newton identities, exact Weil membership, twists |
| v2 | Numerical unitary/symplectic realizations; complete exact Weil census; boundary regressions |
| v3 | Estimator laws; actual small-circuit call instrumentation; independent O1/O2 errors and ancilla leakage |
| v4 | Congruence recovery; full-order abstract and curve witnesses; exact squared distances; former-table regression |
| v5 | Point permutations, gcd fixed-point formula, cycles, coordinate linearity |
| v6 | Same-target comparisons with j dividing m; finite-grid AE experiments; nondivisor regression |
| v7 | Numerical circuit synthesis and recovery; conservative accuracy margin; phase-distance regression |

## Run

Use a virtual environment. Isolated mode (`-I`) intentionally ignores user-site packages.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -I run_all.py
```

On Windows use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Each script can also run separately. The runner exits unsuccessfully if any script fails.
Allow roughly two minutes on hardware comparable to the recorded reference environment.

## Exact versus numerical checks

Finite-field counts, Newton identities, Weil-class membership and witness squared distances use exact integer or rational arithmetic. The exact Weil test uses Hermite moment and localizing matrices and has no floating root filter. Its exhaustive enumeration is not intended for large inputs.

Spectral routines explicitly named `numeric`, statevectors, analytic probability evaluations and circuit simulations use floating-point arithmetic. Matrix tolerances and sampled confidence checks are finite tests. `v3` distinguishes actual counted circuit applications in six small experiments from formula-based cost bookkeeping in larger analytic sampling. Twelve additional full-circuit experiments include independently perturbed O1/O2, global phases, and ancilla leakage. `v6` uses a 361-point grid and does not certify a continuum optimum. `v7` uses a chosen-phase norm as an upper bound, not an exact operator-norm phase minimization; its recovery tests are not an arbitrary-precision implementation of Proposition 5.

The Q=9 curve examples are base changes of prime-field coefficient models. No census of all extension-field coefficient curves is claimed. No test resolves the conjecture or establishes an asymptotic lower bound.

## Results and integrity

`ENVIRONMENT.json` records the actual corrected run; requirements are pinned to it. `results/*.json` and `results/logs/` contain fresh outputs. `results/SUMMARY.json` records all seven exit statuses. Seeds are fixed, but floating-point values can vary slightly across environments.

Before running, verify `sha256sum -c SHA256SUMS`. Running overwrites result files and timing metadata, so verify the distributed manifest first.

## License

MIT License, Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V. Full text is in LICENSE and in every Python file.

Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
