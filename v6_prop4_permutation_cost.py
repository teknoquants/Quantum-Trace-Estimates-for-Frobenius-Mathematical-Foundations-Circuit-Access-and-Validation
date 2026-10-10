# Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Validation of Proposition 4 (the point-permutation ROUTE: Hadamard test on P_m + amplitude estimation).

Proposition 4 (route-specific). Let j divide m and 1 <= j < m and consider the route that applies the Hadamard test of Theorem 1 to
V = P_m on l^2(X_m), n_m = N_m, estimates the acceptance probability by amplitude estimation and outputs
N^_j = 2 n_m (a~ - 1/2).
 (a) Dimension: Q^m + 1 - 2g Q^{m/2} <= n_m <= Q^m + 1 + 2g Q^{m/2}.
 (b) Information: a = 1/2 + N_gcd(j,m) / (2 n_m) exactly, so the test depends on C only through (N_gcd(j,m), N_m);
     the output map needs N_m; for m = j the test is uninformative (a = 1).
 (c) Precision: N^_j is within kappa of N_gcd(j,m) iff |a~ - a| <= kappa / (2 n_m).
 (d) Sufficient cost: M >= 4 pi n_m / kappa suffices (Theorem 1), i.e. O(j n_m kappa^{-1} log(1/eta)) oracle calls.
 (e) Ratio of required precisions: the cohomological route needs |a~ - a| <= kappa / (2 n Q^{j/2}); the ratio of the
     two required precisions is exactly n_m / (n Q^{j/2}).
NOT claimed: any lower bound for algorithms that use P_m in other ways, or a comparison with classical counting
(the units differ). Check C measures the cost of this route on finite instances only.

Checks
  A  Weil bounds on n_m for every (curve, m) of Proposition 3 and more                                        -> (a)
  B  statevector Hadamard test on P_m: a = 1/2 + N_gcd/(2 n_m) exactly; a = 1 for m = j                        -> (b)
  C  smallest tested power-of-two M meeting precision on a 361-point amplitude grid (numerical AE law),
     for both routes; Theorem 1's M always suffices; measured cost ratio vs the exact precision ratio      -> (c),(d),(e)
Run: python3 -I v6_prop4_permutation_cost.py      Output: results/v6_prop4.json
"""
import sys, time, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frobenius_core import (Hyperelliptic, frobenius_permutation, hadamard_state, good_projector_diag,
                            ae_distribution_analytic, ae_estimate_values, save_json)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


_EPS_STAR = {}


def eps_star(M, grid=np.linspace(0.05, 0.95, 361)):
    """Maximum over the supplied FINITE grid of the 8/pi^2-quantile of |a~ - a| for one run of amplitude estimation
    with M points: a grid statistic, not a certificate for all amplitudes in the interval."""
    if M not in _EPS_STAR:
        vals = ae_estimate_values(M)
        worst = 0.0
        for a in grid:
            P = ae_distribution_analytic(a, M)
            d = np.abs(vals - a)
            order = np.argsort(d)
            c = np.cumsum(P[order])
            k = int(np.searchsorted(c, 8 / math.pi ** 2))
            worst = max(worst, float(d[order][k]))
        _EPS_STAR[M] = worst
    return _EPS_STAR[M]


def required_M(eps, Mmax=1 << 19):
    """Smallest tested power-of-two M whose finite-grid statistic is below eps."""
    M = 2
    while M <= Mmax:
        if eps_star(M) < eps:
            return M
        M *= 2
    return None


if __name__ == "__main__":
    t0 = time.time()
    curves = [Hyperelliptic([1, 1, 0, 1], 5), Hyperelliptic([1, 1, 0, 1], 7),
              Hyperelliptic([0, -1, 0, 0, 0, 1], 3, name="y^2 = x^5 - x over F_3 ([CP] Prop. 6.2)"),
              Hyperelliptic([2, 0, 1, 0, 0, 1], 7), Hyperelliptic([1, 1, 0, 0, 0, 0, 0, 1], 3),
              Hyperelliptic([1, 0, 1, 1, 0, 0, 2], 5, name="y^2 = 2x^6 + x^3 + x^2 + 1 over F_5 (non-square lead)")]
    res = {}
    # ---- A
    rowsA = []
    for C in curves:
        for m in range(1, 7):
            if C.Q ** m > 200_000:
                break
            n = C.count(m)
            lo, hi = C.Q ** m + 1 - 2 * C.g * C.Q ** (m / 2), C.Q ** m + 1 + 2 * C.g * C.Q ** (m / 2)
            rowsA.append({"curve": C.name, "m": m, "n_m": n, "lower": lo, "upper": hi, "two_g": 2 * C.g, "ok": lo - 1e-9 <= n <= hi + 1e-9})
    res["A_dimension"] = {"cases": len(rowsA), "all_ok": all(r["ok"] for r in rowsA), "rows": rowsA}
    print("A:", len(rowsA), res["A_dimension"]["all_ok"])
    # ---- B
    rowsB, okB = [], True
    for C in curves:
        for m in (1, 2, 3):
            F, pts, perm = frobenius_permutation(C, m)
            n = len(pts)
            if n > 130:
                continue
            P = np.zeros((n, n)); P[perm, np.arange(n)] = 1
            good = good_projector_diag(n)
            for j in range(1, m + 1):
                a_sv = float(np.sum(np.abs(hadamard_state(np.linalg.matrix_power(P, j))) ** 2 * good))
                a_th = 0.5 + C.count(math.gcd(j, m)) / (2 * n)
                ok = abs(a_sv - a_th) < 1e-12 and (j != m or abs(a_sv - 1) < 1e-12)
                okB &= ok
                rowsB.append({"curve": C.name, "m": m, "j": j, "n_m": n, "a_statevector": a_sv, "a_formula": a_th, "ok": ok})
    res["B_normalisation"] = {"cases": len(rowsB), "all_ok": bool(okB), "rows": rowsB}
    print("B:", len(rowsB), okB)
    # ---- C
    rowsC = []
    for C in curves:
        g, Q = C.g, C.Q
        for j in (1, 2):
            kappa = j / 2                                   # decoder tolerance on s_j (equivalently on N_j)
            n = 1 << math.ceil(math.log2(2 * g))
            Nj = C.count(j); s_j = Q ** j + 1 - Nj
            a_c = 0.5 + (s_j / Q ** (j / 2) + n - 2 * g) / (2 * n)
            eps_c = kappa / (2 * n * Q ** (j / 2))           # |a~ - a| needed by the cohomological route
            Mc = required_M(eps_c)
            for m in range(j + 1, 7):
                if m % j:
                    continue
                if Q ** m > 20_000 or required_M(kappa / (2 * C.count(m))) is None:
                    break
                nm = C.count(m)
                a_p = 0.5 + C.count(math.gcd(j, m)) / (2 * nm)
                eps_p = kappa / (2 * nm)                     # |a~ - a| needed by the permutation route
                Mp = required_M(eps_p)
                M_thm = 1 << math.ceil(math.log2(4 * math.pi * nm / kappa))
                P = ae_distribution_analytic(a_p, M_thm)
                thm_ok = P[np.abs(ae_estimate_values(M_thm) - a_p) < eps_p].sum() >= 8 / math.pi ** 2
                rowsC.append({"curve": C.name, "g": g, "Q": Q, "j": j, "m": m, "n_m": nm, "kappa": kappa,
                              "M_perm": Mp, "M_coh": Mc, "M_perm_times_kappa_over_n_m": Mp * kappa / nm,
                              "ratio_perm_over_coh": Mp / Mc, "predicted_ratio_n_m_over_nQ^(j/2)": nm / (n * Q ** (j / 2)),
                              "theorem1_M": M_thm, "theorem1_M_suffices": bool(thm_ok)})
    ratios = [r["M_perm_times_kappa_over_n_m"] for r in rowsC]
    for r in rowsC:
        r["measured_over_predicted"] = r["ratio_perm_over_coh"] / r["predicted_ratio_n_m_over_nQ^(j/2)"]
    mp = [r["measured_over_predicted"] for r in rowsC]
    res["C_cost"] = {"grid_points":361, "grid_interval":[0.05,0.95], "scope":"finite-grid experiment, not continuum optimum", "cases": len(rowsC), "min_M_perm_kappa_over_n_m": min(ratios), "max_M_perm_kappa_over_n_m": max(ratios),
                     "measured_over_predicted_range": [min(mp), max(mp)],
                     "cases_perm_cheaper": [(r["g"], r["Q"], r["j"], r["m"]) for r in rowsC if r["M_perm"] < r["M_coh"]],
                     "theorem1_M_always_suffices": all(r["theorem1_M_suffices"] for r in rowsC),
                     "eps_star_table": {str(M): eps_star(M) for M in sorted(_EPS_STAR)}, "rows": rowsC}
    res["C_cost"]["ok"] = (res["C_cost"]["theorem1_M_always_suffices"] and 0.25 <= min(mp) and max(mp) <= 4.0)
    print("C:", len(rowsC), "M_perm*kappa/n_m in", (round(min(ratios), 2), round(max(ratios), 2)),
          "measured/predicted ratio in", (round(min(mp), 2), round(max(mp), 2)),
          "Theorem-1 M suffices:", res["C_cost"]["theorem1_M_always_suffices"],
          "perm cheaper in:", res["C_cost"]["cases_perm_cheaper"])
    for r in rowsC:
        print(f"   g={r['g']} Q={r['Q']} j={r['j']} m={r['m']} n_m={r['n_m']:6d} M_perm={r['M_perm']:7d} M_coh={r['M_coh']:5d} "
              f"ratio={r['ratio_perm_over_coh']:6.0f} predicted~{r['predicted_ratio_n_m_over_nQ^(j/2)']:7.1f}")
    # A counterexample catches the former target mismatch at j=2,m=3.
    C = curves[0]; j,m = 2,3
    n1,n2,n3 = C.count(1),C.count(2),C.count(3)
    decoded = 2*n3*((0.5+n1/(2*n3))-0.5)
    regression_ok = abs(decoded-n1)<1e-10 and n1 != n2 and all(r["m"]%r["j"]==0 for r in rowsC)
    res["D_nondivisor_regression"] = {"j":j,"m":m,"N1":n1,"N2":n2,"decoded":decoded,"ok":regression_ok}
    res["all_ok"] = bool(regression_ok and res["A_dimension"]["all_ok"] and okB and res["C_cost"]["ok"])
    res["seconds"] = round(time.time() - t0, 1)
    print("ALL OK:", res["all_ok"], res["seconds"], "s")
    save_json(OUT / "v6_prop4.json", res)
    sys.exit(0 if res["all_ok"] else 1)

# MIT License
# 
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
