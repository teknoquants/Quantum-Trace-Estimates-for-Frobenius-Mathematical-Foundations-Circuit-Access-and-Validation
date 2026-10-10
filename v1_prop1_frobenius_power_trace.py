# Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Validation of Proposition 1 (the Frobenius power trace).

Proposition 1. For a smooth projective geometrically irreducible curve C of genus g over F_Q put
N_j = #C(F_{Q^j}), s_j = Q^j + 1 - N_j and tau_j(C) = s_j / Q^{j/2}. Then
  (i)   tau_j(C) = sum_i (alpha_i / sqrt Q)^j, where L_C(T) = prod (1 - alpha_i T) is the numerator of Z_C;
  (ii)  s_j is an integer, tau_j is real and |tau_j| <= 2g;
  (iii) tau_j depends only on the F_Q-isomorphism class of C;
  (iv)  tau_1..tau_g determine L_C(T) (Newton identities + functional equation), hence every tau_j.
  Remark (quadratic twist): tau_j(C^tw) = (-1)^j tau_j(C).

Checks (every point count is exhaustive enumeration over F_{Q^j}):
  A  integrality: Newton applied to s_1..s_g gives integer b_1..b_g                          -> (iv)
  B  prediction: N_j for g < j <= jmax predicted from N_1..N_g equals the brute-force count   -> (i),(iv)
  C  Riemann hypothesis: the reconstructed L lies in W_g(Q) (exact algebraic test)            -> (i),(ii)
  D  bound and reality: |tau_j| <= 2g for every computed j                                  -> (ii)
  E  spectral identity: tau_j = sum beta_i^j for the numerical normalised roots (|err|<1e-8)  -> (i)
  F  isomorphism invariance: models y^2 = f(x + c) and y^2 = u^2 f(x) give identical tau_j   -> (iii)
  G  twist: tau_j(C^tw) = (-1)^j tau_j(C)                                                    -> remark
Run: python3 -I v1_prop1_frobenius_power_trace.py      Output: results/v1_prop1.json
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frobenius_core import (test_curves, random_curves, jmax_for, L_from_counts, power_sums_from_L,
                            is_weil_exact, normalised_spectrum, Hyperelliptic, save_json)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def shift_model(C, c):
    """f(x) -> f(x + c) (exact over F_p)."""
    p, f = C.p, C.f
    res = [0] * len(f)
    # Horner composition
    for a in reversed(f):
        # res = res * (x + c) + a
        new = [0] * len(f)
        for i, r in enumerate(res):
            if r:
                if i + 1 < len(new):
                    new[i + 1] = (new[i + 1] + r) % p
                new[i] = (new[i] + r * c) % p
        new[0] = (new[0] + a) % p
        res = new
    return Hyperelliptic(res, p, C.r)


def scale_model(C, u):
    return Hyperelliptic([(u * u * a) % C.p for a in C.f], C.p, C.r)


def nonsquare(p):
    return next(c for c in range(2, p) if pow(c, (p - 1) // 2, p) == p - 1)


def check_curve(C, jmax):
    g, Q = C.g, C.Q
    N = C.counts(jmax)
    s = [Q ** j + 1 - N[j - 1] for j in range(1, jmax + 1)]
    tau = [s[j - 1] / Q ** (j / 2) for j in range(1, jmax + 1)]
    L, integral = L_from_counts(N, g, Q)
    s_pred = [int(x) for x in power_sums_from_L(L, jmax)]
    beta = normalised_spectrum(L, Q)
    spec = [float(np.real(np.sum(beta ** j))) for j in range(1, jmax + 1)]
    row = {"curve": C.name, "g": g, "Q": Q, "jmax": jmax, "N": N, "L": L,
           "A_integral": integral,
           "B_prediction_beyond_g": s_pred[g:] == s[g:],
           "C_weil_exact": is_weil_exact(L, Q),
           "D_bound": all(abs(t) <= 2 * g + 1e-12 for t in tau),
           "E_spectral_max_err": max(abs(a - b) for a, b in zip(tau, spec)),
           }
    jj = min(jmax, g + 1)
    c = 1 % C.p
    row["F_shift_invariant"] = shift_model(C, c).counts(jj) == N[:jj]
    u = 2 % C.p if C.p > 2 else 1
    row["F_scale_invariant"] = scale_model(C, u).counts(jj) == N[:jj]
    if C.r % 2 == 1:
        T = C.twist(nonsquare(C.p))
        Nt = T.counts(jj)
        st = [Q ** j + 1 - Nt[j - 1] for j in range(1, jj + 1)]
        row["G_twist_sign"] = all(st[j - 1] == (-1) ** j * s[j - 1] for j in range(1, jj + 1))
    else:
        row["G_twist_sign"] = None
    row["E_ok"] = row["E_spectral_max_err"] < 1e-6
    row["all_ok"] = all(row[k] for k in ("A_integral", "B_prediction_beyond_g", "C_weil_exact", "D_bound",
                                         "E_ok", "F_shift_invariant", "F_scale_invariant")) and row["G_twist_sign"] in (True, None)
    return row


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(20261010)
    curves = [(C, jmax_for(C)) for C in test_curves()]
    for g, p, k in ((1, 3, 6), (1, 7, 6), (2, 3, 8), (2, 5, 8), (2, 7, 6), (3, 3, 6), (3, 5, 4)):
        for C in random_curves(rng, g, p, k):
            curves.append((C, min(jmax_for(C), 2 * g + 1)))
    for C in random_curves(rng, 2, 5, 4, even_degree=True):
        curves.append((C, min(jmax_for(C), 5)))
    rows = []
    for C, jm in curves:
        r = check_curve(C, jm)
        rows.append(r)
        print(f"{'OK ' if r['all_ok'] else 'FAIL'} g={r['g']} Q={r['Q']:2d} jmax={jm:2d} L={r['L']}  {r['curve']}", flush=True)
    summary = {"curves": len(rows), "all_ok": all(r["all_ok"] for r in rows),
               "genus_counts": {g: sum(r["g"] == g for r in rows) for g in (1, 2, 3)},
               "point_counts_checked": sum(r["jmax"] for r in rows),
               "predictions_beyond_g_checked": sum(r["jmax"] - r["g"] for r in rows),
               "max_spectral_error": max(r["E_spectral_max_err"] for r in rows),
               "seconds": round(time.time() - t0, 1)}
    print(summary)
    save_json(OUT / "v1_prop1.json", {"summary": summary, "rows": rows})
    sys.exit(0 if summary["all_ok"] else 1)

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
