# Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Validation of Definition 3 / Proposition 2 (the quantum trace of Frobenius).

Definition 3. A (delta, eta)-quantum trace of order j of C is a random variable p^_j produced by a quantum
algorithm whose only access to C is a Frobenius oracle (Definition 2), with Pr(|p^_j - tau_j(C)| <= delta) >= 1 - eta.

Proposition 2.
 (a) Well-posed: the condition depends only on C, not on the realisation U~ in R(C), nor on the padding n.
 (b) Existence: for every delta in (0, 2g] and eta in (0, 1) a (delta, eta)-quantum trace of order j exists,
     using at most 2 j r M oracle calls (Theorem 1).
 (c) Sufficiency: (delta, eta)-quantum traces of orders 1..g with delta < delta_cong(g, Q) determine L_C(T)
     through the congruence decoder with probability >= 1 - g eta.
 (d) Necessity OF THE CONTRACT (records of orders j <= 2g): for delta >= delta_cong(g, Q) two distinct members of
     W_g(Q) admit one common record satisfying Definition 3 for both, so no decoder of such records alone recovers
     both. This is a statement about the guarantee of Definition 3, not a lower bound for algorithms with oracle
     access. It is restricted to orders j <= 2g: at higher orders the witness pairs can separate further.

Checks
  A  invariance: exact single-run success law for 3 random realisations and 2 paddings, all >= 8/pi^2     -> (a)
  B  end-to-end sampling of (delta, eta)-quantum traces on every test curve: coverage >= 1 - eta          -> (b)
  C  decoding: L recovered from sampled quantum traces (delta = 0.95 delta_cong) with frequency >= 1 - g eta;
     and deterministically for every sign pattern of errors of size 0.999 delta_cong                      -> (c)
  D  witness pairs of W_g(Q) (g = 1..4) and explicit curve pairs found by complete census (g = 1, 2):
     record distance 2 delta_cong, midpoint decoded to at most one member                                  -> (d)
  E  scope of (d): for the witness pairs, the record distance over orders j <= 2g is exactly 2 delta_cong, but at
     orders 2g < j <= 4g it is larger in most cases (so (d) must not be stated for higher-order records)
Run: python3 -I v4_prop2_quantum_trace_estimate.py      Output: results/v4_prop2.json
"""
import sys, time, math, itertools
from fractions import Fraction
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frobenius_core import (test_curves, L_from_counts, usp_realisation_numeric, haar_unitary, padded,
                            hadamard_state, good_projector_diag, ae_distribution_analytic, ae_estimate_values,
                            runs_for_confidence, M_for_precision, delta_cong, congruence_decoder,
                            power_sums_from_L, Hyperelliptic, squarefree, field, save_json)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def sample_quantum_trace(tau_j, g, n, delta, eta, rng):
    """One (delta, eta)-quantum trace via Theorem 1 (exact AE law, median of r runs)."""
    M, r = M_for_precision(n, delta), runs_for_confidence(eta)
    a = 0.5 + (tau_j + n - 2 * g) / (2 * n)
    P = ae_distribution_analytic(a, M)
    ys = rng.choice(M, size=r, p=P)
    est = 2 * n * (np.sin(math.pi * ys / M) ** 2 - 0.5) - (n - 2 * g)
    return float(np.median(est))


def witness_pairs(g, Q):
    L0 = [0] * (2 * g + 1)
    for i in range(g + 1):
        L0[2 * i] = Q ** i
    L1 = L0[:]
    for i in range(g):
        L1[2 * i + 1] = -Q ** i
    M0 = [0] * (2 * g + 1); M0[0] = 1; M0[-1] = Q ** g
    M1 = M0[:]; M1[g] += 1
    return (L0, L1), (M0, M1)


def record(L, Q, m):
    s = [int(x) for x in power_sums_from_L(L, m)]
    return np.array([s[j - 1] / Q ** (j / 2) for j in range(1, m + 1)])


def exact_distance_squared(A, B, Q, m):
    """Squared sup-distance, a rational number despite odd-order square roots."""
    sa,sb=power_sums_from_L(A,m),power_sums_from_L(B,m)
    return max((sa[j-1]-sb[j-1])**2/Fraction(Q**j) for j in range(1,m+1))


def census_counts(p, deg, max_curves=None):
    """N_1, N_2 for every squarefree f of degree deg over F_p with leading coefficient 1 or a fixed non-square."""
    F1, F2 = field(p, 1), field(p, 2)
    out = {}
    leads = [1] + [next(c for c in range(2, p) if pow(c, (p - 1) // 2, p) == p - 1)]
    for lead in leads:
        for tail in itertools.product(range(p), repeat=deg):
            f = list(tail) + [lead]
            if not squarefree(f, p):
                continue
            C = Hyperelliptic(f, p)
            Ns = (C.count(1), C.count(2))
            out.setdefault(Ns, f)
            if max_curves and len(out) >= max_curves:
                return out
    return out


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(20261013)
    data = []
    for C in test_curves():
        N = C.counts(C.g)
        L, _ = L_from_counts(N, C.g, C.Q)
        tau = [(C.Q ** j + 1 - N[j - 1]) / C.Q ** (j / 2) for j in range(1, C.g + 1)]
        data.append((C, L, tau))
    res = {}
    # ---- A invariance
    minA, nA = 1.0, 0
    for C, L, tau in data:
        g = C.g
        U = usp_realisation_numeric(L, C.Q)
        n0 = 1 << math.ceil(math.log2(2 * g))
        delta = delta_cong(g, C.Q)
        for n in (n0, 2 * n0):
            for _ in range(3):
                W = haar_unitary(2 * g, rng)
                V = padded(W @ U @ W.conj().T, n)
                good = good_projector_diag(n)
                M = M_for_precision(n, delta)
                vals = ae_estimate_values(M)
                for j in range(1, g + 1):
                    a = float(np.sum(np.abs(hadamard_state(np.linalg.matrix_power(V, j))) ** 2 * good))
                    P = ae_distribution_analytic(a, M)
                    phat = 2 * n * (vals - 0.5) - (n - 2 * g)
                    minA = min(minA, float(P[np.abs(phat - tau[j - 1]) <= delta].sum())); nA += 1
    res["A_invariance"] = {"cases": nA, "min_single_run_success": minA, "ok": minA >= 8 / math.pi ** 2 - 1e-12}
    print("A:", res["A_invariance"])
    # ---- B coverage
    eta = 0.05
    rowsB = []
    for C, L, tau in data:
        g = C.g; n = 1 << math.ceil(math.log2(2 * g)); delta = 0.95 * delta_cong(g, C.Q)
        trials, hit = 1500, 0
        for _ in range(trials):
            j = int(rng.integers(1, g + 1))
            hit += abs(sample_quantum_trace(tau[j - 1], g, n, delta, eta, rng) - tau[j - 1]) <= delta
        rowsB.append({"curve": C.name, "coverage": hit / trials, "eta": eta})
    res["B_existence_coverage"] = {"rows": rowsB, "min_coverage": min(r["coverage"] for r in rowsB),
                                   "ok": min(r["coverage"] for r in rowsB) >= 1 - eta}
    print("B: min coverage", res["B_existence_coverage"]["min_coverage"])
    # ---- C decoding
    rowsC = []
    for C, L, tau in data:
        g, Q = C.g, C.Q; n = 1 << math.ceil(math.log2(2 * g)); dc = delta_cong(g, Q); delta = 0.95 * dc
        trials, ok = 600, 0
        for _ in range(trials):
            p_hat = [sample_quantum_trace(tau[j - 1], g, n, delta, eta, rng) for j in range(1, g + 1)]
            s_hat = [Q ** (j / 2) * p_hat[j - 1] for j in range(1, g + 1)]
            ok += congruence_decoder(s_hat, g, Q) == L
        worst = all(congruence_decoder([Q ** (j / 2) * (tau[j - 1] + sg[j - 1] * 0.999 * dc) for j in range(1, g + 1)], g, Q) == L
                    for sg in itertools.product((-1, 1), repeat=g))
        rowsC.append({"curve": C.name, "g": g, "recovery_rate": ok / trials, "required": 1 - g * eta, "worst_case_signs_ok": worst})
    res["C_sufficiency"] = {"rows": rowsC, "ok": all(r["recovery_rate"] >= r["required"] and r["worst_case_signs_ok"] for r in rowsC)}
    print("C:", [(r["g"], r["recovery_rate"], r["worst_case_signs_ok"]) for r in rowsC])
    # ---- D necessity
    rowsD = []
    for g in (1, 2, 3, 4):
        for Q in (2, 3, 4, 5, 7, 9, 25):
            dc = delta_cong(g, Q)
            kstar = min(range(1, g + 1), key=lambda k: k / Q ** (k / 2))
            A, B = witness_pairs(g, Q)[0 if kstar == 1 else 1]
            ra, rb = record(A, Q, 2*g), record(B, Q, 2*g)
            dist = float(np.max(np.abs(ra - rb)))
            mid = (ra + rb) / 2
            out = congruence_decoder([Q ** (j / 2) * mid[j - 1] for j in range(1, g + 1)], g, Q)
            assert exact_distance_squared(A,B,Q,2*g) == min(Fraction(k*k,Q**k) for k in range(1,g+1))
            rowsD.append({"g": g, "Q": Q, "pair": [A, B], "record_distance": dist, "two_delta_cong": 2 * dc,
                          "exact_squared_distance": str(exact_distance_squared(A,B,Q,2*g)), "distance_ok": abs(dist - 2 * dc) < 1e-12, "distinct_polynomials": A != B, "midpoint_within_both": bool(max(np.max(np.abs(mid-ra)), np.max(np.abs(mid-rb))) <= dc + 1e-12),
                          "decoder_fails_on_one": (out != A) or (out != B)})
    # Search the stated order range 1..2g, not merely 1..g.
    curve_pairs = []
    for g, Q in ((1,5),(1,7),(1,11),(1,13),(2,3),(2,5)):
        cen = {}
        for deg in ((3,) if g == 1 else (5,6)):
            cen.update(census_counts(Q, deg))
        items = []
        for Ns, f in cen.items():
            L, integral = L_from_counts(list(Ns), g, Q)
            assert integral
            items.append((Ns,f,L,record(L,Q,2*g)))
        found = None
        for x,y in itertools.combinations(items,2):
            dist = float(np.max(np.abs(x[3]-y[3])))
            if abs(dist-2*delta_cong(g,Q)) < 1e-12:
                midpoint = (x[3]+y[3])/2
                errors = [float(np.max(np.abs(midpoint-z[3]))) for z in (x,y)]
                found = {"g":g,"Q":Q,"f1":x[1],"f2":y[1],"counts_1_2":[x[0],y[0]],
                         "L1":x[2],"L2":y[2],"orders_checked":2*g,"distance":dist,
                         "two_delta_cong":2*delta_cong(g,Q),"midpoint_errors":errors,
                         "distinct_polynomials":x[2]!=y[2],"found":True}
                break
        assert found is not None, (g,Q,"no full-order curve witness in this census")
        A,B=found["L1"],found["L2"]
        exact_d2=exact_distance_squared(A,B,Q,2*g)
        assert exact_d2 == min(Fraction(k*k,Q**k) for k in range(1,g+1))
        found["exact_squared_distance"] = str(exact_d2)
        found["enumerated_counts_through_2g"] = []
        for model,L in ((found["f1"],A),(found["f2"],B)):
            counts=Hyperelliptic(model,Q).counts(2*g)
            predicted=[Q**j+1-int(t) for j,t in enumerate(power_sums_from_L(L,2*g),1)]
            assert counts == predicted
            found["enumerated_counts_through_2g"].append(counts)
        curve_pairs.append(found)
    # Regression: former Table 3 pairs must NOT pass an all-orders<=2g assertion.
    old = [(1,5,[[4],[5]]),(1,7,[[8],[9]]),(1,11,[[12],[11]]),
           (1,13,[[20],[19]]),(2,3,[[4,14],[3,13]]),(2,5,[[6,46],[6,44]])]
    old_rows = []
    for g,Q,counts in old:
        rr = [record(L_from_counts(N,g,Q)[0],Q,2*g) for N in counts]
        dist = float(np.max(np.abs(rr[0]-rr[1])))
        old_rows.append({"g":g,"Q":Q,"distance_through_2g":dist,
                         "passes":abs(dist-2*delta_cong(g,Q)) < 1e-12})
    assert sum(not r["passes"] for r in old_rows) == 4
    res["old_table_regression"] = old_rows
    res["D_necessity"] = {"witness_rows": rowsD, "curve_pairs": curve_pairs,
                          "ok": all(r["distance_ok"] and r["distinct_polynomials"] and r["midpoint_within_both"] and r["decoder_fails_on_one"] for r in rowsD) and all(c["found"] for c in curve_pairs)}
    print("D:", len(rowsD), "witness cases;", [(c["g"], c["Q"], c["found"]) for c in curve_pairs])
    # ---- E scope of (d)
    rowsE = []
    for g in (1, 2, 3, 4):
        for Q in (2, 3, 4, 5, 7, 9, 25):
            dc = delta_cong(g, Q)
            kstar = min(range(1, g + 1), key=lambda k: k / Q ** (k / 2))
            A, B = witness_pairs(g, Q)[0 if kstar == 1 else 1]
            ra, rb = record(A, Q, 4 * g), record(B, Q, 4 * g)
            d = np.abs(ra - rb)
            rowsE.append({"g": g, "Q": Q, "pair": "L" if kstar == 1 else "M",
                          "dist_orders_le_2g": float(d[:2 * g].max()), "two_delta_cong": 2 * dc,
                          "dist_orders_2g_to_4g": float(d[2 * g:].max()),
                          "separates_more_beyond_2g": bool(d[2 * g:].max() > 2 * dc + 1e-12),
                          "first_order_beyond": int(2 * g + 1 + np.argmax(d[2 * g:] > 2 * dc + 1e-12)) if d[2 * g:].max() > 2 * dc + 1e-12 else None})
    res["E_scope_orders"] = {"rows": rowsE, "cases": len(rowsE),
                             "cases_separating_more_beyond_2g": sum(r["separates_more_beyond_2g"] for r in rowsE),
                             "ok": all(abs(r["dist_orders_le_2g"] - r["two_delta_cong"]) < 1e-12 for r in rowsE)}
    print("E:", res["E_scope_orders"]["cases"], "cases;", res["E_scope_orders"]["cases_separating_more_beyond_2g"], "separate further at orders > 2g")
    res["all_ok"] = all(res[k]["ok"] for k in ("A_invariance", "B_existence_coverage", "C_sufficiency", "D_necessity", "E_scope_orders"))
    res["seconds"] = round(time.time() - t0, 1)
    print("ALL OK:", res["all_ok"], res["seconds"], "s")
    save_json(OUT / "v4_prop2.json", res)
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
