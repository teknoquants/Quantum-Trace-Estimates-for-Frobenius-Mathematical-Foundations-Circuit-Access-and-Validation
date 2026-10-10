# Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Validation of Proposition 3 (the point-permutation unitary of Frobenius).

Proposition 3. Let X_m = C(F_{Q^m}), n_m = |X_m| = N_m, and phi the Frobenius map (coordinates -> Q-th powers).
 (a) phi is a bijection of X_m with phi^m = id; the operator P_m |P> = |phi(P)> on l^2(X_m) is a real
     permutation matrix, hence unitary, with P_m^m = I.
 (b) Tr(P_m^j) = N_gcd(j, m) for every j >= 1; in particular Tr(P_m^j) = N_j when j | m.
 (c) The spectrum of P_m consists of m-th roots of unity: each phi-orbit of length l contributes every l-th
     root of unity once, and the number of orbits of length l is o_l = (1/l) sum_{d | l} mu(l/d) N_d
     (the number of closed points of degree l).
 (d) On coordinates, x -> x^Q is an F_p-linear bijection Phi of F_{Q^m} = F_p^{rm} with Phi^m = I; it factors into
     at most (rm)^2 + rm elementary row operations, i.e. a reversible circuit of poly(m log Q) gates.

Checks: (a)-(c) by explicit construction for genus 1, 2, 3 curves (odd and even degree, Q = p and Q = 9);
(d) by building Phi from basis images, testing phi(x) = Phi x for every field element, and factoring Phi.
Run: python3 -I v5_prop3_permutation_unitary.py      Output: results/v5_prop3.json
"""
import sys, time, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frobenius_core import Hyperelliptic, frobenius_permutation, field, mobius, save_json

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def frobenius_matrix(p, e, Q):
    """Matrix of x -> x^Q over F_p on F_{p^e} (digits basis)."""
    F = field(p, e)
    cols = []
    for i in range(e):
        b = p ** i
        cols.append(F.digits(F.power(np.array([b]), Q))[0])
    Phi = np.array(cols).T % p
    xs = F.elements()
    lhs = F.digits(F.power(xs, Q))
    rhs = (F.digits(xs) @ Phi.T) % p
    return Phi, bool(np.all(lhs == rhs))


def elementary_factorisation(A, p):
    """Gauss-Jordan elimination over F_p recording elementary operations; returns the count and a
    reconstruction check (product of inverse operations equals A)."""
    A = A.copy() % p
    e = A.shape[0]
    ops = []
    for col in range(e):
        piv = next(r for r in range(col, e) if A[r, col] % p)
        if piv != col:
            A[[col, piv]] = A[[piv, col]]; ops.append(("swap", col, piv))
        inv = pow(int(A[col, col]), -1, p)
        if inv != 1:
            A[col] = (A[col] * inv) % p; ops.append(("scale", col, inv))
        for r in range(e):
            if r != col and A[r, col] % p:
                c = (-int(A[r, col])) % p
                A[r] = (A[r] + c * A[col]) % p; ops.append(("add", r, col, c))
    assert np.array_equal(A % p, np.eye(e, dtype=int))
    # rebuild: A_original = E_1^{-1} ... E_k^{-1}
    R = np.eye(e, dtype=int)
    for op in reversed(ops):
        E = np.eye(e, dtype=int)
        if op[0] == "swap":
            E[[op[1], op[2]]] = E[[op[2], op[1]]]
        elif op[0] == "scale":
            E[op[1], op[1]] = pow(op[2], -1, p)
        else:
            E[op[1], op[2]] = (-op[3]) % p
        R = (E @ R) % p
    return len(ops), R


def check(C, m):
    F, pts, perm = frobenius_permutation(C, m)
    n = len(pts)
    Ns = {d: C.count(d) for d in range(1, m + 1) if m % d == 0 or True}
    is_perm = sorted(perm.tolist()) == list(range(n))
    pm = np.arange(n)
    for _ in range(m):
        pm = perm[pm]
    phi_m_id = bool(np.all(pm == np.arange(n)))
    # traces of powers by fixed points of perm^j
    traces, ok_tr = [], True
    pj = np.arange(n)
    for j in range(1, 2 * m + 1):
        pj = perm[pj]
        fix = int(np.sum(pj == np.arange(n)))
        want = Ns[math.gcd(j, m)]
        traces.append((j, fix, want)); ok_tr &= (fix == want)
    # orbits and Moebius formula
    seen = np.zeros(n, bool); orbit_len = {}
    for i in range(n):
        if not seen[i]:
            l, k = 0, i
            while not seen[k]:
                seen[k] = True; k = perm[k]; l += 1
            orbit_len[l] = orbit_len.get(l, 0) + 1
    ok_orb = all(orbit_len.get(l, 0) == sum(mobius(l // d) * Ns[d] for d in range(1, l + 1) if l % d == 0) // l
                 for l in range(1, m + 1) if m % l == 0)
    # dense unitary + spectrum (for n <= 900)
    spec_ok = None
    if n <= 900:
        P = np.zeros((n, n)); P[perm, np.arange(n)] = 1
        unit = np.linalg.norm(P.T @ P - np.eye(n)) < 1e-12
        ev = np.linalg.eigvals(P)
        roots_ok = np.max(np.abs(ev ** m - 1)) < 1e-6
        # predicted multiset
        pred = []
        for l, cnt in orbit_len.items():
            pred += [np.exp(2j * math.pi * k / l) for k in range(l)] * cnt
        a = np.sort_complex(np.round(ev, 6)); b = np.sort_complex(np.round(np.array(pred), 6))
        spec_ok = bool(unit and roots_ok and len(a) == len(b) and np.max(np.abs(a - b)) < 1e-5)
    Phi, lin_ok = frobenius_matrix(C.p, C.r * m, C.Q)
    Phim = np.linalg.matrix_power(Phi.astype(object), m) % C.p
    nops, R = elementary_factorisation(Phi, C.p)
    e = C.r * m
    return {"curve": C.name, "g": C.g, "Q": C.Q, "m": m, "n_m": n, "two_g": 2 * C.g,
            "bijection": is_perm, "phi_m_identity": phi_m_id, "traces_ok": ok_tr, "orbits_moebius_ok": ok_orb,
            "dense_unitary_spectrum_ok": spec_ok, "Fp_linear": lin_ok,
            "Phi_order_divides_m": bool(np.array_equal(Phim.astype(int), np.eye(e, dtype=int))),
            "elementary_ops": nops, "ops_bound_(rm)^2+rm": e * e + e,
            "factorisation_ok": bool(np.array_equal(R % C.p, Phi % C.p)),
            "orbit_lengths": orbit_len, "trace_table": traces}


if __name__ == "__main__":
    t0 = time.time()
    cases = [(Hyperelliptic([1, 1, 0, 1], 5), [1, 2, 3, 4]),
             (Hyperelliptic([0, -1, 0, 0, 0, 1], 3, name="y^2 = x^5 - x over F_3 ([CP] Prop. 6.2)"), [1, 2, 3, 4, 5, 6]),
             (Hyperelliptic([1, 0, 1, 1, 0, 0, 2], 5, name="y^2 = 2x^6 + x^3 + x^2 + 1 over F_5 (non-square lead)"), [1, 2, 3, 4]),
             (Hyperelliptic([1, 1, 0, 0, 0, 0, 0, 1], 3), [1, 2, 3, 4, 5]),
             (Hyperelliptic([1, 0, 1, 0, 0, 1], 3, r=2), [1, 2, 3]),
             (Hyperelliptic([2, 0, 1, 0, 0, 1], 7), [1, 2, 3])]
    rows = []
    for C, ms in cases:
        for m in ms:
            r = check(C, m)
            r["ok"] = all(r[k] for k in ("bijection", "phi_m_identity", "traces_ok", "orbits_moebius_ok", "Fp_linear",
                                         "Phi_order_divides_m", "factorisation_ok")) and r["dense_unitary_spectrum_ok"] in (True, None) \
                and r["elementary_ops"] <= r["ops_bound_(rm)^2+rm"]
            rows.append(r)
            print(f"{'OK ' if r['ok'] else 'FAIL'} m={m} n_m={r['n_m']:5d} 2g={r['two_g']} ops={r['elementary_ops']:3d}  {C.name}", flush=True)
    res = {"rows": rows, "all_ok": all(r["ok"] for r in rows), "cases": len(rows),
           "trace_identities_checked": sum(len(r["trace_table"]) for r in rows), "seconds": round(time.time() - t0, 1)}
    print("ALL OK:", res["all_ok"], res["cases"], "cases,", res["trace_identities_checked"], "trace identities,", res["seconds"], "s")
    save_json(OUT / "v5_prop3.json", res)
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
