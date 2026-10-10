# Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Validation of Lemma 1 (unitary symplectic realisation) and of Definition 2 (Frobenius oracle).

Lemma 1. Let U_C = Q^{-1/2} F on V_C = H^1(C) (x) C with the cup-product pairing omega
(alternating, nondegenerate, omega(Fx, Fy) = Q omega(x, y)), F semisimple with |eigenvalues| = sqrt Q.
  (a) There is a basis in which omega = J = [[0, I],[-I, 0]] and U_C = diag(B, conj B), |beta_i| = 1;
      hence U_C lies in USp(2g) for the Hermitian product making that basis orthonormal.
  (b) Tr(U_C^j) = tau_j(C) for every j.
  (c) Unitary matrices with the same characteristic polynomial are U(2g)-conjugate; elements of the
      maximal torus of USp(2g) with the same eigenvalue multiset are USp(2g)-conjugate (Weyl group of
      type C_g: permutations and inversions of the beta_i).
  Corollary. The eigenvalues +1 and -1 of U_C occur with even multiplicity.

Checks
  A  for every test curve: diag(B, conj B) is unitary, symplectic, and Tr U^j = tau_j (point counts)   -> (a),(b)
  B  constructive proof: for random 'Frobenius-like' similitudes F = P D P^{-1} (P complex symplectic,
     F^T J F = Q J), the algorithm of the proof returns S with S^T J S = J and S^{-1}(F/sqrt Q)S = diag(B, conj B);
     includes spectra with +-1 blocks of dimension 2, 4                                               -> (a)
  C  U(2g)-conjugacy of random realisations; explicit USp(2g) Weyl conjugators (signed permutations)   -> (c)
  D  parity corollary on complete Weil classes W_g(Q) (exact factorisation)                           -> corollary
  E  necessity of unitarity: a non-normal matrix with the same characteristic polynomial has the same
     power traces but is not unitary (||A|| > 1), so it is not an admissible realisation ([CP] Prop. 1.2)
Run: python3 -I v2_lemma1_unitary_realisation.py     Output: results/v2_lemma1.json
"""
import sys, time, math
from pathlib import Path
import numpy as np
from scipy.linalg import expm, null_space

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frobenius_core import (test_curves, random_curves, jmax_for, L_from_counts, usp_realisation_numeric,
                            factorized_spectrum_numeric, pair_spectrum, J, haar_unitary, weil_class, save_json)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def omega(x, y, g):
    return x @ J(g) @ y          # complex bilinear (not sesquilinear)


def symplectic_unitarising_basis(U, tol=1e-6):
    """Algorithm of the proof of Lemma 1(a). Returns S = [e_1..e_g | f_1..f_g] with S^T J S = J
    and S^{-1} U S diagonal with unimodular entries (B, conj B)."""
    n = U.shape[0]; g = n // 2
    lam = np.linalg.eigvals(U)
    # cluster eigenvalues
    clusters = []
    for z in lam:
        for c in clusters:
            if abs(c[0] - z) < 1e-4:
                c[1] += 1
                break
        else:
            clusters.append([z, 1])
    vals = [c[0] for c in clusters]
    # refine cluster centres and eigenspaces by null spaces
    def eigspace(beta, dim):
        s = np.linalg.svd(U - beta * np.eye(n), compute_uv=False)
        rtol = max(s[-dim] * 10, 1e-10) if dim <= n else 1e-10
        V = null_space(U - beta * np.eye(n), rcond=rtol / max(s[0], 1))
        if V.shape[1] != dim:
            # fall back to the dim smallest right singular vectors
            _, _, Vh = np.linalg.svd(U - beta * np.eye(n))
            V = Vh[-dim:].conj().T
        return V
    E_cols, F_cols = [], []
    done = [False] * len(clusters)
    for i, (beta, dim) in enumerate(clusters):
        if done[i]:
            continue
        if abs(beta - 1) < 1e-4 or abs(beta + 1) < 1e-4:
            done[i] = True
            W = eigspace(beta, dim)
            vecs = [W[:, k] for k in range(W.shape[1])]
            # symplectic Gram-Schmidt for the alternating form omega on W
            while vecs:
                e = vecs.pop(0)
                k = max(range(len(vecs)), key=lambda t: abs(omega(e, vecs[t], g)))
                f = vecs.pop(k)
                f = f / omega(e, f, g)
                E_cols.append(e); F_cols.append(f)
                vecs = [x - omega(x, f, g) * e + omega(x, e, g) * f for x in vecs]
            continue
        if np.imag(beta) < 0:
            continue
        # partner cluster conj(beta)
        k = min(range(len(clusters)), key=lambda t: abs(clusters[t][0] - np.conj(beta)) if not done[t] and t != i else 1e9)
        done[i] = done[k] = True
        Eb = eigspace(beta, dim)
        Fb = eigspace(clusters[k][0], clusters[k][1])
        G = Eb.T @ J(g) @ Fb
        Fd = Fb @ np.linalg.inv(G)
        for t in range(dim):
            E_cols.append(Eb[:, t]); F_cols.append(Fd[:, t])
    S = np.column_stack(E_cols + F_cols)
    return S


def random_complex_symplectic(g, rng, scale=0.6):
    H = rng.normal(size=(2 * g, 2 * g)) + 1j * rng.normal(size=(2 * g, 2 * g))
    H = (H + H.T) / 2 * scale
    return expm(J(g) @ H)


def check_constructive(B, Q, rng):
    g = len(B)
    A = math.sqrt(Q) * B
    Dp = np.diag(np.concatenate([A, Q / A]))
    P = random_complex_symplectic(g, rng)
    F = P @ Dp @ np.linalg.inv(P)
    simil = np.linalg.norm(F.T @ J(g) @ F - Q * J(g)) / Q
    U = F / math.sqrt(Q)
    S = symplectic_unitarising_basis(U)
    Dn = np.linalg.solve(S, U @ S)
    offdiag = np.linalg.norm(Dn - np.diag(np.diag(Dn)))
    d = np.diag(Dn)
    pairing = np.max(np.abs(d[:g] - np.conj(d[g:])))
    sympl = np.linalg.norm(S.T @ J(g) @ S - J(g))
    return {"g": g, "similitude_err": float(simil), "S_symplectic_err": float(sympl / max(1, np.linalg.norm(S) ** 2)),
            "offdiag_err": float(offdiag), "pairing_err": float(pairing),
            "unimodular_err": float(np.max(np.abs(np.abs(d) - 1))),
            "ok": bool(simil < 1e-8 and sympl < 1e-6 * max(1, np.linalg.norm(S) ** 2) and offdiag < 1e-6 and pairing < 1e-6)}


def weyl_conjugator(g, perm, inv):
    """USp(2g) element K with K diag(B, conj B) K^{-1} = diag(B', conj B'), B'_i = B_{perm[i]} or its inverse."""
    K = np.zeros((2 * g, 2 * g), dtype=complex)
    for i, src in enumerate(perm):
        if not inv[i]:
            K[i, src] = 1; K[g + i, g + src] = 1
        else:
            K[i, g + src] = 1; K[g + i, src] = -1
    return K


def parity_on_class(g, Q):
    import sympy as sp
    T = sp.Symbol("T")
    W = weil_class(g, Q)
    rq = math.isqrt(Q)
    bad = 0
    for L in W:
        P = sp.Poly(sum(c * T ** k for k, c in enumerate(L)), T)
        if rq * rq == Q:
            mults = []
            for fac in (1 - rq * T, 1 + rq * T):
                m, R = 0, P
                while True:
                    q_, r_ = sp.div(R, sp.Poly(fac, T))
                    if r_.is_zero:
                        m += 1; R = q_
                    else:
                        break
                mults.append(m)
            ok = all(m % 2 == 0 for m in mults)
        else:
            m, R = 0, P
            while True:
                q_, r_ = sp.div(R, sp.Poly(1 - Q * T ** 2, T))
                if r_.is_zero:
                    m += 1; R = q_
                else:
                    break
            ok = (m % 2 == 0)
        bad += (not ok)
    return {"g": g, "Q": Q, "class_size": len(W), "violations": bad}


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(20261011)
    res = {}
    # ---- A
    rowsA = []
    curves = test_curves() + random_curves(rng, 2, 5, 4) + random_curves(rng, 3, 3, 3)
    for C in curves:
        jm = min(jmax_for(C), 2 * C.g + 2)
        N = C.counts(jm)
        L, _ = L_from_counts(N, C.g, C.Q)
        U = usp_realisation_numeric(L, C.Q)
        g = C.g
        tau = [(C.Q ** j + 1 - N[j - 1]) / C.Q ** (j / 2) for j in range(1, jm + 1)]
        tr = [np.trace(np.linalg.matrix_power(U, j)) for j in range(1, jm + 1)]
        row = {"curve": C.name, "g": g, "Q": C.Q, "L": L,
               "unitary_err": float(np.linalg.norm(U.conj().T @ U - np.eye(2 * g))),
               "symplectic_err": float(np.linalg.norm(U.T @ J(g) @ U - J(g))),
               "trace_err": float(max(abs(a - b) for a, b in zip(tau, tr)))}
        # random U(2g) conjugate keeps traces (Definition 2: any element of the class)
        W = haar_unitary(2 * g, rng)
        U2 = W @ U @ W.conj().T
        row["conjugate_trace_err"] = float(max(abs(np.trace(np.linalg.matrix_power(U2, j)) - tau[j - 1]) for j in range(1, jm + 1)))
        row["ok"] = row["unitary_err"] < 1e-9 and row["symplectic_err"] < 1e-9 and row["trace_err"] < 1e-8 and row["conjugate_trace_err"] < 1e-8
        rowsA.append(row)
    res["A_curve_realisations"] = {"n": len(rowsA), "all_ok": all(r["ok"] for r in rowsA),
                                   "max_trace_err": max(r["trace_err"] for r in rowsA), "rows": rowsA}
    print("A:", res["A_curve_realisations"]["n"], res["A_curve_realisations"]["all_ok"], res["A_curve_realisations"]["max_trace_err"])
    # ---- B
    rowsB = []
    for C in curves:
        N = C.counts(C.g)
        L, _ = L_from_counts(N, C.g, C.Q)
        beta = factorized_spectrum_numeric(L, C.Q)
        try:
            B = pair_spectrum(beta, tol=1e-9)
        except ValueError:
            continue
        rowsB.append({"source": C.name, **check_constructive(B, C.Q, rng)})
    synth = [np.array([1, 1, np.exp(0.7j)]), np.array([-1, np.exp(1.1j)]), np.array([1, -1, -1, np.exp(2.0j)]),
             np.array([1, 1]), np.array([-1, -1, -1]), np.array([np.exp(0.3j), np.exp(0.3j), 1])]
    for B in synth:
        for Q in (3, 4, 7):
            rowsB.append({"source": f"synthetic B={np.round(B, 3).tolist()} Q={Q}", **check_constructive(B.astype(complex), Q, rng)})
    for _ in range(40):
        g = int(rng.integers(1, 5))
        B = np.exp(1j * rng.uniform(0, math.pi, g))
        rowsB.append({"source": "random", **check_constructive(B, int(rng.choice([2, 3, 5, 9, 11])), rng)})
    res["B_constructive_proof"] = {"n": len(rowsB), "all_ok": all(r["ok"] for r in rowsB),
                                   "max_offdiag_err": max(r["offdiag_err"] for r in rowsB), "rows": rowsB}
    print("B:", len(rowsB), res["B_constructive_proof"]["all_ok"], res["B_constructive_proof"]["max_offdiag_err"])
    # ---- C
    okC, n = True, 0
    for _ in range(50):
        g = int(rng.integers(1, 5))
        B = np.exp(1j * rng.uniform(0, math.pi, g))
        T0 = np.diag(np.concatenate([B, np.conj(B)]))
        # U(2g): two random conjugates are conjugate by an explicit unitary
        W1, W2 = haar_unitary(2 * g, rng), haar_unitary(2 * g, rng)
        A1, A2 = W1 @ T0 @ W1.conj().T, W2 @ T0 @ W2.conj().T
        X = W2 @ W1.conj().T
        okC &= np.linalg.norm(X @ A1 @ X.conj().T - A2) < 1e-9
        # USp(2g): Weyl conjugator
        perm = rng.permutation(g); inv = rng.integers(0, 2, g).astype(bool)
        Bp = np.array([B[perm[i]] if not inv[i] else np.conj(B[perm[i]]) for i in range(g)])
        T1 = np.diag(np.concatenate([Bp, np.conj(Bp)]))
        K = weyl_conjugator(g, perm, inv)
        okC &= np.linalg.norm(K.conj().T @ K - np.eye(2 * g)) < 1e-12
        okC &= np.linalg.norm(K.T @ J(g) @ K - J(g)) < 1e-12
        okC &= np.linalg.norm(K @ T0 @ np.linalg.inv(K) - T1) < 1e-12
        n += 1
    res["C_conjugacy"] = {"cases": n, "all_ok": bool(okC)}
    print("C:", n, okC)
    # ---- D
    rowsD = [parity_on_class(g, Q) for g, Qs in ((1, [2, 3, 4, 5, 7, 9, 16, 25]), (2, [2, 3, 4, 5, 9]), (3, [2, 3, 4])) for Q in Qs]
    res["D_parity"] = {"classes": len(rowsD), "polynomials": sum(r["class_size"] for r in rowsD),
                       "violations": sum(r["violations"] for r in rowsD), "rows": rowsD}
    print("D:", res["D_parity"]["polynomials"], "violations", res["D_parity"]["violations"])
    # ---- E
    B = np.exp(1j * np.array([0.4, 1.3]))
    D = np.diag(np.concatenate([B, np.conj(B)]))
    A = D + np.triu(np.ones((4, 4)), 1)
    res["E_non_unitary"] = {
        "same_power_traces": bool(all(abs(np.trace(np.linalg.matrix_power(A, j)) - np.trace(np.linalg.matrix_power(D, j))) < 1e-9 for j in range(1, 9))),
        "spectral_norm": float(np.linalg.norm(A, 2)),
        "is_unitary": bool(np.linalg.norm(A.conj().T @ A - np.eye(4)) < 1e-9)}
    print("E:", res["E_non_unitary"])
    from frobenius_core import is_weil_exact
    # Exact boundary, repeated-root, nonreal-h, malformed and out-of-bound cases.
    regression_cases=[([1,-4,4],4,True),([1,0,-8,0,16],4,True),
                      ([1,0,2],2,True),([1,0,0,0,4],2,True),
                      ([1,-5,4],4,False),([1,0,5,0,4],2,False),
                      ([1,1,3],2,False)]
    exact_ok=all(is_weil_exact(L,Q)==expected for L,Q,expected in regression_cases)
    res["F_exact_membership_regression"]={"cases":len(regression_cases),"ok":exact_ok}
    res["all_ok"] = bool(exact_ok and res["A_curve_realisations"]["all_ok"] and res["B_constructive_proof"]["all_ok"] and okC
                         and res["D_parity"]["violations"] == 0 and res["E_non_unitary"]["same_power_traces"]
                         and not res["E_non_unitary"]["is_unitary"])
    res["all_ok"] = bool(res["all_ok"] and exact_ok)
    res["seconds"] = round(time.time() - t0, 1)
    print("ALL OK:", res["all_ok"], res["seconds"], "s")
    save_json(OUT / "v2_lemma1.json", res)
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
