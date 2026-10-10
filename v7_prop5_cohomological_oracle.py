# Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Validation of Proposition 5 (the cohomological oracle is equivalent to the L-polynomial) on finite instances.
Conjecture 1 (classical hardness) is NOT tested here: no finite computation can establish a complexity lower bound.

Proposition 5.
 (a) From L to oracle: given L_C(T), a circuit for controlled-V, V = diag(beta_1..beta_2g, 1, .., 1) on
     q = ceil(log2 2g) qubits, is produced classically in time poly(g, log Q, log 1/eps) from the Walsh
     expansion of the phases: at most 2^{q+1} - 1 phase rotations and (q-1) 2^{q+1} + 2 CNOT gates, i.e. O(g log g).
 (b) From oracle to L: from the classical description of ANY circuit O of size S on 1 + q + a qubits that is an
     eps-accurate Frobenius oracle in the sense of Definition 2 (for some phase gamma and all psi,
     ||O(psi (x) 0^a) - e^{i gamma} (Lambda_1(V) psi) (x) 0^a|| <= eps ||psi||) with eps <= eps_g / 6,
     eps_g = 1/(2 n Q^{g/2}), L_C(T) is computed classically in time poly(S, 2^{q+a}, g, log Q): simulate,
     project the ancillas, remove the phase using Tr(B_00), read Tr(V'^k) for k <= g and apply the congruence
     decoder (|Tr V'^k - Tr V^k| <= n k ||V' - V|| and ||V' - V|| <= 3 eps).
 (c) With eps <= eps_g/6, effective bounded-arity gates, polynomial bit descriptions, and a numerical
     error budget below k/8 for each scaled trace, O(log g)-ancilla oracles can be constructed classically in
     time poly(g, log Q) if and only if L_C(T) can be computed classically in time poly(g, log Q).

Checks
  A  synthesis (a) for every test curve: the gate list, simulated gate by gate, equals controlled-V up to a
     global phase (error < 1e-12); gate counts within the stated bounds; Hadamard-test traces correct
  B  dequantisation (b): oracles obfuscated by random conjugation W (random two-qubit gates) and by
     ancilla-based parity computation are simulated classically; L recovered exactly for every curve
  C  robustness under Definition 2: for gate-angle noise,
     a numerical upper bound ||B - gamma Lambda_1(V)|| is computed; every trial with eps <= eps_g/6
     yields the exact L; noise far above the bound produces failures (the bound is meaningful)
Run: python3 -I v7_prop5_cohomological_oracle.py      Output: results/v7_prop5.json
"""
import sys, time, math, itertools
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frobenius_core import (test_curves, random_curves, L_from_counts, factorized_spectrum_numeric, pair_spectrum,
                            congruence_decoder, haar_unitary, save_json)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------- gate-level simulator
def apply_gate(state, gate, nq):
    """state: array of shape (2,)*nq (+ batch). Qubit 0 is the most significant."""
    kind = gate[0]
    if kind == "cx":
        c, t = gate[1], gate[2]
        s = np.moveaxis(state, (c, t), (0, 1)).copy()
        s[1] = s[1][::-1].copy()
        return np.moveaxis(s, (0, 1), (c, t))
    if kind == "phase":                     # diag(e^{i phi}, e^{-i phi}) on qubit t
        t, phi = gate[1], gate[2]
        s = np.moveaxis(state, t, 0).copy()
        s[0] *= np.exp(1j * phi); s[1] *= np.exp(-1j * phi)
        return np.moveaxis(s, 0, t)
    if kind == "gphase":
        return state * np.exp(1j * gate[1])
    if kind == "u1":                        # arbitrary 2x2 unitary on qubit t
        t, U = gate[1], gate[2]
        s = np.moveaxis(state, t, 0)
        sh = s.shape
        s = (U @ s.reshape(2, -1)).reshape(sh)
        return np.moveaxis(s, 0, t)
    if kind == "u2":                        # arbitrary 4x4 unitary on qubits (i, k)
        i, k, U = gate[1], gate[2], gate[3]
        s = np.moveaxis(state, (i, k), (0, 1))
        sh = s.shape
        s = (U @ s.reshape(4, -1)).reshape(sh)
        return np.moveaxis(s, (0, 1), (i, k))
    raise ValueError(kind)


def circuit_unitary(gates, nq):
    dim = 2 ** nq
    st = np.eye(dim, dtype=complex).reshape((2,) * nq + (dim,))
    for g in gates:
        st = apply_gate(st, g, nq)
    return st.reshape(dim, dim)


# ----------------------------------------------------------------- (a) synthesis
def synthesise_controlled_V(beta, q, ancilla=None):
    """Gate list for controlled-diag(beta (+) 1) on qubits [0 = control, 1..q = target];
    parities are accumulated on the last qubit of S, or on 'ancilla' if given (returned clean)."""
    k = q + 1
    n = 2 ** q
    theta = np.zeros(2 ** k)
    for t in range(n):
        z = beta[t] if t < len(beta) else 1.0
        theta[(1 << q) | t] = np.angle(z)
    gates = []
    gp = theta.mean()
    gates.append(("gphase", gp))
    xs = np.arange(2 ** k)
    for S in range(1, 2 ** k):
        sign = np.array([(-1) ** bin(x & S).count("1") for x in xs])
        c = float(np.dot(theta, sign) / 2 ** k)
        if abs(c) < 1e-15:
            continue
        qubits = [i for i in range(k) if (S >> (k - 1 - i)) & 1]
        tgt = ancilla if ancilla is not None else qubits[-1]
        srcs = qubits if ancilla is not None else qubits[:-1]
        for s_ in srcs:
            gates.append(("cx", s_, tgt))
        gates.append(("phase", tgt, c))
        for s_ in reversed(srcs):
            gates.append(("cx", s_, tgt))
    return gates


def obfuscate(gates, q, rng, layers=3):
    """Conjugate the target register by a random circuit W of two-qubit gates: c-(W V W^dag) = (I(x)W) c-V (I(x)W^dag)."""
    W = []
    for _ in range(layers):
        if q == 1:
            W.append(("u1", 1, haar_unitary(2, rng)))
        for i in range(1, q):
            W.append(("u2", i, i + 1, haar_unitary(4, rng)))
    Wdag = [(("u1", g[1], g[2].conj().T) if g[0] == "u1" else ("u2", g[1], g[2], g[3].conj().T)) for g in reversed(W)]
    return Wdag + gates + W, None


# ----------------------------------------------------------------- (b) dequantisation
def extract_V(U, q, a):
    """From the unitary of a circuit on 1 + q + a qubits: project ancillas to |0>, split control blocks."""
    n = 2 ** q
    dim = 2 ** (1 + q + a)
    T = U.reshape((2, n, 2 ** a, 2, n, 2 ** a))[:, :, 0, :, :, 0]     # ancilla in/out = 0
    B00, B11 = T[0, :, 0, :], T[1, :, 1, :]
    ph = np.trace(B00) / abs(np.trace(B00))
    return B11 / ph, B00 / ph


def recover_L(Vp, g, Q):
    n = Vp.shape[0]
    s_hat = []
    P = np.eye(n, dtype=complex)
    for k in range(1, g + 1):
        P = P @ Vp
        s_hat.append(Q ** (k / 2) * float(np.real(np.trace(P) - (n - 2 * g))))
    return congruence_decoder(s_hat, g, Q)


def oracle_for(L, Q, g, use_ancilla, rng, obf=True):
    beta_all = factorized_spectrum_numeric(L, Q)
    B = pair_spectrum(beta_all, tol=1e-9)
    beta = np.concatenate([B, np.conj(B)])
    q = max(1, math.ceil(math.log2(2 * g)))
    a = 1 if use_ancilla else 0
    gates = synthesise_controlled_V(beta, q, ancilla=(1 + q) if use_ancilla else None)
    if obf:
        gates, _ = obfuscate(gates, q, rng)
    return gates, beta, q, a


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(20261014)
    curves = test_curves() + random_curves(rng, 2, 7, 4) + random_curves(rng, 3, 5, 3)
    data = []
    for C in curves:
        N = C.counts(C.g)
        L, _ = L_from_counts(N, C.g, C.Q)
        data.append((C, L))
    res = {}
    # ---- A synthesis
    rowsA = []
    for C, L in data:
        g, Q = C.g, C.Q
        gates, beta, q, a = oracle_for(L, Q, g, use_ancilla=False, rng=rng, obf=False)
        nq = 1 + q
        U = circuit_unitary(gates, nq)
        n = 2 ** q
        target = np.diag(np.concatenate([np.ones(n), np.concatenate([beta, np.ones(n - 2 * g)])]))
        err = float(np.linalg.norm(U - target))
        ncx = sum(1 for x in gates if x[0] == "cx"); nph = sum(1 for x in gates if x[0] == "phase")
        Vp, _ = extract_V(U, q, 0)
        N = C.counts(2 * g)
        tau = [(Q ** j + 1 - N[j - 1]) / Q ** (j / 2) for j in range(1, 2 * g + 1)]
        tr_err = max(abs(np.trace(np.linalg.matrix_power(Vp, j)) - (tau[j - 1] + n - 2 * g)) for j in range(1, 2 * g + 1))
        rowsA.append({"curve": C.name, "g": g, "Q": Q, "q": q, "unitary_err": err, "cx": ncx, "phase_gates": nph,
                      "cx_bound": (q - 1) * 2 ** (q + 1) + 2, "phase_bound": 2 ** (q + 1) - 1, "trace_err": float(tr_err),
                      "ok": err < 1e-12 and ncx <= (q - 1) * 2 ** (q + 1) + 2 and nph <= 2 ** (q + 1) - 1 and tr_err < 1e-9})
    res["A_synthesis"] = {"cases": len(rowsA), "all_ok": all(r["ok"] for r in rowsA), "rows": rowsA}
    print("A:", len(rowsA), res["A_synthesis"]["all_ok"], "max unitary err", max(r["unitary_err"] for r in rowsA))
    # ---- B dequantisation
    rowsB = []
    for C, L in data:
        g, Q = C.g, C.Q
        for use_anc in (False, True):
            gates, beta, q, a = oracle_for(L, Q, g, use_ancilla=use_anc, rng=rng, obf=True)
            nq = 1 + q + a
            t1 = time.time()
            U = circuit_unitary(gates, nq)
            Vp, B00 = extract_V(U, q, a)
            Lr = recover_L(Vp, g, Q)
            rowsB.append({"curve": C.name, "g": g, "Q": Q, "ancillas": a, "gates": len(gates),
                          "recovered": Lr == L, "classical_seconds": round(time.time() - t1, 4)})
    res["B_dequantisation"] = {"cases": len(rowsB), "all_ok": all(r["recovered"] for r in rowsB), "rows": rowsB}
    print("B:", len(rowsB), res["B_dequantisation"]["all_ok"])
    # ---- C robustness
    rowsC = []
    for C, L in data:
        g, Q = C.g, C.Q
        q = max(1, math.ceil(math.log2(2 * g))); n = 2 ** q
        eps_g = 1 / (2 * n * Q ** (g / 2))
        gates0, beta, q, a = oracle_for(L, Q, g, use_ancilla=False, rng=rng, obf=False)
        U0 = circuit_unitary(gates0, 1 + q)
        V0, _ = extract_V(U0, q, 0)
        nphase = sum(1 for x in gates0 if x[0] == "phase")
        for factor in (0.15, 30.0):
            trials, devs, n_below, ok_below, ok_all = 60, [], 0, 0, 0
            for _ in range(trials):
                # random angle noise, scaled so that the operator deviation is about factor * eps_g
                sigma = factor * eps_g / max(1, math.sqrt(nphase))
                noisy = [(x[0], x[1], x[2] + float(rng.normal(0, sigma))) if x[0] == "phase" else x for x in gates0]
                Un = circuit_unitary(noisy, 1 + q)
                Vp, _ = extract_V(Un, q, 0)
                Lam = np.zeros((2 * n, 2 * n), dtype=complex); Lam[:n, :n] = np.eye(n); Lam[n:, n:] = V0
                gam = np.trace(Lam.conj().T @ Un); gam /= abs(gam)
                dev = float(np.linalg.norm(Un - gam * Lam, 2))          # upper bound from Frobenius-optimal phase; not operator-norm minimization
                devs.append(dev)
                rec = recover_L(Vp, g, Q) == L
                ok_all += rec
                if dev <= eps_g / 6:
                    n_below += 1; ok_below += rec
            rowsC.append({"curve": C.name, "g": g, "Q": Q, "eps_g": eps_g, "noise_factor": factor,
                          "median_deviation_over_eps_g": float(np.median(devs) / eps_g),
                          "trials_below_bound": n_below, "recovered_below_bound": ok_below,
                          "recovery_rate_all": ok_all / trials})
    tb = sum(r["trials_below_bound"] for r in rowsC); rb = sum(r["recovered_below_bound"] for r in rowsC)
    above = [r for r in rowsC if r["noise_factor"] == 30.0]
    res["C_robustness"] = {"trials_with_upper_bound_at_most_eps_g_over_6": tb, "recovered_among_them": rb,
                           "far_above_bound_mean_recovery": float(np.mean([r["recovery_rate_all"] for r in above])),
                           "rows": rowsC}
    res["C_robustness"]["ok"] = tb > 0 and rb == tb and res["C_robustness"]["far_above_bound_mean_recovery"] < 1.0
    print("C:", {k: v for k, v in res["C_robustness"].items() if k != "rows"})
    # Regression: trace alignment is not the operator-norm optimal phase.
    u = np.array([1,1,1,1j]); gam = u.sum()/abs(u.sum())
    trace_phase_bound = float(np.max(np.abs(u-gam)))
    midpoint_bound = float(np.max(np.abs(u-np.exp(1j*math.pi/4))))
    phase_ok = midpoint_bound < trace_phase_bound - 0.1
    res["D_phase_regression"] = {"trace_phase_upper_bound":trace_phase_bound,
                                 "midpoint_upper_bound":midpoint_bound,"ok":phase_ok}
    res["all_ok"] = bool(phase_ok and res["A_synthesis"]["all_ok"] and res["B_dequantisation"]["all_ok"] and res["C_robustness"]["ok"])
    res["seconds"] = round(time.time() - t0, 1)
    print("ALL OK:", res["all_ok"], res["seconds"], "s")
    save_json(OUT / "v7_prop5.json", res)
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
