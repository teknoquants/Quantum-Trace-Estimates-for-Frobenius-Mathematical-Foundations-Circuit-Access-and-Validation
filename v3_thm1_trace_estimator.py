# Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Validation of Theorem 1 (the estimator: from oracle calls to a number).

Theorem 1. Let V = U~ (+) I_{n-2g} on C^n (n = 2^q >= 2g), U~ in the realisation class R(C) of Lemma 1.
 (i)   The real Hadamard test A_j (Bell state |Phi_n>, controlled (I (x) V^j)) accepts with probability
       a_j = 1/2 + (tau_j(C) + n - 2g)/(2n);  the imaginary test accepts with probability exactly 1/2,
       because tau_j(C) is real (Proposition 1), so only the real test is needed.
 (ii)  Canonical amplitude estimation with M >= 4 pi n / delta register points returns a~ with
       p^ = 2n(a~ - 1/2) - (n - 2g) satisfying Pr(|p^ - tau_j| <= delta) >= 8/pi^2.
 (iii) The median of r >= ln(1/eta) / (2 (8/pi^2 - 1/2)^2) independent runs is within delta with
       probability >= 1 - eta.
 (iv)  Oracle calls: r (2M - 1) j <= 2 r j M controlled applications of V or V^dagger.

Checks
  A  statevector Hadamard test on realisations of the test curves (random conjugation W, two paddings n)
     reproduces a_j (real) and 1/2 (imaginary) to 1e-12                                              -> (i)
  B  full statevector simulation of amplitude estimation (phase estimation of the Grover iterate on the
     actual Hadamard-test state) equals the analytic law used below, to 1e-10                         -> (ii)
  C  the Brassard-Hoyer-Mosca-Tapp bound holds with probability >= 8/pi^2 on a grid of 2001 amplitudes
     and M = 8..2048 (exact probabilities, no sampling)                                               -> (ii)
  D  with M = M(n, delta) the exact single-run success Pr(|p^ - tau_j| <= delta) >= 8/pi^2 for every test
     curve, j <= 2g and delta in {0.5, 0.2, 0.1, 0.05, delta_cong}                                      -> (ii)
  E  median: exact failure probability <= eta for eta in {0.1, 0.01, 0.001}; end-to-end sampling          -> (iii)
  F  separately instrumented small full circuits count actual O1/O2 applications; large sampling uses bookkeeping                                    -> (iv)
  G  approximate access (Definition 2, Lemma 2, Theorem 1'): replacing V by an eps-accurate V' (random
     perturbation, ||V' - V|| = eps) changes the exact output law of one amplitude-estimation run by at most
     K eps in total variation, K = j (2M - 1); and the success probability stays >= 8/pi^2 - K eps        -> Theorem 1'
     (exact statevector simulation, no sampling; a global phase on the oracle is shown to be irrelevant)
Run: python3 -I v3_thm1_trace_estimator.py      Output: results/v3_thm1.json
"""
import sys, time, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frobenius_core import (test_curves, L_from_counts, usp_realisation_numeric, haar_unitary, padded,
                            hadamard_state, good_projector_diag, ae_distribution_statevector,
                            ae_distribution_analytic, ae_estimate_values, ae_error_bound,
                            median_failure_probability, runs_for_confidence, M_for_precision,
                            delta_cong, save_json)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def imag_state(X):
    n = X.shape[0]
    Phi = np.eye(n, dtype=complex).reshape(n * n) / math.sqrt(n)
    phiX = -1j * (np.kron(np.eye(n), X) @ Phi)
    return np.concatenate([(Phi + phiX) / 2, (Phi - phiX) / 2])


def curve_data():
    out = []
    for C in test_curves():
        N = C.counts(2 * C.g)
        L, _ = L_from_counts(N, C.g, C.Q)
        tau = [(C.Q ** j + 1 - N[j - 1]) / C.Q ** (j / 2) for j in range(1, 2 * C.g + 1)]
        out.append((C, L, tau))
    return out


class AnalyticSampler:
    """Analytic sampling with formula-based cost bookkeeping, not circuit instrumentation."""
    def __init__(self):
        self.calls = 0

    def run_ae(self, a, M, j, rng):
        # state preparation A uses 1 call of c-V^j; each Grover iterate uses A and A^dagger (2 calls of c-V^j)
        # phase estimation applies the iterate M - 1 times in total (controlled powers 1, 2, 4, ..., M/2)
        self.calls += j * (1 + 2 * (M - 1))
        P = ae_distribution_analytic(a, M)
        y = rng.choice(M, p=P)
        return math.sin(math.pi * y / M) ** 2


def gate_on(state, gate, axes, nqubits):
    """Apply an actual small gate to specified tensor axes of the full state."""
    rest = [i for i in range(nqubits) if i not in axes]
    order = list(axes)+rest
    block = state.reshape([2]*nqubits).transpose(order).reshape(2**len(axes),-1)
    block = gate @ block
    return block.reshape([2]*nqubits).transpose(np.argsort(order)).reshape(-1)


def explicit_ae(V, j, M, oracle_matrices=None):
    """Small full-circuit AE test, n=2, one private ancilla, counted oracle applications.

    O1 and O2 are separately supplied; their whole-call phases and arbitrary
    ancilla leakage are preserved. No extra control is added to either oracle.
    """
    t = int(math.log2(M)); assert M == 2**t and V.shape == (2,2)
    nq = t+4; c,ref,target,anc = t,t+1,t+2,t+3
    H = np.array([[1,1],[1,-1]],complex)/math.sqrt(2)
    CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)
    I = np.eye(2)
    def controlled(U):
        z=np.zeros_like(U); return np.block([[np.eye(len(U)),z],[z,U]])
    ideals = {1:np.kron(controlled(V),I),2:np.kron(controlled(controlled(V)),I)}
    Os = ideals if oracle_matrices is None else oracle_matrices
    calls = {"O1":0,"O1_dagger":0,"O2":0,"O2_dagger":0}
    def A(state, phase_control=None, inverse=False):
        seq=[(H,[ref]),(CX,[ref,target]),(H,[c])]+[(None,None)]*j+[(H,[c])]
        if inverse: seq=seq[::-1]
        for gate,axes in seq:
            if gate is None:
                k=1 if phase_control is None else 2
                axes=([c,target,anc] if k==1 else [phase_control,c,target,anc])
                gate=Os[k].conj().T if inverse else Os[k]
                calls['O'+str(k)+('_dagger' if inverse else '')]+=1
            else:
                gate=gate.conj().T if inverse else gate
                if phase_control is not None:
                    gate=controlled(gate); axes=[phase_control]+axes
            state=gate_on(state,gate,axes,nq)
        return state
    state=np.zeros(2**nq,complex); state[0]=1
    state=A(state)
    for k in range(t): state=gate_on(state,H,[k],nq)
    indices=np.arange(2**nq)
    def bit(axis): return (indices >> (nq-1-axis)) & 1
    for k in range(t):
        selected=bit(k)==1
        for _ in range(2**(t-1-k)):
            # controlled Q = controlled(-A S0 A^dagger Sgood)
            state[selected & (bit(c)==0)] *= -1
            state=A(state,k,True)
            state[selected & (((indices // 2) % 8)==0)] *= -1  # reflect accessible work only; private ancilla untouched
            state=A(state,k,False)
            state[selected] *= -1
    state=np.fft.fft(state.reshape(M,-1),axis=0)/math.sqrt(M)
    return np.sum(abs(state)**2,axis=1),calls,ideals


def circuit_regressions(rng):
    from scipy.linalg import expm
    V=np.diag(np.exp(1j*np.array([0.7,-0.7])))
    rows=[]; contract_rows=[]
    for j in (1,2,3):
        for M in (8,16):
            P,calls,ideals=explicit_ae(V,j,M)
            a=0.5+float(np.trace(np.linalg.matrix_power(V,j)).real)/4
            err=float(np.max(abs(P-ae_distribution_analytic(a,M))))
            total=sum(calls.values()); K=j*(2*M-1)
            rows.append({"j":j,"M":M,"calls":calls,"total":total,"predicted":K,
                         "law_error":err,"ok":total==K and err<1e-10})
            for strength in (0.001,0.004):
                Os={}; eps=0.; inverse_error=0.; phases={1:0.37,2:-0.61}
                for k,U in ideals.items():
                    d=len(U); Z=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
                    Z=(Z+Z.conj().T)/2; Z/=np.linalg.norm(Z,2)
                    O=np.exp(1j*phases[k])*U@expm(1j*strength*Z)
                    E=np.eye(d)[:,::2]  # ancilla=0 columns, arbitrary controls/target
                    ek=float(np.linalg.norm((O-np.exp(1j*phases[k])*U)@E,2))
                    inv=float(np.linalg.norm((O.conj().T-np.exp(-1j*phases[k])*U.conj().T)@E,2))
                    assert inv <= ek+1e-12
                    Os[k]=O; eps=max(eps,ek); inverse_error=max(inverse_error,inv)
                Pn,cn,_=explicit_ae(V,j,M,Os)
                tv=float(abs(P-Pn).sum()/2)
                contract_rows.append({"j":j,"M":M,"epsilon":eps,"inverse_error":inverse_error,
                                      "tv":tv,"K_epsilon":K*eps,
                                      "ok":tv<=K*eps+1e-10 and sum(cn.values())==K})
    return ({"cases":len(rows),"rows":rows,"ok":all(r["ok"] for r in rows)},
            {"cases":len(contract_rows),"rows":contract_rows,"ok":all(r["ok"] for r in contract_rows)})


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(20261012)
    data = curve_data()
    res = {}
    # ---- A
    maxA, maxI, nA = 0.0, 0.0, 0
    for C, L, tau in data:
        g = C.g
        U = usp_realisation_numeric(L, C.Q)
        W = haar_unitary(2 * g, rng)
        Ut = W @ U @ W.conj().T
        n0 = 1 << math.ceil(math.log2(2 * g))
        for n in (n0, 2 * n0):
            V = padded(Ut, n)
            good = good_projector_diag(n)
            for j in range(1, 2 * g + 1):
                X = np.linalg.matrix_power(V, j)
                psi = hadamard_state(X)
                a_sv = float(np.sum(np.abs(psi) ** 2 * good))
                a_th = 0.5 + (tau[j - 1] + n - 2 * g) / (2 * n)
                psiI = imag_state(X)
                aI = float(np.sum(np.abs(psiI) ** 2 * good))
                maxA = max(maxA, abs(a_sv - a_th)); maxI = max(maxI, abs(aI - 0.5)); nA += 1
    res["A_hadamard"] = {"cases": nA, "max_err_real": maxA, "max_dev_imag_from_half": maxI,
                         "ok": maxA < 1e-12 and maxI < 1e-12}
    print("A:", res["A_hadamard"])
    # ---- B
    maxB, nB = 0.0, 0
    for C, L, tau in data[:8]:
        g = C.g
        U = usp_realisation_numeric(L, C.Q)
        n = 1 << math.ceil(math.log2(2 * g))
        V = padded(U, n)
        for j in (1, 2):
            psi = hadamard_state(np.linalg.matrix_power(V, j))
            good = good_projector_diag(n)
            a = float(np.sum(np.abs(psi) ** 2 * good))
            for M in (8, 32, 128):
                P_sv = ae_distribution_statevector(psi, good, M)
                P_an = ae_distribution_analytic(a, M)
                maxB = max(maxB, float(np.max(np.abs(P_sv - P_an)))); nB += 1
    res["B_ae_statevector_vs_analytic"] = {"cases": nB, "max_abs_diff": maxB, "ok": maxB < 1e-10}
    print("B:", res["B_ae_statevector_vs_analytic"])
    # ---- C
    minC = 1.0
    for M in (8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        vals = ae_estimate_values(M)
        for a in np.linspace(0, 1, 2001):
            P = ae_distribution_analytic(a, M)
            ok = np.abs(vals - a) <= ae_error_bound(a, M) + 1e-15
            minC = min(minC, float(P[ok].sum()))
    res["C_bhmt_bound"] = {"min_success_probability": minC, "threshold_8_over_pi2": 8 / math.pi ** 2,
                           "ok": minC >= 8 / math.pi ** 2 - 1e-12}
    print("C:", res["C_bhmt_bound"])
    # ---- D
    rowsD, minD = [], 1.0
    for C, L, tau in data:
        g = C.g
        n = 1 << math.ceil(math.log2(2 * g))
        for delta in (0.5, 0.2, 0.1, 0.05, delta_cong(g, C.Q)):
            M = M_for_precision(n, delta)
            vals = ae_estimate_values(M)
            for j in range(1, 2 * g + 1):
                a = 0.5 + (tau[j - 1] + n - 2 * g) / (2 * n)
                P = ae_distribution_analytic(a, M)
                phat = 2 * n * (vals - 0.5) - (n - 2 * g)
                succ = float(P[np.abs(phat - tau[j - 1]) <= delta].sum())
                minD = min(minD, succ)
                rowsD.append({"curve": C.name, "j": j, "delta": delta, "M": M, "success": succ})
    res["D_precision_guarantee"] = {"cases": len(rowsD), "min_success": minD, "ok": minD >= 8 / math.pi ** 2 - 1e-12}
    print("D:", {k: v for k, v in res["D_precision_guarantee"].items()})
    # ---- E
    p_fail = 1 - 8 / math.pi ** 2
    rowsE = []
    for eta in (0.1, 0.01, 0.001):
        r = runs_for_confidence(eta)
        rowsE.append({"eta": eta, "r": r, "exact_median_failure": median_failure_probability(p_fail, r),
                      "ok": median_failure_probability(p_fail, r) <= eta})
    # end-to-end sampling for the [CP] curve, j = 1, 2, delta = delta_cong, eta = 0.01
    C, L, tau = next(d for d in data if "Prop. 6.2" in d[0].name)
    g, n = C.g, 4
    delta, eta = delta_cong(g, C.Q), 0.01
    M, r = M_for_precision(n, delta), runs_for_confidence(eta)
    orc = AnalyticSampler()
    trials, hits = 4000, 0
    for _ in range(trials):
        j = 2
        a = 0.5 + (tau[j - 1] + n - 2 * g) / (2 * n)
        est = [2 * n * (orc.run_ae(a, M, j, rng) - 0.5) - (n - 2 * g) for _ in range(r)]
        hits += abs(float(np.median(est)) - tau[j - 1]) <= delta
    res["E_median"] = {"rows": rowsE, "end_to_end": {"curve": C.name, "j": 2, "delta": delta, "eta": eta, "M": M, "r": r,
                                                     "trials": trials, "coverage": hits / trials},
                       "ok": all(x["ok"] for x in rowsE) and hits / trials >= 1 - eta}
    print("E:", res["E_median"]["rows"], res["E_median"]["end_to_end"])
    # ---- F
    expected = trials * r * 2 * (1 + 2 * (M - 1))
    res["F_formula_bookkeeping"] = {"formula_total": orc.calls, "formula_r(2M-1)j_times_trials": expected,
                             "bound_2rjM_times_trials": trials * 2 * r * 2 * M, "ok": orc.calls == expected <= trials * 2 * r * 2 * M}
    print("F bookkeeping:", res["F_formula_bookkeeping"])
    # ---- G approximate access
    from scipy.linalg import expm
    rowsG, okG = [], True
    for C, L, tau in data[:9]:
        g = C.g
        U = usp_realisation_numeric(L, C.Q)
        n = 1 << math.ceil(math.log2(2 * g))
        V = padded(U, n)
        good = good_projector_diag(n)
        for eps in (1e-4, 1e-3, 1e-2):
            H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); H = (H + H.conj().T) / 2
            H /= np.linalg.norm(H, 2)
            Vp = V @ expm(1j * eps * H)                              # Lambda_1(V') is eps-close to Lambda_1(V)
            e_act = float(np.linalg.norm(Vp - V, 2))
            for j in (1, 2):
                for M in (16, 64):
                    K = j * (2 * M - 1)
                    psi = hadamard_state(np.linalg.matrix_power(V, j))
                    psip = hadamard_state(np.linalg.matrix_power(Vp, j))
                    P = ae_distribution_statevector(psi, good, M)
                    Pp = ae_distribution_statevector(psip, good, M)
                    tv = 0.5 * float(np.abs(P - Pp).sum())
                    a = float(np.sum(np.abs(psi) ** 2 * good))
                    vals = ae_estimate_values(M)
                    okmask = np.abs(vals - a) <= ae_error_bound(a, M) + 1e-15
                    succ_noisy = float(Pp[okmask].sum())
                    ok = tv <= K * e_act + 1e-12 and succ_noisy >= 8 / math.pi ** 2 - K * e_act - 1e-12
                    okG &= ok
                    rowsG.append({"curve": C.name, "eps": e_act, "j": j, "M": M, "K": K, "tv": tv,
                                  "K_eps": K * e_act, "success_noisy": succ_noisy, "ok": ok})
    # phase placement: e^{i gamma} Lambda_1(V) (allowed by the contract) leaves the law unchanged, because the phase is
    # global for the Hadamard test and cancels between A and A^dagger inside each Grover iterate; Lambda_1(e^{i gamma} V)
    # (phase inside the control) is NOT an eps-accurate oracle for small eps and changes the law.
    C, L, tau = data[6]; U = usp_realisation_numeric(L, C.Q); n = 4; V = padded(U, n); good = good_projector_diag(n)
    psi0 = hadamard_state(np.linalg.matrix_power(V, 2))
    P0 = ae_distribution_statevector(psi0, good, 64)
    P_global = ae_distribution_statevector(np.exp(2.1j) * psi0, good, 64)          # e^{i gamma} on the whole controlled call
    P_inside = ae_distribution_statevector(hadamard_state(np.linalg.matrix_power(V * np.exp(1.05j), 2)), good, 64)
    res["G_approximate_access"] = {"cases": len(rowsG), "max_tv_over_K_eps": max(r["tv"] / r["K_eps"] for r in rowsG),
                                   "phase_on_controlled_call_max_diff": float(np.max(np.abs(P0 - P_global))),
                                   "phase_inside_control_tv": 0.5 * float(np.abs(P0 - P_inside).sum()),
                                   "rows": rowsG,
                                   "ok": bool(okG and np.max(np.abs(P0 - P_global)) < 1e-12 and 0.5 * np.abs(P0 - P_inside).sum() > 0.1)}
    print("G:", {k: v for k, v in res["G_approximate_access"].items() if k != "rows"})
    res["F_instrumented_circuit"],res["H_clean_ancilla_contract"] = circuit_regressions(rng)
    print("F actual circuit:",res["F_instrumented_circuit"]["ok"])
    print("H independent O1/O2 with leakage:",res["H_clean_ancilla_contract"]["ok"])
    # cost table for the paper (delta = delta_cong, eta = 0.01)
    table = []
    for g, Q in ((1, 3), (1, 25), (2, 3), (2, 7), (2, 25), (3, 3), (3, 5)):
        n = 1 << math.ceil(math.log2(2 * g))
        d = delta_cong(g, Q)
        M, r = M_for_precision(n, d), runs_for_confidence(0.01)
        calls = sum(r * (2 * M - 1) * j for j in range(1, g + 1))
        table.append({"g": g, "Q": Q, "n": n, "delta_cong": d, "M": M, "r": r, "calls_all_j<=g": calls})
    res["cost_table"] = table
    res["all_ok"] = all(res[k]["ok"] for k in ("A_hadamard", "B_ae_statevector_vs_analytic", "C_bhmt_bound",
                                                 "D_precision_guarantee", "E_median", "F_formula_bookkeeping", "F_instrumented_circuit", "G_approximate_access", "H_clean_ancilla_contract"))
    res["seconds"] = round(time.time() - t0, 1)
    print("ALL OK:", res["all_ok"], res["seconds"], "s")
    save_json(OUT / "v3_thm1.json", res)
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
