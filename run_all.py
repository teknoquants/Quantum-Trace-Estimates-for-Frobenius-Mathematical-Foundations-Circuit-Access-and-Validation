# Quantum trace estimates for Frobenius : Roger Cooperman and A.E. Petrunina V.V.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Run every validation script and write results/SUMMARY.json."""
import json, subprocess, sys, time, platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["v1_prop1_frobenius_power_trace.py", "v2_lemma1_unitary_realisation.py", "v3_thm1_trace_estimator.py",
           "v4_prop2_quantum_trace_estimate.py", "v5_prop3_permutation_unitary.py", "v6_prop4_permutation_cost.py",
           "v7_prop5_cohomological_oracle.py"]

if __name__ == "__main__":
    log = []
    for s in SCRIPTS:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-I", str(HERE / s)], capture_output=True, text=True)
        ok = r.returncode == 0
        (HERE / "results" / "logs").mkdir(parents=True, exist_ok=True)
        (HERE / "results" / "logs" / (s[:-3] + ".log")).write_text(r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""))
        log.append({"script": s, "passed": ok, "seconds": round(time.time() - t0, 1)})
        print(f"{'PASS' if ok else 'FAIL'}  {s}  ({log[-1]['seconds']} s)", flush=True)
        if not ok:
            print(r.stdout[-3000:], r.stderr[-3000:])
    import numpy, scipy, sympy
    import mpmath
    env = {"python": platform.python_version(), "implementation": platform.python_implementation(),
           "platform": platform.platform(), "machine": platform.machine(),
           "numpy": numpy.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__, "mpmath": mpmath.__version__}
    summary = {"all_passed": all(x["passed"] for x in log), "scripts": log, "environment": env}
    (HERE / "ENVIRONMENT.json").write_text(json.dumps(env, indent=1))
    (HERE / "results" / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print("ALL PASSED" if summary["all_passed"] else "SOME FAILED")
    sys.exit(0 if summary["all_passed"] else 1)

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
