import os, sys, warnings, importlib.util, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PySpice.Spice.Netlist import Circuit
import mlrf
# mlrfのインポートは、PySpiceのインポートよりも後に行わなければならない。
orig = None
if len(sys.argv) > 1:
    spec = importlib.util.spec_from_file_location("mlrf_orig", sys.argv[1])
    orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)

z0 = 50.0
# ---- DUT: shunt (Rsh + C) to ground (2-port), exact state space
Cd, Rsh = 1e-12, 5.0
p = -2 / (Cd * (2 * Rsh + z0))
d21 = 2 * Rsh / (2 * Rsh + z0)
rr = 2 * (1 + p * Cd * Rsh) / (Cd * (2 * Rsh + z0))
A = np.eye(2) * p; B = np.eye(2); C = np.full((2, 2), rr); D = np.array([[d21 - 1, d21], [d21, d21 - 1]])
vs = lambda t: np.sin(2 * np.pi * 1e9 * t)
h, T = 1e-12, 3e-9

def ref(Rs, RL, CL, inl=lambda v: 0.0):
    def vn_alg(t, vc):
        f = lambda v: (vs(t) - v) / Rs - (v - vc) / Rsh - v / RL - inl(v)
        return brentq(f, -5, 5, xtol=1e-15)
    if CL > 0:
        rhs = lambda t, y: [(y[1] - y[0]) / Rsh / Cd,
                            ((vs(t) - y[1]) / Rs - (y[1] - y[0]) / Rsh - y[1] / RL - inl(y[1])) / CL]
        return lambda t: solve_ivp(rhs, (0, T), [0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-13, method="Radau").y[1]
    rhs = lambda t, y: [(vn_alg(t, y[0]) - y[0]) / Rsh / Cd]
    def out(t):
        vc = solve_ivp(rhs, (0, T), [0.0], t_eval=t, rtol=1e-10, atol=1e-13, method="Radau").y[0]
        return np.array([vn_alg(ti, ci) for ti, ci in zip(t, vc)])
    return out

ok = True
def check(name, err, lim):
    global ok
    flag = err < lim; ok &= flag
    print(f"[{'OK' if flag else 'NG'}] {name}: {err:.3e} (< {lim:g})")

# 1) Rs = z0, load RC, compare with original and analytic
terms = [("source", vs), ("load", dict(R=100.0, C=0.5e-12))]
t, V = mlrf.PortCircuitSimulator(A, B, C, D, z0, terms).run(T, h)
if orig is not None:
    to, Vo = orig.PortCircuitSimulator(A, B, C, D, z0, terms).run(T, h)
    check("new vs original (Rs=z0)", np.max(np.abs(V - Vo)), 1e-12)
check("vs analytic (Rs=z0, RC load)", np.max(np.abs(V[:, 1] - ref(50, 100, 0.5e-12)(t))), 1e-4)
# 2) Rs != z0
terms = [("source", dict(v=vs, R=10.0)), ("load", dict(R=100.0, C=0.5e-12))]
t, V = mlrf.PortCircuitSimulator(A, B, C, D, z0, terms).run(T, h)
check("vs analytic (Rs=10)", np.max(np.abs(V[:, 1] - ref(10, 100, 0.5e-12)(t))), 1e-4)
check("source port voltage = node (shunt)", np.max(np.abs(V[:, 0] - V[:, 1])), 1e-9)
# 3) C=0 load (algebraic row)
terms = [("source", dict(v=vs, R=10.0)), ("load", dict(R=100.0))]
t, V = mlrf.PortCircuitSimulator(A, B, C, D, z0, terms).run(T, h)
check("vs analytic (C_L=0)", np.max(np.abs(V[:, 1] - ref(10, 100, 0.0)(t))), 1e-4)
# 4) nonlinear diode, C=0, Rs != z0
Is, Vt = 1e-12, 0.02585
dio = lambda v: (Is * (np.exp(v / Vt) - 1), Is / Vt * np.exp(v / Vt))
terms = [("source", dict(v=vs, R=10.0)), ("load", dict(R=100.0, nl=dio))]
t, V = mlrf.PortCircuitSimulator(A, B, C, D, z0, terms).run(T, h)
r = ref(10, 100, 0.0, lambda v: dio(v)[0])(t)
check("nonlinear vs analytic (C_L=0)", np.max(np.abs(V[:, 1] - r)), 1e-3)
# 5) purely resistive DUT (n=0 states): series R between ports, diode load, C=0 -> static solve
Rser = 20.0
S11 = Rser / (Rser + 2 * z0); S21 = 2 * z0 / (Rser + 2 * z0)
Dr = np.array([[S11, S21], [S21, S11]])
src_dc = lambda t: 1.5
sim = mlrf.PortCircuitSimulator(np.zeros((0, 0)), np.zeros((0, 2)), np.zeros((2, 0)), Dr, z0,
                                [("source", dict(v=src_dc, R=0.0)), ("load", dict(nl=dio))])
t, V = sim.run(1e-10, 1e-11)
vd = brentq(lambda v: (1.5 - v) / Rser - dio(v)[0], 0, 1.5)
check("n=0 DUT, ideal source, diode (static)", abs(V[-1, 1] - vd), 1e-9)
# 6) DC op = steady state with DC source
terms = [("source", dict(v=lambda t: 1.0, R=10.0)), ("load", dict(R=100.0, C=0.5e-12, nl=dio))]
sim = mlrf.PortCircuitSimulator(A, B, C, D, z0, terms)
z = sim.dc_op()
t, V = sim.run(1e-10, 1e-12, init="dc")
vd = brentq(lambda v: (1.0 - v) / 10 - v / 100 - dio(v)[0], 0, 1)  # C open at DC
check("DC op matches analytic", abs(V[0, 1] - vd), 1e-9)
check("DC op is stationary", np.max(np.abs(V[:, 1] - vd)), 1e-9)

# 7) StaticNonlinearGP: optimize=False on fresh object, variance grows outside, sample extrapolation
rng = np.random.default_rng(0)
Vd = np.linspace(-0.5, 0.8, 12); Id = 1e-9 * (np.exp(Vd / 0.04) - 1) * (1 + 0.01 * rng.standard_normal(12))
g = mlrf.StaticNonlinearGP(1e-6, bounds=[(-0.5, 0.8)]).fit(Vd, Id, optimize=False)
print("optimize=False on fresh object: OK")
g.fit(Vd, Id)
print("z var at 1.2V (outside data) with / without β uncertainty:",
      g.z_var(np.array([1.2]))[0], g.z_var(np.array([1.2]), mean_uncertainty=False)[0])
s = g.sample_functions(np.linspace(-0.6, 0.9, 151), 5)
i_out, g_out = s[0](np.array([1.5, 2.0]))
check("sample extrapolation finite", float(not np.all(np.isfinite(i_out))), 0.5)
vv = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
ib, gb = s(vv)
ii = np.array([s[j](vv[j:j+1])[0][0] for j in range(5)])
check("batched sample eval == individual", np.max(np.abs(ib - ii)), 1e-20)
print("suggest (bounds):", g.suggest())
print(g.spice_bsource("B1", "a", "0", digits=4)[:200])

# 8) ensemble == individual runs
terms = [("source", vs), ("load", dict(R=100.0, C=0.5e-12, nl=g.current))]
sim = mlrf.PortCircuitSimulator(A, B, C, D, z0, terms)
t0 = time.perf_counter(); te, Ve = sim.run_ensemble(1e-9, 2e-12, [s], 5); t_e = time.perf_counter() - t0
t0 = time.perf_counter(); Vi = np.array([sim.run(1e-9, 2e-12, nl_funcs=[s[j]])[1] for j in range(5)]); t_i = time.perf_counter() - t0
check("ensemble vs individual", np.max(np.abs(Ve - Vi)), 1e-9)
print(f"ensemble {t_e:.2f}s vs individual {t_i:.2f}s")
print("ALL OK" if ok else "SOME FAILED")
