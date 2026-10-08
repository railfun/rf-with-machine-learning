"""write_ltspice の検証: 書き出したモデルを LTspice (バッチモード) で解き、Python 側と比較する

    python verify_ltspice.py            (LTspice の場所は環境変数 LTSPICE で変更できる)
1) AC 解析で S パラメータを求め、CompactModel.response と比較
2) ダイオードモデルの DC スイープを StaticNonlinearGP.current と比較
3) フィルタ → ダイオードリミッタの過渡解析を PortCircuitSimulator と比較
"""
import sys, os, subprocess, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, skrf as rf
from scipy.special import lambertw
from skrf.media import DefinedGammaZ0
import mlrf
LT = os.environ.get("LTSPICE", os.path.expandvars(r"%LOCALAPPDATA%\Programs\ADI\LTspice\LTspice.exe"))
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltspice_check")
os.makedirs(HERE, exist_ok=True)

def read_raw(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    nv = int(next(l for l in lines if l.startswith("No. Variables:")).split(":")[1])
    iv = lines.index("Variables:"); names = [lines[iv + 1 + k].split("\t")[2] for k in range(nv)]
    vals = lines[lines.index("Values:") + 1:]
    cplx = any("complex" in l for l in lines[:iv])
    data, row = [], []
    for l in vals:
        if not l.strip(): continue
        tok = l.split("\t")[-1].strip()
        x = complex(*map(float, tok.split(","))) if cplx else float(tok)
        row.append(x)
        if len(row) == nv: data.append(row); row = []
    d = np.array(data)
    return {n.lower(): d[:, k] for k, n in enumerate(names)}

def run(net, tag):
    p = os.path.join(HERE, tag + ".cir")
    open(p, "w").write(net)
    subprocess.run([LT, "-b", "-ascii", p], check=True, timeout=300)
    return read_raw(p[:-4] + ".raw")

# --- models (same as demo, smaller)
f = np.linspace(0.5e9, 5e9, 181)
med = DefinedGammaZ0(frequency=rf.Frequency.from_f(f, unit="hz"), z0=50)
true = lambda th: (med.shunt_inductor(th[0]*1e-9) ** med.shunt_capacitor(2e-12) ** med.capacitor(th[1]*1e-12)
                   ** med.shunt_inductor(th[0]*1e-9) ** med.shunt_capacitor(2e-12)).s
ths = np.array([[a, b] for a in np.linspace(1.5, 3, 4) for b in np.linspace(0.3, 1.2, 4)])
gp = mlrf.ParametricRationalGP([(1.5, 3.0), (0.3, 1.2)], names=["L_nH", "Cc_pF"]).fit(ths, f, np.array([true(t) for t in ths]))
TH = [2.2, 0.7]
cm = gp.compact(TH)
IS, N_ID, RS, VT = 5e-9, 1.05, 10.0, 0.025865
dtrue = lambda v: np.real(N_ID*VT/RS*lambertw(IS*RS/(N_ID*VT)*np.exp((v+IS*RS)/(N_ID*VT)))) - IS
V = np.r_[np.linspace(-1, 0.9, 12), 0.6, 0.7, 0.75, 0.8]
dg = mlrf.StaticNonlinearGP(1e-6).fit(V, dtrue(V))
lib = mlrf.write_ltspice(os.path.join(HERE, "models"), [cm, dg], names=["filt", "dml"])
print("lib:", lib, sum(1 for _ in open(lib)), "lines")

# --- 1) AC: S parameters (drive one port with 1 V AC through 50 ohm, other port 50 ohm)
ff = np.linspace(0.2e9, 6e9, 59)
S_lt = np.zeros((len(ff), 2, 2), complex)
for k in (1, 2):
    o = 3 - k
    net = f"""* S extraction port {k}
.include {lib}
X1 p1 p2 filt
V1 s 0 AC 1
R1 s p{k} 50
R2 p{o} 0 50
.ac list {' '.join(f'{x:.6g}' for x in ff)}
.end
"""
    r = run(net, f"ac{k}")
    S_lt[:, k-1, k-1] = 2 * r[f"v(p{k})"] - 1
    S_lt[:, o-1, k-1] = 2 * r[f"v(p{o})"]
err = np.max(np.abs(S_lt - cm.response(ff)))
print(f"AC: max|S_LTspice - S_compact| = {err:.2e}")

# --- 2) DC sweep of diode model
net = f"""* diode DC
.include {lib}
V1 a 0 0
X1 a 0 dml
.dc V1 -1.2 1.0 0.01
.end
"""
r = run(net, "dc")
vv = np.real(r["v(a)"]); i_lt = -np.real(r["i(v1)"])
i_py = dg.current(vv)[0]
m = np.abs(i_py) > 1e-9
print(f"DC: max rel err |I_LT - I_py|/|I_py| (|I|>1nA) = {np.max(np.abs(i_lt[m]-i_py[m])/np.abs(i_py[m])):.2e}")

# --- 3) transient: filter -> diode limiter, compare with PortCircuitSimulator
f0, AMP, H, TEND = 2.0e9, 2.0, 2e-12, 8e-9
A, B, C, D = cm.state_space()
sim = mlrf.PortCircuitSimulator(A, B, C, D, 50.0, [("source", lambda t: AMP*np.sin(2*np.pi*f0*t)),
                                                   ("load", dict(R=50.0, C=0.5e-12, nl=dg.current))])
t, Vp = sim.run(TEND, H / 4)
net = f"""* transient
.include {lib}
Vs src 0 SIN(0 {AMP} {f0})
Rs src p1 50
X1 p1 p2 filt
RL p2 0 50
CL p2 0 0.5p
X2 p2 0 dml
.tran 0 {TEND} 0 {H/4}
.options plotwinsize=0 reltol=1e-5 abstol=1e-12 vntol=1e-7
.end
"""
r = run(net, "tran")
tl = np.abs(np.real(r["time"])); vl = np.real(r["v(p2)"])
vpy = np.interp(tl, t, Vp[:, 1]); sl = tl > 2e-9
print(f"TRAN: RMS |v_LT - v_py| (t>2ns) = {np.sqrt(np.mean((vl-vpy)[sl]**2))*1e3:.3f} mV, peak |v| = {np.max(np.abs(vl[sl])):.3f} V")
