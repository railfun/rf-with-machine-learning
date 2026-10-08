"""load_touchstone / load_iv と、write_ltspice が作るシンボル (.asy) の検証

    python verify_ltspice_symbols.py      (LTspice の場所は環境変数 LTSPICE で変更できる)
1) θ をファイル名に含む Touchstone 群 (周波数点がばらばら・DC 点あり) を load_touchstone で読み、
   配列から直接学習した GP と比べる
2) I-V の CSV (mA 単位・ヘッダ付き) を load_iv で読む
3) .lib + .asy を書き出し、シンボルを置いた回路図 (.asc) を LTspice で解いて、ネットリスト版と比べる
   (ref ピン付きのシンボルも確認)
"""
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, skrf as rf
from scipy.special import lambertw
from skrf.media import DefinedGammaZ0
import mlrf

LT = os.environ.get("LTSPICE", os.path.expandvars(r"%LOCALAPPDATA%\Programs\ADI\LTspice\LTspice.exe"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltspice_check")
DATA = os.path.join(OUT, "data")
os.makedirs(DATA, exist_ok=True)


def read_raw(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    nv = int(next(l for l in lines if l.startswith("No. Variables:")).split(":")[1])
    iv = lines.index("Variables:")
    names = [lines[iv + 1 + k].split("\t")[2].lower() for k in range(nv)]
    cplx = any("complex" in l for l in lines[:iv])
    vals, row, data = lines[lines.index("Values:") + 1:], [], []
    for l in vals:
        if not l.strip():
            continue
        tok = l.split("\t")[-1].strip()
        row.append(complex(*map(float, tok.split(","))) if cplx else float(tok))
        if len(row) == nv:
            data.append(row); row = []
    d = np.array(data)
    return {n: d[:, k] for k, n in enumerate(names)}


def run_lt(path):
    subprocess.run([LT, "-b", "-ascii", path], check=True, timeout=300)
    return read_raw(os.path.splitext(path)[0] + ".raw")


# ---------------------------------------------------------------- 1) Touchstone 群
def filt(th, f):
    med = DefinedGammaZ0(frequency=rf.Frequency.from_f(f, unit="hz"), z0=50)
    return (med.shunt_inductor(th[0] * 1e-9) ** med.shunt_capacitor(2e-12) ** med.capacitor(th[1] * 1e-12)
            ** med.shunt_inductor(th[0] * 1e-9) ** med.shunt_capacitor(2e-12))

f = np.linspace(0.5e9, 5e9, 91)
ths = np.array([[a, b] for a in np.linspace(1.5, 3, 4) for b in np.linspace(0.3, 1.2, 4)])
for k, th in enumerate(ths):
    fk = np.r_[0.0, f] if k == 0 else (np.linspace(0.4e9, 5.2e9, 120) if k == 1 else f)   # DC 点・別グリッド
    filt(th, fk).write_touchstone(os.path.join(DATA, f"filter_L{th[0]:.3f}nH_Cc{th[1]:.3f}pF.s2p"))
th_l, f_l, S_l = mlrf.load_touchstone(os.path.join(DATA, "*.s2p"), pattern=r"L([\d.]+)nH_Cc([\d.]+)pF")
print(f"load_touchstone: {len(th_l)} ファイル, F={len(f_l)}, f=[{f_l[0]/1e9:.2f}, {f_l[-1]/1e9:.2f}] GHz")
order = np.lexsort(ths.T[::-1]); order_l = np.lexsort(th_l.T[::-1])
print("  θ の取り出し 最大誤差:", np.max(np.abs(th_l[order_l] - ths[order])))
gp_file = mlrf.ParametricRationalGP([(1.5, 3.0), (0.3, 1.2)]).fit_touchstone(
    os.path.join(DATA, "*.s2p"), pattern=r"L([\d.]+)nH_Cc([\d.]+)pF")
TH = [2.2, 0.7]
S_exact = np.array([filt(t, f_l).s for t in th_l])
print(f"  読み込んだ S と厳密値の最大差 (別グリッドのファイルは 3 次補間): {np.max(np.abs(S_l - S_exact)):.1e}")

# ---------------------------------------------------------------- 2) I-V の CSV
IS, N_ID, RS, VT = 5e-9, 1.05, 10.0, 0.025865
dtrue = lambda v: np.real(N_ID*VT/RS*lambertw(IS*RS/(N_ID*VT)*np.exp((v+IS*RS)/(N_ID*VT)))) - IS
V = np.r_[np.linspace(-1, 0.9, 12), 0.6, 0.7, 0.75, 0.8]
csv = os.path.join(DATA, "diode_iv.csv")
with open(csv, "w", encoding="utf-8") as fp:
    fp.write("# diode measurement\nVd [V],Id [mA],temp\n")
    fp.writelines(f"{v:.6f},{i*1e3:.9e},25\n" for v, i in zip(V, dtrue(V)))
Vc, Ic = mlrf.load_iv(csv, v_col="Vd [V]", i_col=1, i_scale=1e-3)
print(f"load_iv: {len(Vc)} 点, 電流の最大相対誤差 {np.max(np.abs(Ic - dtrue(V)) / np.abs(dtrue(V))):.1e}")
dg = mlrf.StaticNonlinearGP(1e-6).fit_file(csv, v_col=0, i_col="Id [mA]", i_scale=1e-3)

# ---------------------------------------------------------------- 3) シンボル付きで書き出し → 回路図で解く
cm = gp_file.compact(TH)
lib = mlrf.write_ltspice(os.path.join(OUT, "sym_models.lib"), [cm, dg], names=["filt_s", "dml_s"])
lib_r = mlrf.write_ltspice(os.path.join(OUT, "sym_models_ref.lib"), cm, names="filt_r", ref_pin=True)
print("書き出し:", [os.path.basename(p) for p in sorted(os.listdir(OUT)) if p.endswith((".asy", ".lib")) and "sym" in p or p.startswith(("filt_", "dml_"))])

TR = ".tran 0 8n 0 0.5p"
OPT = ".options plotwinsize=0 reltol=1e-5 abstol=1e-12 vntol=1e-7"
asc = "\n".join([
    "Version 4", "SHEET 1 1200 680",
    "SYMBOL filt_s 0 0 R0", "SYMATTR InstName X1",
    "FLAG -64 0 p1", "FLAG 64 0 p2",
    "SYMBOL dml_s 400 0 R0", "SYMATTR InstName X2",
    "FLAG 416 0 p2", "FLAG 416 64 0",
    "SYMBOL filt_r 0 300 R0", "SYMATTR InstName X3",
    "FLAG -64 300 q1", "FLAG 64 300 q2", "FLAG 0 332 0",
    "TEXT -100 450 Left 2 !Vs src 0 SIN(0 2 2G)",
    "TEXT -100 480 Left 2 !Rs src p1 50",
    "TEXT -100 510 Left 2 !RL p2 0 50",
    "TEXT -100 540 Left 2 !CL p2 0 0.5p",
    "TEXT -100 570 Left 2 !Vq srcq 0 SIN(0 2 2G)",
    "TEXT -100 600 Left 2 !Rq srcq q1 50",
    "TEXT -100 630 Left 2 !RqL q2 0 50",
    f"TEXT -100 660 Left 2 !{TR}",
    f"TEXT -100 690 Left 2 !{OPT}",
]) + "\n"
asc_path = os.path.join(OUT, "sym_test.asc")
open(asc_path, "w", encoding="ascii", newline="\r\n").write(asc)
subprocess.run([LT, "-netlist", asc_path], check=True, timeout=120)
net = open(os.path.splitext(asc_path)[0] + ".net", encoding="latin-1").read()
print("回路図から LTspice が作ったネットリスト (抜粋):")
print("\n".join("   " + l for l in net.splitlines() if l.startswith(("X", ".lib", ".include", ".inc"))))
r_asc = run_lt(asc_path)

net_path = os.path.join(OUT, "sym_ref_netlist.cir")
open(net_path, "w").write(f"""* netlist version
.include {os.path.join(OUT, 'sym_models.lib')}
X1 p1 p2 filt_s
X2 p2 0 dml_s
Vs src 0 SIN(0 2 2G)
Rs src p1 50
RL p2 0 50
CL p2 0 0.5p
{TR}
{OPT}
.end
""")
r_net = run_lt(net_path)
t1, v1 = np.abs(r_asc["time"]), r_asc["v(p2)"]
t2, v2 = np.abs(r_net["time"]), r_net["v(p2)"]
print(f"回路図 (シンボル) vs ネットリスト: v(p2) の最大差 {np.max(np.abs(np.interp(t2, t1, v1) - v2))*1e3:.4f} mV")
A, B, C, D = cm.state_space()
sim = mlrf.PortCircuitSimulator(A, B, C, D, 50.0, [("source", lambda t: 2*np.sin(2*np.pi*2e9*t)), ("load", dict(R=50.0))])
t, Vp = sim.run(8e-9, 0.5e-12)
tq, vq = np.abs(r_asc["time"]), r_asc["v(q2)"]
print(f"ref ピン付きシンボル (ref=GND, 50Ω 負荷) vs PortCircuitSimulator: RMS 差 "
      f"{np.sqrt(np.mean((np.interp(tq, t, Vp[:, 1]) - vq)**2))*1e3:.4f} mV")
