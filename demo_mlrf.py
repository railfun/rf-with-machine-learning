"""
demo_mlrf.py — mlrf の通し実行例

1) 線形素子: 2 段共振器フィルタ (θ = 共振器 L [nH], 結合容量 Cc [pF]) を能動学習で GP 化
2) 非線形素子: ダイオードの I-V 測定 (雑音あり) を能動学習で GP 化
3) θ* でコンパクトモデルを作り、フィルタ出力にダイオードリミッタをつないだ回路を過渡解析
   - Python (PortCircuitSimulator) と ngspice (真の回路 / 書き出した ML モデル) で比較
4) ダイオードモデルの不確かさを事後サンプルで過渡波形まで伝播
"""
import os, shutil, subprocess, tempfile, time, warnings
import numpy as np
import skrf as rf
from scipy.special import lambertw
from skrf.media import DefinedGammaZ0
from PySpice.Spice.Netlist import Circuit
from mlrf import ParametricRationalGP, StaticNonlinearGP, active_learning, PortCircuitSimulator
# mlrfのインポートは、PySpiceのインポートよりも後に行わなければならない。

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ 「測定/EM シミュレーション」の代役
f = np.linspace(0.5e9, 5e9, 181)
med = DefinedGammaZ0(frequency=rf.Frequency.from_f(f, unit="hz"), z0=50)
C_RES = 2e-12


def filter_true(th):
    L, Cc = th[0] * 1e-9, th[1] * 1e-12
    return (med.shunt_inductor(L) ** med.shunt_capacitor(C_RES) ** med.capacitor(Cc)
            ** med.shunt_inductor(L) ** med.shunt_capacitor(C_RES)).s


def filter_measure(th, noise=3e-3):
    return f, filter_true(th) + noise * (rng.standard_normal((len(f), 2, 2))
                                         + 1j * rng.standard_normal((len(f), 2, 2)))


IS, N_ID, RS, VT = 5e-9, 1.05, 10.0, 1.380649e-23 * 300.15 / 1.602176634e-19


def diode_true(v):
    a = N_ID * VT
    return np.real(a / RS * lambertw(IS * RS / a * np.exp((v + IS * RS) / a))) - IS


def diode_measure(v):
    i = diode_true(v)
    return i * (1 + 0.02 * rng.standard_normal(np.shape(v))) + 1e-9 * rng.standard_normal(np.shape(v))


# ------------------------------------------------------------------ 1) 線形素子
print("=== 1) 線形素子: パラメトリック有理 GP + 能動学習 ===")
bounds = [(1.5, 3.0), (0.3, 1.2)]
t0 = time.perf_counter()
gp, thetas, _ = active_learning(filter_measure, bounds, n_init=4, n_iter=12, verbose=False)
print(f"  学習: {len(thetas)} サンプル, {time.perf_counter()-t0:.1f} s, 超パラメータ: "
      f"α/2π={gp.hyper['alpha_Hz']/1e9:.3f} GHz, ℓ={np.round(gp.hyper['lengthscale'], 3)}, "
      f"λ={gp.hyper['noise_ratio']:.1e}")
test = np.random.default_rng(7).uniform([1.5, 0.3], [3.0, 1.2], (40, 2))


def evaluate(label):
    err, cover, zs = [], [], []
    for th in test:
        M, sd = gp.predict(th, f)                               # sd: (F, P, P) S_ij ごとの 1σ
        e = np.abs(M - filter_true(th))
        err.append(np.sqrt(np.mean(e ** 2))); cover.append(np.mean(e <= 2 * sd)); zs.append(np.mean((e / sd) ** 2))
    print(f"  未学習の θ 40 点 [{label}]: RMS 誤差 平均 {np.mean(err):.4f} / 最大 {np.max(err):.4f}, "
          f"±2σ に真値が入る割合 {np.mean(cover)*100:.1f} % (理想 98.2 %), "
          f"√mean(|e|²/σ²) = {np.sqrt(np.mean(zs)):.2f} (理想 1)")


evaluate("較正前")
print(f"  leave-one-out による σ の倍率 κ = {gp.calibrate_std():.3f}")
evaluate("較正後")

TH = np.array([2.2, 0.7])
t0 = time.perf_counter()
cm = gp.compact(TH)
print(f"  θ*={TH}: コンパクトモデル 次数 {cm.order}, 受動 {cm.vf.is_passive()}, "
      f"真値との RMS 誤差 {np.sqrt(np.mean(np.abs(cm.response(f) - filter_true(TH))**2)):.4f}, "
      f"{time.perf_counter()-t0:.2f} s")

# ------------------------------------------------------------------ 2) 非線形素子
print("=== 2) 非線形素子: ダイオード I-V の GP + 能動学習 ===")
V = np.linspace(-1.0, 0.9, 6)
I = diode_measure(V)
dg = StaticNonlinearGP(I0=1e-6).fit(V, I)
cand = np.linspace(-1.0, 0.9, 381)
for _ in range(10):
    vn = dg.suggest(cand)
    V, I = np.r_[V, vn], np.r_[I, diode_measure(np.array([vn]))]
    dg.fit(V, I)
vv = np.linspace(-0.5, 0.9, 200)
rel = np.abs(dg.current(vv)[0] - diode_true(vv)) / (np.abs(diode_true(vv)) + 1e-6)
print(f"  測定 {len(V)} 点, 相対誤差 (I>1µA の領域) 中央値 {np.median(rel[diode_true(vv) > 1e-6])*100:.2f} %")

# ------------------------------------------------------------------ 3) 過渡解析
print("=== 3) 過渡解析: フィルタ → ダイオードリミッタ ===")
f0 = f[np.argmax(np.abs(filter_true(TH)[:, 1, 0]))]
AMP, RL, CL, H, TEND = 2.0, 50.0, 0.5e-12, 2e-12, 8e-9
A, B, C, D = cm.state_space()
src = lambda t: AMP * np.sin(2 * np.pi * f0 * t)
sim = PortCircuitSimulator(A, B, C, D, 50.0,
                           [("source", src), ("load", dict(R=RL, C=CL, nl=dg.current))])
t0 = time.perf_counter()
t, Vp = sim.run(TEND, H)
t_py = time.perf_counter() - t0
print(f"  Python: 状態数 {A.shape[0]}, {len(t)-1} ステップ, Newton 反復 {sim.n_newton} 回, {t_py:.2f} s")


def ngspice(netlist):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "c.cir"), "w") as fp:
        fp.write(netlist)
    t0 = time.perf_counter()
    subprocess.run(["ngspice", "-b", "c.cir"], cwd=d, capture_output=True)
    el = time.perf_counter() - t0
    data = np.loadtxt(os.path.join(d, "out.txt"))
    shutil.rmtree(d)
    return data[:, 0], data[:, 1], el


ctrl = f""".control
set filetype=ascii
tran {H} {TEND} 0 {H}
wrdata out.txt v(p2)
quit
.endc
.end
"""
true_net = f"""* true circuit
Vs src 0 SIN(0 {AMP} {f0})
Rs src p1 50
L1 p1 0 {TH[0]}n
C1 p1 0 {C_RES}
Cc p1 p2 {TH[1]}p
L2 p2 0 {TH[0]}n
C2 p2 0 {C_RES}
RL p2 0 {RL}
CL p2 0 {CL}
D1 p2 0 DMOD
.model DMOD D(IS={IS} N={N_ID} RS={RS})
""" + ctrl
sub = os.path.join(OUT, "filter_model.sp")
cm.write_spice(sub, name="filt")
diode_line = dg.spice_bsource("BD1", "p2", "0")
with open(os.path.join(OUT, "diode_model.inc"), "w") as fp:
    fp.write("* GP diode model (mean)\n" + diode_line + "\n")
ml_net = f"""* ML models
.include {sub}
Vs src 0 SIN(0 {AMP} {f0})
Rs src p1 50
X1 p1 p2 filt
RL p2 0 {RL}
CL p2 0 {CL}
{diode_line}
""" + ctrl
if shutil.which("ngspice"):
    tt, vt, el_true = ngspice(true_net)
    tm, vm, el_ml = ngspice(ml_net)
    v_py = np.interp(tt, t, Vp[:, 1])
    v_ml = np.interp(tt, tm, vm)
    sl = tt > 2e-9
    print(f"  ngspice 真の回路: {el_true:.2f} s / ngspice ML モデル: {el_ml:.2f} s (起動時間込み)")
    print(f"  v(p2) の RMS 差 (t>2ns): Python-ML vs 真値 {np.sqrt(np.mean((v_py-vt)[sl]**2))*1e3:.1f} mV, "
          f"ngspice-ML vs 真値 {np.sqrt(np.mean((v_ml-vt)[sl]**2))*1e3:.1f} mV, "
          f"Python-ML vs ngspice-ML {np.sqrt(np.mean((v_py-v_ml)[sl]**2))*1e3:.2f} mV  "
          f"(振幅 {np.max(np.abs(vt[sl])):.2f} V)")
else:
    tt = vt = None
    print("  ngspice が見つからないので比較を省略")

# ------------------------------------------------------------------ 4) 不確かさの伝播
print("=== 4) ダイオードの不確かさを過渡波形へ伝播 (事後サンプル 20 本) ===")
samples = dg.sample_functions(np.linspace(-1.2, 1.2, 241), 20, seed=1)
t0 = time.perf_counter()
_, W = sim.run_ensemble(TEND, H, [samples], len(samples))     # 20 本をまとめて解く (LU を共有)
W = W[:, :, 1]
lo, hi = np.percentile(W, [2.5, 97.5], axis=0)
sl = t > 2e-9
print(f"  {time.perf_counter()-t0:.1f} s, 95% 帯の幅 平均 {np.mean(hi[sl]-lo[sl])*1e3:.1f} mV, "
      f"最大 {np.max(hi[sl]-lo[sl])*1e3:.1f} mV")

# ------------------------------------------------------------------ 図
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
M, sd = gp.predict(TH, f)
Tr = filter_true(TH)
ax[0, 0].plot(f / 1e9, np.abs(Tr[:, 1, 0]), "k", label="true |S21|")
ax[0, 0].plot(f / 1e9, np.abs(M[:, 1, 0]), "C0--", label="GP mean")
ax[0, 0].fill_between(f / 1e9, np.abs(M[:, 1, 0]) - 2 * sd[:, 1, 0], np.abs(M[:, 1, 0]) + 2 * sd[:, 1, 0],
                      color="C0", alpha=0.25, label="±2σ")
ax[0, 0].plot(f / 1e9, np.abs(cm.response(f)[:, 1, 0]), "C3:", label=f"compact (order {cm.order})")
ax[0, 0].set(title=f"Linear: θ*={TH} (not in training set)", xlabel="GHz"); ax[0, 0].legend(fontsize=8)
lo_i, hi_i = dg.current_band(vv)
ax[0, 1].semilogy(vv, np.abs(diode_true(vv)), "k", label="true")
ax[0, 1].semilogy(vv, np.abs(dg.current(vv)[0]), "C0--", label="GP mean")
ax[0, 1].fill_between(vv, np.abs(lo_i) + 1e-12, np.abs(hi_i), color="C0", alpha=0.25, label="±2σ")
ax[0, 1].semilogy(V, np.abs(I) + 1e-12, "kx", label="measured")
ax[0, 1].set(title="Nonlinear: diode I-V", xlabel="V", ylabel="|I| (A)", ylim=(1e-11, 1)); ax[0, 1].legend(fontsize=8)
if tt is not None:
    ax[1, 0].plot(tt * 1e9, vt, "k", lw=2, label="ngspice: true circuit")
    ax[1, 0].plot(t * 1e9, Vp[:, 1], "C0--", label="Python: ML models")
    ax[1, 0].plot(tm * 1e9, vm, "C3:", label="ngspice: exported ML models")
ax[1, 0].set(title="Transient v(p2)", xlabel="ns", ylabel="V", xlim=(4, 8)); ax[1, 0].legend(fontsize=8)
ax[1, 1].plot(t * 1e9, (hi - lo) * 1e3, "C1", label="95% band width (diode UQ)")
ax[1, 1].plot(t * 1e9, Vp[:, 1] * 10, "C0", lw=0.6, alpha=0.5, label="v(p2) ×10 [mV scale ref]")
ax[1, 1].set(title="Uncertainty propagation", xlabel="ns", ylabel="mV", xlim=(4, 8)); ax[1, 1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "demo_mlrf.png"), dpi=120)
print(f"図: {os.path.join(OUT, 'demo_mlrf.png')}")
