"""
mlrf.py — 機械学習 (ガウス過程) による高周波素子モデルと高速シミュレータ

線形素子   : ParametricRationalGP
    周波数方向 = Szegö カーネル (有理関数・安定・実係数を保証)
    設計/バイアス方向 θ = ARD-RBF カーネル
    K = Kθ ⊗ Ks のクロネッカー構造で O(Nθ³ + F³) の学習、予測分散 (不確かさ) 付き
    信号分散 σ² は S_ij ごとに推定 (大きさの違う S11 と S21 で不確かさを別々に較正)
    任意の θ で「極固定・留数が θ の関数」の有理モデル → VF で縮約 → 状態空間 / SPICE
非線形素子 : StaticNonlinearGP
    I(V) を asinh 変換して GP 回帰 (1 次の平均関数)。平均・解析的な微分 (Newton 法用)・
    平均関数の係数の不確かさまで含めた分散・事後サンプル (まとめて評価できる)
    平均は exp 項の和なので SPICE の B 素子 (ビヘイビア電源) の式に直接書き出せる
能動学習   : active_learning()  (IMSE 基準 / 予測分散基準で次に測定/シミュレーションする θ を選ぶ)
シミュレータ: PortCircuitSimulator
    線形 N ポート状態空間 + ポート終端 (内部抵抗つき電圧源 / R ∥ C ∥ 非線形負荷) の過渡解析
    台形則 + Newton 法、線形部分の LU 分解は 1 回だけ、非線形部分は Woodbury の低ランク更新
    C = 0 の負荷 (代数方程式)、DC 動作点、事後サンプルをまとめて解くアンサンブル実行に対応
"""
import warnings
import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve, cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.stats import qmc
import skrf as rf
from skrf.vectorFitting import VectorFitting
import PySpice

__all__ = ["ParametricRationalGP", "CompactModel", "StaticNonlinearGP", "SplineSamples", "write_touchstone",
           "write_ltspice", "load_touchstone", "load_iv",
           "active_learning", "PortCircuitSimulator"]


# =====================================================================
# カーネル
# =====================================================================
def szego_real(w1, w2, a):
    """[Re; Im] 実数化した Szegö カーネル (k) + 擬似カーネル (c)。s = jω (正規化済み)"""
    k = 1 / (2 * np.pi * (2 * a + 1j * (w1[:, None] - w2[None, :])))
    c = 1 / (2 * np.pi * (2 * a + 1j * (w1[:, None] + w2[None, :])))
    RR, II = 0.5 * (k + c).real, 0.5 * (k - c).real
    RI, IR = 0.5 * (-k + c).imag, 0.5 * (k + c).imag
    return np.block([[RR, RI], [IR, II]])


def rbf(X1, X2, ell):
    d = (X1[:, None, :] - X2[None, :, :]) / ell
    return np.exp(-0.5 * np.sum(d * d, -1))


# =====================================================================
# 線形素子: パラメトリック有理 GP
# =====================================================================
class ParametricRationalGP:
    """S(θ, s) を学習する GP。θ: 設計パラメータやバイアス (D 次元)

    bounds           : [(min, max), ...]  θ の範囲 (正規化に使用)
    per_output_scale : True なら信号分散 σ² を S_ij ごとに推定する (False で全 S_ij 共通 = 旧版)
    noise_bounds     : 雑音比 λ = σ_n²/σ² の探索範囲。雑音を補間してしまう (λ が下限に張り付く) ときは下限を上げる
    """

    def __init__(self, bounds, names=None, per_output_scale=True, noise_bounds=(1e-9, 1e-1)):
        b = np.asarray(bounds, float)
        if b.ndim != 2 or b.shape[1] != 2 or np.any(b[:, 1] <= b[:, 0]):
            raise ValueError("bounds は [(min, max), ...] の形で min < max にしてください")
        self.lb, self.ub = b[:, 0], b[:, 1]
        self.D = len(b)
        self.names = names or [f"theta{i}" for i in range(self.D)]
        self.per_output_scale = per_output_scale
        self.noise_bounds = noise_bounds
        self.std_scale = 1.0                                        # calibrate_std() で決まる σ の倍率

    def _nt(self, th):
        return (np.atleast_2d(np.asarray(th, float)) - self.lb) / (self.ub - self.lb)

    # ------------------------------------------------------------- 学習
    def fit(self, thetas, freqs, S, optimize=True, hyper_stride=2, x0=None, n_restarts=0, seed=0):
        """thetas: (Nθ, D), freqs: (F,) [Hz] 全サンプル共通, S: (Nθ, F, P, P) 複素

        n_restarts : 超パラメータ最適化の追加ランダム初期点の数 (局所解対策)
        """
        S = np.asarray(S)
        thetas = np.atleast_2d(np.asarray(thetas, float))
        freqs = np.asarray(freqs, float)
        if S.ndim != 4 or S.shape[2] != S.shape[3]:
            raise ValueError(f"S は (Nθ, F, P, P) の形にしてください (受け取った形: {S.shape})")
        if thetas.shape != (len(S), self.D):
            raise ValueError(f"thetas の形 {thetas.shape} が (Nθ={len(S)}, D={self.D}) と合いません")
        if S.shape[1] != len(freqs):
            raise ValueError(f"周波数点数 {len(freqs)} と S の周波数方向 {S.shape[1]} が合いません")
        if np.any(freqs <= 0):
            raise ValueError("f > 0 の点だけを渡してください (DC 点は擬似カーネルで縮退)")
        self.P = S.shape[-1]
        self.freqs = freqs
        self.w0 = 2 * np.pi * self.freqs.max()
        self.wn = 2 * np.pi * self.freqs / self.w0
        self.T = self._nt(thetas)
        Sr = S.reshape(len(S), len(freqs), -1)                     # (Nθ, F, R)
        self.Y = np.concatenate([Sr.real, Sr.imag], axis=1)        # (Nθ, 2F, R)
        D, nh = self.D, self.D + 3
        if x0 is None:
            prev = getattr(self, "_h", None)
            x0 = prev if prev is not None and len(prev) == nh else \
                np.r_[np.log(0.02), np.zeros(D) + np.log(0.5), np.log(1e-4), 0.0]
        x0 = np.asarray(x0, float)
        if len(x0) != nh:
            raise ValueError(f"x0 の長さは {nh} (log α, log ℓ×D, log λ, log c0) にしてください")
        if optimize:
            idx = np.arange(0, len(freqs), max(1, hyper_stride))  # 超パラメータは間引いた周波数で
            Ysub = np.concatenate([Sr[:, idx].real, Sr[:, idx].imag], axis=1)
            bnds = [(np.log(1e-4), np.log(0.5))] + [(np.log(0.03), np.log(10.0))] * D \
                   + [tuple(np.log(self.noise_bounds)), (np.log(1e-3), np.log(1e3))]
            lo, hi = np.array(bnds).T
            rng = np.random.default_rng(seed)
            starts = [np.clip(x0, lo, hi)] + [rng.uniform(lo, hi) for _ in range(n_restarts)]
            obj = lambda h: self._nll(h, self.wn[idx], self.T, Ysub)[0]
            self._h = min((minimize(obj, s, method="L-BFGS-B", bounds=bnds) for s in starts),
                          key=lambda r: r.fun).x
        else:
            self._h = x0
        _, self._cache = self._nll(self._h, self.wn, self.T, self.Y, keep=True)
        return self

    def fit_touchstone(self, files, thetas=None, pattern=None, f=None, z0=50.0, **fit_kw):
        """Touchstone ファイル群から学習する。引数は load_touchstone と同じ (fit_kw は fit に渡す)"""
        return self.fit(*load_touchstone(files, thetas=thetas, pattern=pattern, f=f, z0=z0), **fit_kw)

    def _unpack(self, h):
        D = self.D
        return np.exp(h[0]), np.exp(h[1:1 + D]), np.exp(h[1 + D]), np.exp(h[2 + D])

    def _nll(self, h, wn, T, Y, keep=False):
        a, ell, lam, c0 = self._unpack(h)
        F = len(wn)
        Ks = szego_real(wn, wn, a)
        Ks[:F, :F] += c0                                           # 実定数項 d の事前分散
        es, Us = eigh(Ks); es = np.clip(es, 0, None)
        et, Ut = eigh(rbf(T, T, ell)); et = np.clip(et, 0, None)
        L = et[:, None] * es[None, :] + lam                        # Kθ⊗Ks + λI の固有値
        Yt = np.einsum("ij,jkr,kl->ilr", Ut.T, Y, Us, optimize=True)
        n1, R = L.size, Y.shape[2]                                 # n1: 1 出力あたりのデータ数
        Q = np.sum(Yt ** 2 / L[:, :, None], axis=(0, 1))           # (R,) yᵀ K⁻¹ y
        if self.per_output_scale:                                  # σ_r² を出力ごとに解析的に消去
            s2 = np.maximum(Q / n1, 1e-12 * Q.max() / n1 + 1e-300) # 恒等的に 0 の出力 (S12=0 など) 対策
            nll = 0.5 * n1 * np.sum(np.log(s2)) + 0.5 * R * np.sum(np.log(L))
        else:                                                      # 全出力で共通の σ²
            s2 = np.full(R, Q.sum() / (n1 * R))
            nll = 0.5 * n1 * R * np.log(s2[0]) + 0.5 * R * np.sum(np.log(L))
        if not keep:
            return nll, None
        W = np.einsum("ij,jkr,kl->ilr", Ut, Yt / L[:, :, None], Us.T, optimize=True)   # (K)^-1 y
        return nll, dict(a=a, ell=ell, lam=lam, c0=c0, s2=s2, Us=Us, es=es, Ut=Ut, L=L, W=W)

    @property
    def hyper(self):
        c = self._cache
        return dict(alpha_Hz=c["a"] * self.w0 / (2 * np.pi), lengthscale=c["ell"],
                    noise_ratio=c["lam"], sigma2=c["s2"].reshape(self.P, self.P), c0=c["c0"])

    # ------------------------------------------------------------- 予測
    def rational(self, theta):
        """θ における有理モデル: 極 (Im≥0), 留数 (R, F), 定数項 d (R,)"""
        c = self._cache
        kt = rbf(self.T, self._nt(theta), c["ell"])[:, 0]
        Wt = np.einsum("i,ikr->kr", kt, c["W"])                    # (2F, R)
        F = len(self.wn)
        gamma = 0.5 * (Wt[:F] + 1j * Wt[F:])
        poles = self.w0 * (-2 * c["a"] + 1j * self.wn)
        residues = self.w0 * gamma.T / (2 * np.pi)
        d = c["c0"] * Wt[:F].sum(0)
        return poles, residues, d

    def predict(self, theta, f, return_std=True):
        """平均 S (Ft, P, P) と 1σ (Ft, P, P) [Re と Im の分散の和の平方根, 雑音を含まない]

        複素誤差が円対称な正規分布なら |誤差| ≤ 2σ に入る確率は 1 - e⁻⁴ ≈ 98.2 %
        """
        poles, res, d = self.rational(theta)
        f = np.asarray(f, float)
        s = 2j * np.pi * f[:, None]
        M = d[None, :] + (1 / (s - poles)) @ res.T + (1 / (s - np.conj(poles))) @ np.conj(res).T
        M = M.reshape(len(f), self.P, self.P)
        if not return_std:
            return M
        var = self._var(self._nt(theta), f)
        Ft = len(f)
        v = np.clip(var[:Ft] + var[Ft:], 0, None)
        sd = self.std_scale * np.sqrt(v[:, None] * self._cache["s2"][None, :])
        return M, sd.reshape(Ft, self.P, self.P)

    def calibrate_std(self, apply=True):
        """θ 方向の leave-one-out で予測 σ の倍率 κ を推定する (超パラメータは固定)

        各 θ_i を抜いた GP で θ_i の測定値を予測し、κ² = mean(|誤差|² / 予測分散 (雑音込み))。
        周波数方向の固有値分解を共有するので、追加の計算は Nθ 回の小さな固有値分解だけ。
        apply=True なら以後の predict() の σ に κ を掛ける。κ を返す
        """
        c = self._cache
        Nt, F2 = len(self.T), 2 * len(self.wn)
        if Nt < 3:
            raise ValueError("leave-one-out には θ のサンプルが 3 点以上必要です")
        Us, es, lam, s2 = c["Us"], c["es"], c["lam"], c["s2"]
        Kth = rbf(self.T, self.T, c["ell"])
        V2 = Us ** 2 * es ** 2                                      # (Ks Us)² の各列 (2F, 2F)
        prior = np.sum(Us ** 2 * es, 1)                             # diag(Ks)
        z2 = []
        for i in range(Nt):
            m = np.arange(Nt) != i
            et, Ut = eigh(Kth[np.ix_(m, m)]); et = np.clip(et, 0, None)
            L = et[:, None] * es[None, :] + lam
            Yt = np.einsum("ij,jkr,kl->ilr", Ut.T, self.Y[m], Us, optimize=True)
            u = Ut.T @ Kth[m, i]
            mean = (Us * es) @ np.einsum("i,ikr->kr", u, Yt / L[:, :, None])           # (2F, R)
            var = prior - V2 @ ((u ** 2) @ (1 / L)) + lam                              # (2F,) 雑音込み
            var = (var[:F2 // 2] + var[F2 // 2:])[:, None] * s2[None, :]
            e2 = (mean[:F2 // 2] - self.Y[i, :F2 // 2]) ** 2 + (mean[F2 // 2:] - self.Y[i, F2 // 2:]) ** 2
            z2.append(np.mean(e2 / var))
        kappa = float(np.sqrt(np.mean(z2)))
        if apply:
            self.std_scale = kappa
        return kappa

    def _Vtest(self, f):
        c = self._cache
        wt = 2 * np.pi * f / self.w0
        Kx = szego_real(wt, self.wn, c["a"])
        Ft, F = len(wt), len(self.wn)
        Kx[:Ft, :F] += c["c0"]
        prior = np.diag(szego_real(wt, wt, c["a"])).copy()
        prior[:Ft] += c["c0"]
        return Kx @ c["Us"], prior

    def _var(self, Tn, f):
        """σ² = 1 に規格化した事後分散 (2Ft,)"""
        c = self._cache
        V, prior = self._Vtest(f)
        u = c["Ut"].T @ rbf(self.T, Tn, c["ell"])[:, 0]
        q = (V ** 2) @ ((u ** 2) @ (1 / c["L"]))
        return prior - q

    def acquisition(self, cand, f=None, mode="imse", n_ref=256, seed=0):
        """能動学習の評価値 (大きいほど次に測る価値が高い)

        mode="imse": θ 空間全体の予測分散の減少量 (積分平均二乗誤差の削減)。境界に偏りにくい
        mode="var" : その点の周波数平均の予測分散
        """
        c = self._cache
        if mode == "imse":
            D = self.D
            ref = qmc.Sobol(d=D, seed=seed).random(n_ref)            # 正規化 θ 空間の参照点
            Cn = self._nt(cand)
            nug = c["lam"] / np.max(c["L"]) + 1e-8
            Kt = rbf(self.T, self.T, c["ell"]) + nug * np.eye(len(self.T))
            Lk = cho_factor(Kt)
            kTr, kTc = rbf(self.T, ref, c["ell"]), rbf(self.T, Cn, c["ell"])
            cov_rc = rbf(ref, Cn, c["ell"]) - kTr.T @ cho_solve(Lk, kTc)       # 事後共分散 (n_ref, Nc)
            var_c = 1 + nug - np.sum(kTc * cho_solve(Lk, kTc), 0)
            return np.mean(cov_rc ** 2, 0) / np.clip(var_c, 1e-12, None)
        if mode != "var":
            raise ValueError(f"mode は 'imse' か 'var' です (受け取った値: {mode!r})")
        f = self.freqs if f is None else np.asarray(f, float)
        V, prior = self._Vtest(f)
        colV = np.sum(V ** 2, 0)                                    # (2F,)
        U = c["Ut"].T @ rbf(self.T, self._nt(cand), c["ell"])       # (Nθ, Nc)
        q = ((U ** 2).T @ (1 / c["L"])) @ colV                      # (Nc,)
        return np.mean(c["s2"]) * (prior.sum() - q) / len(f)

    # ------------------------------------------------------------- 縮約
    def compact(self, theta, f=None, n_dense=401, enforce_passivity=True, z0=50.0):
        """θ での平均応答を VF で低次の有理モデルに縮約 (MOR の代わり) し、受動性を強制

        受動性の強制で S 全体を縮小した場合は、その倍率を CompactModel.passivity_scale に残す
        """
        f = np.linspace(self.freqs.min(), self.freqs.max(), n_dense) if f is None else np.asarray(f, float)
        M = self.predict(theta, f, return_std=False)
        nw = rf.Network(f=f, s=M, f_unit="Hz", z0=z0)
        vf = VectorFitting(nw)
        vf.auto_fit()
        scale = 1.0
        if enforce_passivity and not vf.is_passive():
            orig = (vf.poles.copy(), vf.residues.copy(), vf.constant_coeff.copy())
            try:
                vf.passivity_enforce(n_samples=1000)                # 留数の摂動 (scikit-rf)
            except Exception as ex:
                warnings.warn(f"passivity_enforce が例外で失敗しました: {ex!r}")
            if not vf.is_passive():
                # 失敗した場合 (無損失に近い素子で起きやすい): 元に戻して全体を一様に縮小
                vf.poles, vf.residues, vf.constant_coeff = orig
                fg = np.r_[0.0, np.logspace(np.log10(f[0]) - 3, np.log10(f[-1]) + 2, 6000)]
                Mg = np.array([[vf.get_model_response(i, j, fg) for j in range(self.P)]
                               for i in range(self.P)]).transpose(2, 0, 1)
                smax = np.max(np.linalg.svd(Mg, compute_uv=False))
                scale = 1 / (smax * (1 + 1e-4))
                vf.residues = vf.residues * scale
                vf.constant_coeff = vf.constant_coeff * scale
                warnings.warn(f"留数の摂動で受動性を強制できなかったため、S 全体を {scale:.4f} 倍しました")
        cm = CompactModel(vf, self.P, z0)
        cm.passivity_scale = scale
        return cm

    def write_touchstone(self, filename, theta, f=None, z0=50.0, form="ri"):
        """θ での予測平均を Touchstone (.sNp) に書き出す。write_touchstone(filename, self, theta, ...) と同じ"""
        return write_touchstone(filename, self, theta=theta, f=f, z0=z0, form=form)


class CompactModel:
    """低次の有理モデル (scikit-rf VectorFitting を内包)。状態空間化と SPICE 出力"""

    def __init__(self, vf, P, z0=50.0):
        self.vf, self.P, self.z0 = vf, P, z0
        self.poles = vf.poles
        self.residues = vf.residues
        self.d = vf.constant_coeff
        self.e = vf.proportional_coeff
        self.passivity_scale = 1.0

    @property
    def order(self):
        return int(np.sum(np.where(self.poles.imag == 0, 1, 2)))

    def response(self, f):
        s = 2j * np.pi * np.asarray(f, float)[:, None]
        cplx = self.poles.imag != 0                                 # 実極は 1 回、複素極は共役も足す
        M = (self.d[None, :] + s * self.e[None, :] + (1 / (s - self.poles)) @ self.residues.T
             + (1 / (s - np.conj(self.poles[cplx]))) @ np.conj(self.residues[:, cplx]).T)
        return M.reshape(len(f), self.P, self.P)

    def state_space(self):
        """実数の (A, B, C, D)。S パラメータ (入射波 a → 反射波 b) の状態方程式"""
        if np.any(self.e != 0):
            raise ValueError("比例項 e≠0 は状態空間 (A,B,C,D) では表せません")
        P, I = self.P, np.eye(self.P)
        A, B, C = [], [], []
        for k, p in enumerate(self.poles):
            R = self.residues[:, k].reshape(P, P)
            if p.imag == 0:
                A.append(p.real * I); B.append(I); C.append(R.real)
            else:
                A.append(np.block([[p.real * I, p.imag * I], [-p.imag * I, p.real * I]]))
                B.append(np.vstack([2 * I, 0 * I])); C.append(np.hstack([R.real, R.imag]))
        n = sum(a.shape[0] for a in A)
        AA, i = np.zeros((n, n)), 0
        for a in A:
            m = a.shape[0]; AA[i:i + m, i:i + m] = a; i += m
        return AA, np.vstack(B), np.hstack(C), self.d.reshape(P, P).real

    def write_spice(self, path, name="mlrf_model"):
        """SPICE サブサーキット (scikit-rf の等価回路出力)"""
        self.vf.write_spice_subcircuit_s(path, fitted_model_name=name)
        return path

    def write_touchstone(self, filename, f=None, form="ri"):
        """Touchstone (.sNp) に書き出す。write_touchstone(filename, self, ...) と同じ"""
        return write_touchstone(filename, self, f=f, form=form)


def write_touchstone(filename, model, theta=None, f=None, z0=None, form="ri"):
    """学習結果の S パラメータを Touchstone 形式 (.sNp) のファイルに書き出す

    filename : 出力ファイル名。拡張子 .sNp (N = ポート数) が無ければ付ける。無いディレクトリは作る
    model    : ParametricRationalGP (θ での予測平均を出力) または CompactModel (縮約モデルの応答を出力)
    theta    : model が ParametricRationalGP のとき必須。出力する設計パラメータ (D,)
    f        : 出力する周波数 [Hz]。省略時は GP の学習周波数 / CompactModel の縮約に使った周波数
    z0       : 基準インピーダンス [Ω]。省略時は CompactModel の z0、GP では 50
    form     : 数値の形式 "ri" (実部・虚部), "ma" (振幅・角度), "db" (dB・角度)
    戻り値   : 書き出したファイルのパス
    """
    import os
    if isinstance(model, ParametricRationalGP):
        if theta is None:
            raise ValueError("ParametricRationalGP を書き出すときは theta を指定してください")
        f = model.freqs if f is None else np.asarray(f, float)
        S = model.predict(theta, f, return_std=False)
        z0 = 50.0 if z0 is None else z0
        comment = "mlrf ParametricRationalGP mean, " + ", ".join(
            f"{n}={v:.6g}" for n, v in zip(model.names, np.ravel(np.asarray(theta, float))))
    elif isinstance(model, CompactModel):
        f = model.vf.network.f if f is None else np.asarray(f, float)
        S = model.response(f)
        z0 = model.z0 if z0 is None else z0
        comment = f"mlrf CompactModel, order={model.order}, passivity_scale={model.passivity_scale:.6g}"
    else:
        raise TypeError("model は ParametricRationalGP か CompactModel にしてください "
                        "(StaticNonlinearGP は S パラメータを持たないので Touchstone にできません)")
    if form not in ("ri", "ma", "db"):
        raise ValueError(f"form は 'ri', 'ma', 'db' のいずれかです (受け取った値: {form!r})")
    P = S.shape[-1]
    ext = f".s{P}p"
    path = os.fspath(filename)
    if not path.lower().endswith(ext):
        path += ext
    d, base = os.path.split(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    nw = rf.Network(f=f, s=S, f_unit="Hz", z0=z0, name=base[:-len(ext)], comments=comment)
    nw.write_touchstone(filename=base, dir=d, form=form)            # 拡張子付きで渡す (名前に "." があると skrf が付けないため)
    return os.path.join(d, base)


def write_ltspice(filename, models, names=None, theta=None, zmax=300.0, symbols=True, ref_pin=False,
                  model_file="absolute", **compact_kw):
    """学習結果を LTspice のサブサーキット (.lib) とシンボル (.asy) に書き出す。AC 解析と過渡解析の両方に使える

    filename   : 出力ファイル名。拡張子が無ければ .lib を付ける。無いディレクトリは作る
    models     : 1 つのモデル、またはモデルのリスト (1 ファイルにまとめて書き出す)
        ParametricRationalGP : theta で compact() して書き出す (compact_kw は compact() に渡す)
        CompactModel         : .subckt NAME p1 … pP [ref]  (各ポートの基準インピーダンスは cm.z0)
        StaticNonlinearGP    : .subckt NAME a k            (a→k に流れる電流 I(V(a,k)) の 2 端子素子, 1 入力のみ)
    names      : サブサーキット名 (文字列またはリスト)。省略時は mlrf_lin1, mlrf_nl1, …
    theta      : ParametricRationalGP を書き出すときの設計パラメータ
    zmax       : 非線形素子の sinh の引数の上限 (オーバーフロー防止)
    symbols    : True なら .lib と同じフォルダに <NAME>.asy (LTspice のシンボル) も書き出す
    ref_pin    : True なら線形素子にポートの基準端子 ref を付ける (False ならポートの基準はグラウンド 0)
    model_file : シンボルの ModelFile 属性に書く .lib のパス。"absolute" (絶対パス, 既定) / "name" (ファイル名だけ。
                 .lib を回路図と同じフォルダか LTspice のライブラリ検索パスに置く場合)
    戻り値     : 書き出した .lib のパス (シンボルは <.lib のフォルダ>/<NAME>.asy)

    使い方 (LTspice):
      回路図: .asy を回路図と同じフォルダ (または シンボル検索パス) に置き、部品の追加 (F2) で NAME を選ぶ。
              .lib はシンボルの ModelFile で自動的に読み込まれる (.include 不要)
      ネットリスト: .include <ファイル> を書き、X1 n1 n2 NAME のように使う
    """
    import os
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    if names is None:
        cnt = {"lin": 0, "nl": 0}
        names = []
        for m in models:
            key = "nl" if isinstance(m, StaticNonlinearGP) else "lin"
            cnt[key] += 1
            names.append(f"mlrf_{key}{cnt[key]}")
    elif isinstance(names, str):
        names = [names]
    if len(names) != len(models):
        raise ValueError("names の数が models の数と合いません")
    if model_file not in ("absolute", "name"):
        raise ValueError("model_file は 'absolute' か 'name' です")
    path = os.fspath(filename)
    if not os.path.splitext(path)[1]:
        path += ".lib"
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blocks, syms = [], []
    for m, nm in zip(models, names):
        if isinstance(m, ParametricRationalGP):
            if theta is None:
                raise ValueError("ParametricRationalGP を書き出すときは theta を指定してください")
            cm = m.compact(theta, **compact_kw)
            desc = "ParametricRationalGP at " + ", ".join(
                f"{n}={v:.6g}" for n, v in zip(m.names, np.ravel(np.asarray(theta, float))))
            blocks.append("* " + desc + "\n" + _ltspice_linear(cm, nm, ref_pin))
            syms.append((nm, _asy_linear(cm.P, ref_pin), f"mlrf {cm.P}-port S-parameter model ({desc})"))
        elif isinstance(m, CompactModel):
            blocks.append(_ltspice_linear(m, nm, ref_pin))
            syms.append((nm, _asy_linear(m.P, ref_pin), f"mlrf {m.P}-port S-parameter model (order {m.order})"))
        elif isinstance(m, StaticNonlinearGP):
            blocks.append(_ltspice_nonlinear(m, nm, zmax))
            syms.append((nm, _asy_nonlinear(), "mlrf nonlinear I-V model (GP), current flows a -> k"))
        else:
            raise TypeError(f"書き出せないモデルの型です: {type(m).__name__}")
    with open(path, "w", encoding="ascii", errors="replace", newline="\r\n") as fp:
        fp.write("* LTspice models generated by mlrf.py\n\n" + "\n\n".join(blocks) + "\n")
    if symbols:
        mf = path if model_file == "absolute" else os.path.basename(path)
        for nm, (gfx, pins), desc in syms:
            attrs = [f"SYMATTR Value {nm}", "SYMATTR Prefix X", f"SYMATTR ModelFile {mf}",
                     f"SYMATTR Description {desc}"]
            with open(os.path.join(os.path.dirname(path), nm + ".asy"), "w", encoding="ascii",
                      errors="replace", newline="\r\n") as fp:
                fp.write("\n".join(["Version 4", "SymbolType BLOCK"] + gfx + attrs + pins) + "\n")
    return path


def _asy_linear(P, ref_pin):
    """P ポートのブロック図形: 奇数ポートは左、偶数ポートは右、ref は下"""
    rows = (P + 1) // 2
    top, bot = -32, 32 * (rows - 1) + 32
    gfx = [f"RECTANGLE Normal -64 {top} 64 {bot}",
           f"TEXT 0 {(top + bot) // 2} Center 2 S{P}P",
           f"WINDOW 0 0 {top} Bottom 2", f"WINDOW 3 0 {bot + (24 if ref_pin else 0)} Top 2"]
    pins = []
    for k in range(P):
        x, y, side = (-64 if k % 2 == 0 else 64), 32 * (k // 2), ("LEFT" if k % 2 == 0 else "RIGHT")
        pins += [f"PIN {x} {y} {side} 8", f"PINATTR PinName p{k + 1}", f"PINATTR SpiceOrder {k + 1}"]
    if ref_pin:
        pins += [f"PIN 0 {bot} BOTTOM 8", "PINATTR PinName ref", f"PINATTR SpiceOrder {P + 1}"]
    return gfx, pins


def _asy_nonlinear():
    """2 端子 (a: 上, k: 下) のダイオード風の図形 + 箱"""
    gfx = ["RECTANGLE Normal -8 12 40 52",
           "LINE Normal 0 20 32 20", "LINE Normal 0 20 16 44", "LINE Normal 32 20 16 44",
           "LINE Normal 0 44 32 44", "LINE Normal 16 0 16 12", "LINE Normal 16 52 16 64",
           "TEXT 52 32 Left 1 GP", "WINDOW 0 48 0 Left 2", "WINDOW 3 48 64 Left 2"]
    pins = ["PIN 16 0 NONE 0", "PINATTR PinName a", "PINATTR SpiceOrder 1",
            "PIN 16 64 NONE 0", "PINATTR PinName k", "PINATTR SpiceOrder 2"]
    return gfx, pins


def _ltspice_linear(cm, name, ref_pin=False):
    """CompactModel → R, C, E, F, G だけの等価回路 (正規化した状態空間の実現)

    波の電圧 Aw = (v + z0 i)/2 = √z0·a, Bw = (v - z0 i)/2 = √z0·b,  状態 X = √z0·ω0·x とおくと
        (1/ω0) dX/dt = (A/ω0) X + B Aw,   Bw = (C/ω0) X + D Aw,   ポート: v = z0 i + 2 Bw
    係数がすべて O(1) になり、ノード電圧も V オーダー (SPICE の許容誤差より十分大きい) になる。
    ref_pin=True ならポート電圧 v = V(pk) - V(ref) で、ポート電流は ref から戻る (内部の補助ノードはグラウンド基準)
    """
    A, B, C, D = cm.state_space()
    P, n, z0 = cm.P, A.shape[0], float(cm.z0)
    w0 = float(np.max(np.abs(cm.poles))) if n else 1.0
    g = lambda v: f"{v:.15g}"
    rn = "ref" if ref_pin else "0"
    L = [f"* mlrf CompactModel: {P} ports, order {n}, z0={g(z0)} ohm, passivity_scale={cm.passivity_scale:.6g}",
         f"* pins: p1..p{P}" + (" ref (port reference)" if ref_pin else " (reference = ground 0)"),
         f".subckt {name} " + " ".join(f"p{k + 1}" for k in range(P)) + (" ref" if ref_pin else "")]
    for k in range(1, P + 1):
        L += [f"Vs{k} p{k} q{k} 0",                                  # ポート電流 i (流れ込む向き) を測る
              f"Rz{k} q{k} r{k} {g(z0)}",
              f"Eb{k} r{k} {rn} bw{k} 0 2",                           # v = z0 i + 2 Bw
              f"Ra{k} aw{k} 0 1",                                     # Aw = v/2 + (z0/2) i
              f"Gav{k} 0 aw{k} p{k} {rn} 0.5",
              f"Fai{k} 0 aw{k} Vs{k} {g(z0 / 2)}",
              f"Rb{k} bw{k} 0 1"]                                     # Bw = (C/ω0) X + D Aw
        L += [f"Gbx{k}_{j + 1} 0 bw{k} x{j + 1} 0 {g(C[k - 1, j] / w0)}" for j in range(n) if C[k - 1, j] != 0]
        L += [f"Gba{k}_{l + 1} 0 bw{k} aw{l + 1} 0 {g(D[k - 1, l])}" for l in range(P) if D[k - 1, l] != 0]
    for j in range(n):                                                 # (1/ω0) dX/dt = (A/ω0) X + B Aw
        L.append(f"Cx{j + 1} x{j + 1} 0 {g(1 / w0)}")
        L += [f"Gxx{j + 1}_{i + 1} 0 x{j + 1} x{i + 1} 0 {g(A[j, i] / w0)}" for i in range(n) if A[j, i] != 0]
        L += [f"Gxa{j + 1}_{l + 1} 0 x{j + 1} aw{l + 1} 0 {g(B[j, l])}" for l in range(P) if B[j, l] != 0]
    L.append(f".ends {name}")
    return "\n".join(L)


def _ltspice_nonlinear(dg, name, zmax, digits=12):
    """StaticNonlinearGP → B 素子 1 個の 2 端子サブサーキット (LTspice の式: ^ は使わず積で 2 乗)"""
    if dg.X.shape[1] != 1:
        raise ValueError("LTspice への書き出しは 1 入力の StaticNonlinearGP だけに対応しています")
    g = lambda v: f"{v:.{digits}g}"
    l2 = 2 * dg.ell[0] ** 2
    xs, inv = np.unique(dg.X[:, 0], return_inverse=True)
    ws = np.zeros(len(xs)); np.add.at(ws, inv, dg.sf2 * dg.w)          # 重複点の項をまとめる
    terms = [f"+ {wi:+.{digits}g}*exp(-(V(a,k)-({g(xi)}))*(V(a,k)-({g(xi)}))/{g(l2)})"
             for wi, xi in zip(ws, xs) if wi != 0]
    vmin, vmax = dg.X[:, 0].min(), dg.X[:, 0].max()
    return "\n".join(
        [f"* mlrf StaticNonlinearGP: I(a->k) = I0*sinh(z(V(a,k))), I0={g(dg.I0)}, "
         f"trained V range [{vmin:.4g}, {vmax:.4g}] V (outside: linear extrapolation of z)",
         f".subckt {name} a k",
         f"B1 a k I={g(dg.I0)}*sinh(limit({g(dg.beta[0])}{dg.beta[1]:+.{digits}g}*V(a,k)"]
        + terms + [f"+ , {g(-zmax)}, {g(zmax)}))", f".ends {name}"])


# =====================================================================
# 非線形素子: 静的 I(V) の GP (asinh 変換)
# =====================================================================
class StaticNonlinearGP:
    """I = I0·sinh(z(V)), z ~ GP(1 次の平均関数, ARD-RBF)。指数的な I-V も 1 つの滑らかな関数で扱える

    I0     : 扱う電流範囲の下の方 (例: 1µA) に選ぶ
    bounds : 入力電圧の範囲 [(min, max), ...]。suggest() で候補点を省略したときに使う
    reml   : True なら平均関数の係数を周辺化した制限付き尤度 (REML) で超パラメータを推定
             (既定は False = 最尤。ダイオードの能動学習の比較では最尤の方が精度が良かった)
    """

    Z_MAX = 700.0                                                   # sinh/cosh のオーバーフロー防止

    def __init__(self, I0=1e-6, bounds=None, reml=False):
        self.I0 = I0
        self.bounds = None if bounds is None else np.atleast_2d(np.asarray(bounds, float))
        self.reml = reml

    def fit(self, V, I, optimize=True):
        X = np.asarray(V, float)
        self.X = X[:, None] if X.ndim == 1 else X
        I = np.asarray(I, float).ravel()
        if len(I) != len(self.X):
            raise ValueError(f"V ({len(self.X)} 点) と I ({len(I)} 点) の点数が合いません")
        self.z = np.arcsinh(I / self.I0)
        D = self.X.shape[1]
        span = np.ptp(self.X, 0) + 1e-12
        h0 = np.r_[np.log(span / 3), np.log(np.var(self.z) + 1e-6), np.log(1e-3)]
        prev = getattr(self, "_h", None)
        if optimize:
            bnds = [(np.log(s / 30), np.log(s * 10)) for s in span] + [(-10, 15), (-14, 5)]
            self._h = minimize(self._nll, h0, method="L-BFGS-B", bounds=bnds).x
        elif prev is None or len(prev) != D + 2:                    # 超パラメータが未設定なら初期値で
            self._h = h0
        self._post()
        return self

    def fit_file(self, path, optimize=True, **load_kw):
        """I-V のファイル (CSV / テキスト) から学習する。load_kw は load_iv に渡す (v_col, i_col, i_scale など)"""
        return self.fit(*load_iv(path, **load_kw), optimize=optimize)

    def _parts(self, h):
        D = self.X.shape[1]
        return np.exp(h[:D]), np.exp(h[D]), np.exp(h[D + 1])

    def _H(self, X):
        """平均関数の基底: 定数 + 各入力の 1 次項 (指数領域で log I ≒ V の 1 次式になるのを利用)"""
        return np.hstack([np.ones((len(X), 1)), X])

    def _gls(self, h):
        """コレスキー分解と一般化最小二乗の平均関数係数"""
        ell, sf2, sn2 = self._parts(h)
        K = sf2 * rbf(self.X, self.X, ell) + (sn2 + 1e-10) * np.eye(len(self.z))
        L = cho_factor(K)
        H = self._H(self.X)
        KiH = cho_solve(L, H)
        A = H.T @ KiH
        beta = np.linalg.lstsq(A, KiH.T @ self.z, rcond=None)[0]
        return L, KiH, A, beta

    def _nll(self, h):
        try:
            L, KiH, A, beta = self._gls(h)
        except np.linalg.LinAlgError:
            return 1e10
        r = self.z - self._H(self.X) @ beta
        nll = 0.5 * r @ cho_solve(L, r) + np.sum(np.log(np.diag(L[0])))
        if self.reml:
            nll += 0.5 * np.linalg.slogdet(A)[1]
        return nll

    def _post(self):
        self.ell, self.sf2, self.sn2 = self._parts(self._h)
        self._L, self._KiH, A, self.beta = self._gls(self._h)
        self._Ainv = np.linalg.pinv(A)                              # 平均関数の係数 β の事後共分散 (σ² 込み)
        self.w = cho_solve(self._L, self.z - self._H(self.X) @ self.beta)   # 平均 = β·h(v) + Σ w_i k(v, X_i)

    def _X(self, v):
        v = np.asarray(v, float)
        return v[:, None] if v.ndim == 1 and self.X.shape[1] == 1 else np.atleast_2d(v)

    def z_mean(self, v, grad=False):
        Xv = self._X(v)
        k = self.sf2 * rbf(Xv, self.X, self.ell)                     # (n, N)
        mz = self._H(Xv) @ self.beta + k @ self.w
        if not grad:
            return mz
        dk = -k[:, :, None] * (Xv[:, None, :] - self.X[None, :, :]) / self.ell ** 2
        return mz, np.einsum("nid,i->nd", dk, self.w) + self.beta[1:][None, :]

    def _cov_cross(self, Xa, Xb):
        """事後共分散 (平均関数の係数の不確かさを含む universal kriging)"""
        ka, kb = self.sf2 * rbf(Xa, self.X, self.ell), self.sf2 * rbf(Xb, self.X, self.ell)
        Ra = self._H(Xa).T - self._KiH.T @ ka.T                      # (p, na)
        Rb = self._H(Xb).T - self._KiH.T @ kb.T
        return (self.sf2 * rbf(Xa, Xb, self.ell) - ka @ cho_solve(self._L, kb.T)
                + Ra.T @ self._Ainv @ Rb)

    def z_var(self, v, mean_uncertainty=True):
        """z の事後分散。mean_uncertainty=True なら平均関数の係数 β の不確かさも含める (外挿で大きくなる)"""
        Xv = self._X(v)
        k = self.sf2 * rbf(Xv, self.X, self.ell)
        var = self.sf2 - np.sum(k * cho_solve(self._L, k.T).T, 1)
        if mean_uncertainty:
            Rm = self._H(Xv).T - self._KiH.T @ k.T
            var = var + np.sum(Rm * (self._Ainv @ Rm), 0)
        return np.clip(var, 0, None)

    def current(self, v):
        """平均の電流と微分コンダクタンス dI/dV。Newton 法でそのまま使える

        1 入力なら dI/dV は (n,)、多入力なら各入力についての偏微分 (n, D)
        """
        mz, dz = self.z_mean(v, grad=True)
        mz = np.clip(mz, -self.Z_MAX, self.Z_MAX)
        g = self.I0 * np.cosh(mz)[:, None] * dz
        return self.I0 * np.sinh(mz), (g[:, 0] if g.shape[1] == 1 else g)

    def current_band(self, v, k=2.0):
        mz, sz = self.z_mean(v), np.sqrt(self.z_var(v))
        return self.I0 * np.sinh(mz - k * sz), self.I0 * np.sinh(mz + k * sz)

    def suggest(self, cand=None, n_cand=401):
        """能動学習: 予測分散 × tanh²(z) が最大の候補点

        tanh²(z) は |I| ≫ I0 で 1、|I| ≪ I0 で (I/I0)² になる重み。
        回路動作にほとんど効かない微小電流領域 (逆方向など) を測りすぎるのを防ぐ。
        cand を省略すると bounds の範囲の等間隔点 (1 入力) / Sobol 点 (多入力) を使う。
        β の不確かさは含めない (含めると範囲の端ばかり選び、精度が落ちた)。
        """
        if cand is None:
            if self.bounds is None:
                raise ValueError("cand を渡すか、コンストラクタで bounds を指定してください")
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            cand = (np.linspace(lo[0], hi[0], n_cand) if len(lo) == 1 else
                    qmc.scale(qmc.Sobol(d=len(lo), seed=0).random(n_cand), lo, hi))
        cand = np.asarray(cand)
        return cand[np.argmax(self.z_var(cand, mean_uncertainty=False) * np.tanh(self.z_mean(cand)) ** 2)]

    def sample_functions(self, vgrid, n, seed=0):
        """事後分布から I(V) の関数サンプルを n 本 (1 入力)

        戻り値は SplineSamples。samples[j](v) -> (i, di/dv) で 1 本ずつ使えるほか、
        samples(v) (v の形 (n,)) で「j 番目のサンプルを v[j] で評価」をまとめて計算できる
        (PortCircuitSimulator.run_ensemble 用)。グリッドの外は z を 1 次で外挿する。
        """
        if self.X.shape[1] != 1:
            raise ValueError("sample_functions は 1 入力のモデルにだけ対応しています")
        vg = np.unique(np.asarray(vgrid, float))
        Xg = vg[:, None]
        mean = self._H(Xg) @ self.beta + self.sf2 * rbf(Xg, self.X, self.ell) @ self.w
        cov = self._cov_cross(Xg, Xg)
        e, U = np.linalg.eigh(0.5 * (cov + cov.T))                  # cholesky より頑健な平方根
        Lc = U * np.sqrt(np.clip(e, 0, None))
        Z = mean[:, None] + Lc @ np.random.default_rng(seed).standard_normal((len(vg), n))
        return SplineSamples(vg, Z, self.I0, self.Z_MAX)

    def spice_bsource(self, name, n_plus, n_minus, digits=10):
        """SPICE の B 素子 (電流源) として出力。平均関数をそのまま式にする (1 入力)

        同じ電圧で重複測定した点の項はまとめる (大きな正負の係数の打ち消し合いを避ける)
        """
        if self.X.shape[1] != 1:
            raise ValueError("spice_bsource は 1 入力のモデルにだけ対応しています")
        vn = f"V({n_plus},{n_minus})"
        l2 = 2 * self.ell[0] ** 2
        xs, inv = np.unique(self.X[:, 0], return_inverse=True)
        ws = np.zeros(len(xs)); np.add.at(ws, inv, self.sf2 * self.w)
        terms = "".join(f"{wi:+.{digits}g}*exp(-(({vn}-({xi:.{digits}g}))^2)/{l2:.{digits}g})"
                        for wi, xi in zip(ws, xs) if wi != 0)
        lin = f"{self.beta[0]:.{digits}g}{self.beta[1]:+.{digits}g}*{vn}"
        return f"{name} {n_plus} {n_minus} I={{{self.I0:.{digits}g}*sinh({lin}{terms})}}"


class SplineSamples:
    """GP 事後サンプル z_j(v) (グリッド上の 3 次スプライン + グリッド外は 1 次外挿) の集合"""

    def __init__(self, vg, Z, I0, z_max=700.0):
        self.vg, self.I0, self.z_max = vg, I0, z_max
        cs = CubicSpline(vg, Z, axis=0)
        self.c = cs.c                                                # (4, G-1, n)
        d = cs.derivative()
        self.z_lo, self.z_hi = Z[0], Z[-1]
        self.s_lo, self.s_hi = d(vg[0]), d(vg[-1])

    def __len__(self):
        return self.c.shape[2]

    def _eval(self, v, j):
        vg = self.vg
        vc = np.clip(v, vg[0], vg[-1])
        k = np.clip(np.searchsorted(vg, vc, side="right") - 1, 0, len(vg) - 2)
        x = vc - vg[k]
        c = self.c[:, k, j]
        z = ((c[0] * x + c[1]) * x + c[2]) * x + c[3]
        dz = (3 * c[0] * x + 2 * c[1]) * x + c[2]
        lo, hi = v < vg[0], v > vg[-1]
        z = np.where(lo, self.z_lo[j] + self.s_lo[j] * (v - vg[0]),
                     np.where(hi, self.z_hi[j] + self.s_hi[j] * (v - vg[-1]), z))
        dz = np.where(lo, self.s_lo[j], np.where(hi, self.s_hi[j], dz))
        z = np.clip(z, -self.z_max, self.z_max)
        return self.I0 * np.sinh(z), self.I0 * np.cosh(z) * dz

    def __call__(self, v):
        """v: (n,) → j 番目のサンプルを v[j] で評価した (i, di/dv)"""
        v = np.asarray(v, float)
        return self._eval(v, np.arange(len(self)))

    def __getitem__(self, j):
        if not -len(self) <= j < len(self):
            raise IndexError(j)
        j = j % len(self)
        return lambda v: self._eval(np.asarray(v, float), j)

    def __iter__(self):
        return (self[j] for j in range(len(self)))


# =====================================================================
# 学習データの読み込み (Touchstone / CSV・テキスト)
# =====================================================================
def _expand_files(files, exts):
    import glob, os
    if isinstance(files, (str, os.PathLike)):
        p = os.fspath(files)
        if os.path.isdir(p):
            out = sorted(q for q in glob.glob(os.path.join(p, "*")) if q.lower().endswith(exts))
        elif any(ch in p for ch in "*?["):
            out = sorted(glob.glob(p))
        else:
            out = [p]
    else:
        out = [os.fspath(q) for q in files]
    if not out:
        raise FileNotFoundError(f"ファイルが見つかりません: {files}")
    return out


def load_touchstone(files, thetas=None, pattern=None, f=None, z0=50.0):
    r"""θ ごとの Touchstone ファイル群を読み込み、ParametricRationalGP.fit 用の (thetas, f, S) を返す

    files   : ファイルパスのリスト、ワイルドカード ("data/*.s2p")、またはフォルダ (中の .sNp / .ts を名前順)
    thetas  : 各ファイルの θ (Nθ, D)。files (展開後の名前順) と同じ順に並べる
    pattern : thetas を省略したとき、ファイル名から θ を取り出す正規表現。各グループが θ の 1 成分になる
              例: r"L([\d.]+)nH_Cc([\d.]+)pF" → "filter_L2.2nH_Cc0.7pF.s2p" から θ = (2.2, 0.7)
    f       : 学習に使う周波数 [Hz]。省略時:
              全ファイルの周波数が同じならそれを使う。違う場合は 1 つ目のファイルの周波数のうち
              全ファイルに共通する範囲を使い、他のファイルはそこへ補間する
    z0      : 基準インピーダンス [Ω]。ファイルの z0 が違えばこの値に変換 (renormalize) する
    f ≤ 0 の点 (DC) は自動で除く (Szegö カーネルでは使えないため)
    戻り値  : thetas (Nθ, D), f (F,), S (Nθ, F, P, P)
    """
    import os, re
    paths = _expand_files(files, tuple(f".s{k}p" for k in range(1, 10)) + (".ts",))
    if thetas is None:
        if pattern is None:
            raise ValueError("thetas (各ファイルの θ) か、ファイル名から θ を取り出す pattern を指定してください")
        rx, th = re.compile(pattern), []
        for p in paths:
            m = rx.search(os.path.basename(p))
            if m is None or not m.groups():
                raise ValueError(f"pattern {pattern!r} がファイル名 {os.path.basename(p)!r} に一致しません")
            th.append([float(x) for x in m.groups()])
        thetas = np.array(th)
    thetas = np.asarray(thetas, float).reshape(len(paths), -1)
    nets = []
    for p in paths:
        nw = rf.Network(p)
        if not np.allclose(nw.z0, z0):
            nw.renormalize(z0)
        nets.append(nw)
    P = nets[0].nports
    if any(n.nports != P for n in nets):
        raise ValueError("ポート数が違うファイルが混ざっています")
    if f is None:
        f0 = nets[0].f
        if all(len(n.f) == len(f0) and np.allclose(n.f, f0) for n in nets):
            f = f0
        else:
            lo, hi = max(n.f.min() for n in nets), min(n.f.max() for n in nets)
            f = f0[(f0 >= lo) & (f0 <= hi)]
            warnings.warn(f"ファイルごとに周波数点が違うため、{lo/1e9:.4g}–{hi/1e9:.4g} GHz の {len(f)} 点に補間しました")
    f = np.asarray(f, float)
    if np.any(f <= 0):
        warnings.warn(f"f ≤ 0 の点 ({np.sum(f <= 0)} 点) を除きました")
        f = f[f > 0]
    S = []
    for n in nets:
        if len(n.f) == len(f) and np.allclose(n.f, f):
            S.append(n.s)
        else:
            if f.min() < n.f.min() * (1 - 1e-12) or f.max() > n.f.max() * (1 + 1e-12):
                raise ValueError(f"{n.name}: 指定の周波数がファイルの周波数範囲の外です (外挿はしません)")
            S.append(n.interpolate(rf.Frequency.from_f(f, unit="hz"), kind="cubic").s)
    return thetas, f, np.array(S)


def load_iv(path, v_col=0, i_col=1, delimiter=None, v_scale=1.0, i_scale=1.0):
    """I-V 測定データ (CSV / TSV / 空白区切りのテキスト) を読み込み、StaticNonlinearGP.fit 用の (V, I) を返す

    path     : ファイルのパス (LTspice の波形エクスポート .txt もそのまま読める)
    v_col    : 電圧の列。列番号 (0 始まり) またはヘッダの列名。リストにすると多入力 (N, D)
    i_col    : 電流の列。列番号またはヘッダの列名
    delimiter: 区切り文字。None なら ',' / タブ / ';' / 空白 から自動判定
    v_scale, i_scale : 単位換算の倍率 (例: 電流の列が mA なら i_scale=1e-3)
    数値として読めない行 (ヘッダ)、空行、'#' '!' '*' で始まる行は飛ばす。列名には最後のヘッダ行を使う
    戻り値   : V (N,) または (N, D), I (N,)
    """
    for enc in ("utf-8-sig", "cp932", "latin-1"):
        try:
            with open(path, encoding=enc) as fp:
                lines = fp.read().splitlines()
            break
        except UnicodeDecodeError:
            continue
    lines = [l.strip() for l in lines if l.strip() and not l.lstrip().startswith(("#", "!", "*"))]
    if delimiter is None:
        sample = lines[-1] if lines else ""
        delimiter = next((d for d in (",", "\t", ";") if d in sample), None)
    split = (lambda l: [t.strip().strip('"').strip("'") for t in l.split(delimiter)]) if delimiter \
        else (lambda l: l.split())
    header, rows = None, []
    for l in lines:
        tok = split(l)
        try:
            rows.append([float(t) for t in tok])
        except ValueError:
            if not rows:
                header = tok                                         # データ前の読めない行 = ヘッダ
            else:
                warnings.warn(f"数値として読めない行を飛ばしました: {l[:60]!r}")
    if not rows:
        raise ValueError(f"{path}: 数値データが見つかりません")
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows])

    def col(c):
        if isinstance(c, str):
            if header is None:
                raise ValueError("列名で指定するにはファイルにヘッダ行が必要です")
            names = [h.lower() for h in header]
            if c.lower() not in names:
                raise ValueError(f"列名 {c!r} がありません (ヘッダ: {header})")
            c = names.index(c.lower())
        return data[:, c]

    V = (np.column_stack([col(c) for c in v_col]) if isinstance(v_col, (list, tuple)) else col(v_col)) * v_scale
    return V, col(i_col) * i_scale


# =====================================================================
# 能動学習 (線形・パラメトリック)
# =====================================================================
def active_learning(simulate, bounds, n_init=4, n_iter=8, n_cand=2048, seed=0,
                    model=None, refit_hyper_every=2, mode="imse", verbose=True):
    """simulate(theta) -> (f, S[F,P,P]) を呼んで、評価値 (mode="imse"/"var") 最大の θ を逐次追加する"""
    bounds = np.asarray(bounds, float)
    D = len(bounds)
    lhs = qmc.LatinHypercube(d=D, seed=seed)
    th = qmc.scale(lhs.random(n_init), bounds[:, 0], bounds[:, 1])
    cand = qmc.scale(qmc.Sobol(d=D, seed=seed + 1).random(n_cand), bounds[:, 0], bounds[:, 1])
    f = None

    def run(t):
        nonlocal f
        fi, Si = simulate(t)
        fi = np.asarray(fi, float)
        if f is None:
            f = fi
        elif fi.shape != f.shape or not np.allclose(fi, f, rtol=1e-12, atol=0):
            raise ValueError("simulate() は全ての θ で同じ周波数点を返す必要があります (クロネッカー構造)")
        return Si

    S = [run(t) for t in th]
    gp = model or ParametricRationalGP(bounds)
    gp.fit(th, f, np.array(S))
    hist = []
    label = "IMSE 削減量" if mode == "imse" else "平均予測分散"
    for it in range(n_iter):
        acq = gp.acquisition(cand, mode=mode)
        j = int(np.argmax(acq))
        tnew = cand[j]; cand = np.delete(cand, j, 0)
        th = np.vstack([th, tnew]); S.append(run(tnew))
        gp.fit(th, f, np.array(S), optimize=((it + 1) % refit_hyper_every == 0))
        hist.append((tnew, float(acq[j])))
        if verbose:
            print(f"  AL {it+1:2d}: θ={np.round(tnew, 4)}  {label}={acq[j]:.3e}")
    return gp, th, hist


# =====================================================================
# 高速シミュレータ: 線形 N ポート状態空間 + ポート終端
# =====================================================================
class PortCircuitSimulator:
    """S パラメータ状態空間 (ẋ=Ax+Ba, b=Cx+Da; 基準 z0) に終端をつないだ回路の過渡解析

    terms : ポートごとの終端 (D の行数と同じ数)
        ("source", vs)                          内部抵抗 z0 の電圧源 vs(t)
        ("source", dict(v=vs, R=Rs))            内部抵抗 Rs の電圧源 (Rs=0 で理想電圧源)
        ("load", dict(R=, C=, nl=fun or None))  R ∥ C ∥ 非線形素子 (fun(v)->(i, di/dv))。
                                                R, C は省略可 (R=∞, C=0)。C=0 は代数方程式として扱う
    アルゴリズム: 台形則 (A 安定、C=0 の行は後退オイラー) + Newton 法。
        ヤコビアン = (定数の線形部分) + (非線形ポート数ランクの更新) なので、
        線形部分の LU は 1 回だけ分解し、各反復は前進後退代入 + 小さな nL×nL の解のみ。
    状態 z = [x; v]  (x: DUT の状態, v: 負荷ポートの電圧)
        diag(1, C) ż = M z + N e(t) - P i_nl(v)
    """

    def __init__(self, A, B, C, D, z0, terms):
        A, B, C, D = (np.asarray(m, float) for m in (A, B, C, D))
        P, n = D.shape[0], A.shape[0]
        if len(terms) != P:
            raise ValueError(f"terms の数 {len(terms)} がポート数 {P} と合いません")
        self.A, self.B, self.C, self.D, self.z0 = A, B, C, D, z0
        self.terms = terms
        rz = np.sqrt(z0)
        gam = np.zeros(P)                                           # a = Γ b + q (ポートごとの終端の反射係数)
        self.iS, self.iL, self.src, src_scale = [], [], [], []
        Rinv, Cl, self.nl_default = [], [], []
        for k, (kind, spec) in enumerate(terms):
            if kind == "source":
                vs, Rs = (spec, z0) if callable(spec) else (spec["v"], spec.get("R", z0))
                if Rs < 0:
                    raise ValueError("電圧源の内部抵抗は 0 以上にしてください")
                gam[k] = (Rs - z0) / (Rs + z0)
                self.iS.append(k); self.src.append(vs); src_scale.append(rz / (Rs + z0))
            elif kind == "load":
                R, Ck = spec.get("R", np.inf), spec.get("C", 0.0)
                if R <= 0 or Ck < 0:
                    raise ValueError("負荷は R > 0, C ≥ 0 にしてください")
                gam[k] = -1.0                                       # a_L = v/√z0 - b_L
                self.iL.append(k); Rinv.append(1 / R); Cl.append(Ck); self.nl_default.append(spec.get("nl"))
            else:
                raise ValueError(f"未知の終端の種類: {kind!r}")
        iS, iL = self.iS, self.iL
        nS, nL = len(iS), len(iL)
        Se = np.zeros((P, nS)); Se[iS, np.arange(nS)] = 1
        Sv = np.zeros((P, nL)); Sv[iL, np.arange(nL)] = 1 / rz
        Wm = np.eye(P) - gam[:, None] * D
        if np.linalg.cond(Wm) > 1e12:
            raise ValueError("I - ΓD が特異です。DUT がポートで高周波短絡 (D_LL の固有値が -1、"
                             "例: ポートに直接つながるシャント容量) か、理想電圧源が短絡を駆動しています。"
                             "負荷側に直列抵抗を入れるなどしてください")
        W = np.linalg.inv(Wm)                                       # a = Ax x + Av v + Ae e
        self.Ax, self.Av, self.Ae = W @ (gam[:, None] * C), W @ Sv, W @ Se
        # 負荷 KCL: C v̇ = v/z0 - 2 a_L/√z0 - v/R - I_nl
        M = np.zeros((n + nL, n + nL)); N = np.zeros((n + nL, nS))
        M[:n, :n] = A + B @ self.Ax; M[:n, n:] = B @ self.Av; N[:n] = B @ self.Ae
        M[n:, :n] = -(2 / rz) * self.Ax[iL]
        M[n:, n:] = np.eye(nL) / z0 - (2 / rz) * self.Av[iL] - np.diag(Rinv)
        N[n:] = -(2 / rz) * self.Ae[iL]
        self.M, self.N, self.n, self.nL = M, N, n, nL
        self.mass = np.r_[np.ones(n), Cl]
        self.src_scale = np.array(src_scale)
        self.Pm = np.zeros((n + nL, nL)); self.Pm[n:] = np.eye(nL)

    # ------------------------------------------------------------- 補助
    def _sources(self, t):
        """入射波の源 e(t) (len(t), nS)"""
        e = np.zeros((len(t), len(self.iS)))
        for k, vs in enumerate(self.src):
            e[:, k] = [vs(ti) for ti in t]
        return e * self.src_scale

    @staticmethod
    def _nl_eval(nl, v):
        """v: (nL, Ns) → (i, g) 各 (nL, Ns)"""
        i, g = np.zeros_like(v), np.zeros_like(v)
        for k, fun in enumerate(nl):
            if fun is not None:
                ik, gk = fun(v[k])
                i[k], g[k] = ik, gk
        return i, g

    def _ports(self, z, e):
        """全ポートの電圧 (Ns, P)"""
        n = self.n
        x, v = z[:n], z[n:]
        a = self.Ax @ x + self.Av @ v + (self.Ae @ e)[:, None]
        b = self.C @ x + self.D @ a
        return (np.sqrt(self.z0) * (a + b)).T

    # ------------------------------------------------------------- DC 動作点
    def dc_op(self, nl_funcs=None, t0=0.0, n_members=1, gmin=1e-12, tol=1e-10,
              max_iter=100, dv_max=0.1, n_source_steps=10):
        """0 = M z + N e(t0) - P i_nl(v) を Newton 法で解く (失敗したら電源ステッピング)。z: (n+nL, Ns)"""
        nl = self.nl_default if nl_funcs is None else nl_funcs
        n, nL, Ns = self.n, self.nL, n_members
        e0 = self._sources(np.array([t0]))[0]
        Mg = self.M.copy(); Mg[n:, n:] -= gmin * np.eye(nL)        # 浮いたノード対策の微小コンダクタンス

        def newton(z, scale):
            for _ in range(max_iter):
                i, g = self._nl_eval(nl, z[n:])
                F = Mg @ z + (self.N @ (scale * e0))[:, None] - self.Pm @ i
                J = np.repeat(Mg[None], Ns, 0)
                J[:, n:, n:] -= g.T[:, :, None] * np.eye(nL)[None]
                dz = np.linalg.solve(J, F.T[:, :, None])[:, :, 0].T
                if nL:
                    step = np.max(np.abs(dz[n:]), 0)
                    dz *= np.where(step > dv_max, dv_max / np.maximum(step, 1e-300), 1.0)
                z = z - dz
                if np.all(np.abs(dz) <= tol * (1 + np.abs(z))):
                    return z, True
            return z, False

        z, ok = newton(np.zeros((n + nL, Ns)), 1.0)
        if not ok:                                                  # 電源ステッピング
            z = np.zeros((n + nL, Ns))
            for sc in np.linspace(0, 1, n_source_steps + 1)[1:]:
                z, ok = newton(z, sc)
            if not ok:
                warnings.warn("DC 動作点の Newton 法が収束しませんでした")
        return z

    # ------------------------------------------------------------- 過渡解析
    def run(self, t_end, h, nl_funcs=None, tol=1e-9, max_newton=50, dv_max=0.1, init="zero"):
        """過渡解析。戻り値 t (T,), V (T, P) [全ポートの電圧]

        nl_funcs : 負荷ごとの非線形関数のリスト (省略時は terms の nl)
        init     : "zero" (全状態 0), "dc" (t=0 の DC 動作点), または z の初期値 (n+nL,)
        """
        nl = self.nl_default if nl_funcs is None else nl_funcs
        t, V = self._integrate(t_end, h, nl, 1, init, tol, max_newton, dv_max)
        return t, V[:, 0]

    def run_ensemble(self, t_end, h, nl_batch, n_members, tol=1e-9, max_newton=50, dv_max=0.1,
                     init="zero"):
        """n_members 個の回路 (非線形素子だけが違う) をまとめて過渡解析する。LU 分解を共有

        nl_batch : 負荷ごとの関数のリスト。fun(v) は v の形 (n_members,) で、
                   j 番目の要素を j 番目のメンバーの素子で評価した (i, di/dv) を返す
                   (StaticNonlinearGP.sample_functions の戻り値がそのまま使える)
        戻り値 t (T,), V (n_members, T, P)
        """
        t, V = self._integrate(t_end, h, nl_batch, n_members, init, tol, max_newton, dv_max)
        return t, V.transpose(1, 0, 2)

    def _integrate(self, t_end, h, nl, Ns, init, tol, max_newton, dv_max):
        n, nL, M, N, Pm = self.n, self.nL, self.M, self.N, self.Pm
        if len(nl) != nL:
            raise ValueError(f"非線形関数の数 {len(nl)} が負荷の数 {nL} と合いません")
        w = np.where(self.mass > 0, 0.5, 1.0)                       # 台形則 / C=0 の行は後退オイラー
        Em = np.diag(self.mass)
        K0 = Em - h * w[:, None] * M
        Kp = Em + h * (1 - w)[:, None] * M
        E1, E0 = (h * w)[:, None] * Pm, (h * (1 - w))[:, None] * Pm
        N1, N0 = (h * w)[:, None] * N, (h * (1 - w))[:, None] * N
        lu = lu_factor(K0)                                          # 1 回だけ分解
        KE = lu_solve(lu, E1); PKE = KE[n:]                          # K0⁻¹E, Pᵀ K0⁻¹ E
        steps = int(round(t_end / h))
        t = np.arange(steps + 1) * h
        e = self._sources(t)
        if isinstance(init, str) and init == "dc":
            z = self.dc_op(nl, t0=t[0], n_members=Ns)
        elif isinstance(init, str) and init == "zero":
            z = np.zeros((n + nL, Ns))
        else:
            z = np.repeat(np.asarray(init, float).reshape(-1, 1), Ns, 1)
        V = np.zeros((steps + 1, Ns, self.D.shape[0]))
        i0, _ = self._nl_eval(nl, z[n:])
        V[0] = self._ports(z, e[0])
        n_newton = n_fail = 0
        I_L = np.eye(nL)[None]
        for s in range(steps):
            rhs = Kp @ z + (N0 @ e[s] + N1 @ e[s + 1])[:, None] - E0 @ i0
            z1 = z.copy()
            active = np.ones(Ns, bool)
            for it in range(max_newton):
                i1, g1 = self._nl_eval(nl, z1[n:])
                r = K0 @ z1 + E1 @ i1 - rhs
                dz = lu_solve(lu, r)
                if nL:                                               # Woodbury (ランク nL の更新)
                    Jm = I_L + g1.T[:, :, None] * PKE[None]
                    cr = np.linalg.solve(Jm, (g1 * dz[n:]).T[:, :, None])[:, :, 0].T
                    dz -= KE @ cr
                    step = np.max(np.abs(dz[n:]), 0)
                    dz *= np.where(step > dv_max, dv_max / np.maximum(step, 1e-300), 1.0)  # 電圧ステップ制限
                    conv = np.max(np.abs(dz[n:]), 0) < tol * (1 + np.max(np.abs(z1[n:] - dz[n:]), 0))
                else:
                    conv = np.ones(Ns, bool)                         # 線形回路は 1 回で解ける
                dz[:, ~active] = 0
                z1 -= dz
                n_newton += int(active.sum())
                active &= ~conv
                if not active.any():
                    break
            n_fail += int(active.sum())
            z = z1
            i0, _ = self._nl_eval(nl, z[n:])
            V[s + 1] = self._ports(z, e[s + 1])
        self.n_newton, self.n_fail = n_newton, n_fail
        if n_fail:
            warnings.warn(f"Newton 法が {n_fail} 回 (ステップ×メンバー) 収束しませんでした。"
                          "刻み h を小さくするか max_newton を増やしてください")
        return t, V
