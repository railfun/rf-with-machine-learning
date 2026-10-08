# mlrf — ガウス過程による回路素子モデルと高速シミュレータ

`mlrf.py` は、線形素子(S パラメータ)と非線形素子(静的な I-V)をガウス過程(GP)で学習します。学習したモデルは、低次の状態空間・SPICE ネットリストへの書き出しや、Python 上での過渡解析に使えます。

| 機能 | クラス / 関数 |
|---|---|
| 線形素子 S(θ, f) の学習・予測・縮約 | `ParametricRationalGP` → `CompactModel` |
| 非線形素子 I(V) の学習・予測・事後サンプル | `StaticNonlinearGP` → `SplineSamples` |
| 線形素子の能動学習(次に測る θ の自動選択) | `active_learning` |
| N ポート線形素子+終端(電源・非線形負荷)の過渡解析 | `PortCircuitSimulator` |
| 学習結果の S パラメータを Touchstone(.sNp)に書き出す | `write_touchstone` |
| 学習結果を LTspice 用モデル(.lib のサブサーキット+ .asy シンボル)に書き出す | `write_ltspice` |
| 学習データ(Touchstone 群・I-V の CSV/テキスト)を読み込む | `load_touchstone`, `load_iv` |

依存:numpy、scipy、scikit-rf(2.0 以降)。ngspice は任意です(書き出したモデルの確認用)。

```python
from mlrf import (ParametricRationalGP, StaticNonlinearGP, active_learning,
                  PortCircuitSimulator, write_touchstone, write_ltspice,
                  load_touchstone, load_iv)
```

---

## 典型的な流れ

```python
# 0) 測定/シミュレーション済みのファイルから学習する場合 (§9)
gp = ParametricRationalGP([(1.5, 3.0), (0.3, 1.2)]).fit_touchstone(
        "data/*.s2p", pattern=r"L([\d.]+)nH_Cc([\d.]+)pF")       # θ をファイル名から取り出す
dg = StaticNonlinearGP(I0=1e-6).fit_file("diode_iv.csv", v_col=0, i_col=1)

# 1) 線形素子: θ を変えて S パラメータを集め、GP を学習(能動学習)
gp, thetas, hist = active_learning(simulate, bounds=[(1.5, 3.0), (0.3, 1.2)], n_iter=12)
gp.calibrate_std()                         # 予測 σ の較正 (任意)
M, sd = gp.predict([2.2, 0.7], f)          # 任意の θ での S と 1σ

# 2) θ* で低次の有理モデルに縮約 → 状態空間 / SPICE
cm = gp.compact([2.2, 0.7])
A, B, C, D = cm.state_space()
cm.write_spice("filter_model.sp", name="filt")
write_touchstone("filter_pred", gp, theta=[2.2, 0.7])  # → filter_pred.s2p
write_ltspice("filter.lib", cm, names="filt")          # LTspice 用モデル (.lib + filt.asy)

# 3) 非線形素子: I-V 測定から GP を学習
dg = StaticNonlinearGP(I0=1e-6).fit(V, I)
print(dg.spice_bsource("BD1", "p2", "0"))  # SPICE の B 素子として書き出し

# 4) 過渡解析: ポート1 = 電源, ポート2 = R ∥ C ∥ ダイオード
sim = PortCircuitSimulator(A, B, C, D, 50.0,
        [("source", lambda t: 2.0 * np.sin(2 * np.pi * 2e9 * t)),
         ("load", dict(R=50.0, C=0.5e-12, nl=dg.current))])
t, Vp = sim.run(t_end=8e-9, h=2e-12)       # Vp[:, k] = ポート k の電圧

# 5) 非線形素子の不確かさを波形まで伝播
samples = dg.sample_functions(np.linspace(-1.2, 1.2, 241), n=20)
t, W = sim.run_ensemble(8e-9, 2e-12, [samples], len(samples))   # W: (20, T, P)
```

通しの実行例は `demo_mlrf.py`、動作確認は `test_mlrf.py` を参照してください。

---

## 1. `ParametricRationalGP` — 線形素子(パラメトリック S パラメータ)

S(θ, f) を学習します。周波数方向には Szegö カーネルを使うので、予測は常に「安定な極・実係数の有理関数」になります。θ(設計パラメータやバイアス)方向は ARD-RBF カーネルです。

### コンストラクタ
```python
gp = ParametricRationalGP(bounds, names=None, per_output_scale=True, noise_bounds=(1e-9, 1e-1))
```
| 引数 | 意味 |
|---|---|
| `bounds` | θ の範囲 `[(min, max), ...]`(D 次元)。θ を [0,1] に正規化するのに使う |
| `names` | θ の名前のリスト(任意) |
| `per_output_scale` | `True`:信号分散 σ² を S_ij ごとに推定する(既定)。`False`:全 S_ij で共通(旧版の挙動) |
| `noise_bounds` | 雑音比 λ=σ_n²/σ² の探索範囲。λ が下限に張り付いて雑音を補間してしまうときは、下限を上げる |

### `fit(thetas, freqs, S, optimize=True, hyper_stride=2, x0=None, n_restarts=0, seed=0)`
学習データを渡して GP を作ります。戻り値は `self` です。
| 引数 | 意味 |
|---|---|
| `thetas` | `(Nθ, D)` 学習した θ |
| `freqs` | `(F,)` 周波数 [Hz]。**全サンプルで共通**、かつ **f > 0** |
| `S` | `(Nθ, F, P, P)` 複素 S パラメータ |
| `optimize` | `True` なら超パラメータを L-BFGS-B で最適化する。`False` なら前回の値(または `x0`)をそのまま使う |
| `hyper_stride` | 超パラメータの推定は周波数を間引いて行う(2 なら1点おき) |
| `x0` | 超パラメータの初期値 `[log α, log ℓ×D, log λ, log c0]` |
| `n_restarts` | ランダムな初期点を追加する数(局所解の対策) |

学習後の超パラメータは `gp.hyper` で見られます(`alpha_Hz`、`lengthscale`、`noise_ratio`、`sigma2 (P,P)`、`c0`)。

### `predict(theta, f, return_std=True)`
任意の θ・周波数で予測します。
- 戻り値は `M`(`(Ft, P, P)` 複素 S)と `sd`(`(Ft, P, P)` S_ij ごとの 1σ、Re と Im の分散の和の平方根、雑音を含まない)。
- `return_std=False` なら `M` だけを返します。
- 複素誤差が正規分布に従うなら、|誤差| ≤ 2σ に入る確率は約 98.2% です。

### `calibrate_std(apply=True)`
θ 方向の leave-one-out で σ の倍率 κ を推定し、以後の `predict` の σ に掛けます。戻り値は κ です。θ のサンプルが3点以上必要です。能動学習の後に1回呼ぶのがおすすめです。

### `rational(theta)`
θ での有理モデル `(poles, residues, d)` を返します。
- `poles`:Im≥0 の側の極 [rad/s]
- `residues`:`(P², F)`
- `d`:定数項 `(P²,)`
- 共役の極と留数は自動で補われます。

### `acquisition(cand, f=None, mode="imse", n_ref=256, seed=0)`
候補 θ `cand (Nc, D)` の能動学習の評価値を返します。大きいほど次に測る価値が高い点です。
- `"imse"`:θ 空間全体の予測分散の減少量。平均誤差を下げやすい。
- `"var"`:その点での周波数平均の予測分散。最大誤差を下げやすい。

### `compact(theta, f=None, n_dense=401, enforce_passivity=True, z0=50.0)`
θ での平均応答を scikit-rf の VectorFitting で低次の有理モデルに縮約し、受動性を強制します。戻り値は `CompactModel` です。
- 留数の摂動で受動性にできなかった場合は、S 全体を一様に縮小し、警告を出します。縮小倍率は `cm.passivity_scale` に残ります。
- `f` を省略すると、学習帯域を `n_dense` 点で使います。

### `write_touchstone(filename, theta, f=None, z0=50.0, form="ri")`
θ での予測平均を Touchstone に書き出します(§7 を参照)。

---

## 2. `CompactModel` — 低次の有理モデル

`gp.compact()` が返します。中身は scikit-rf の `VectorFitting`(`cm.vf`)です。

| メソッド / 属性 | 処理 |
|---|---|
| `cm.order` | 状態数(実極は 1、複素極は 2 と数える) |
| `cm.response(f)` | `(F, P, P)` の S を計算する |
| `cm.state_space()` | 実数の `(A, B, C, D)` を返す。入射波 a → 反射波 b の状態方程式 ẋ=Ax+Ba, b=Cx+Da。`PortCircuitSimulator` にそのまま渡せる |
| `cm.write_spice(path, name="mlrf_model")` | SPICE サブサーキットを書き出す(ノード名 `p1 … pP`) |
| `cm.write_touchstone(filename, f=None, form="ri")` | Touchstone に書き出す(§7 を参照) |
| `cm.poles, cm.residues, cm.d, cm.e` | 極・留数・定数項・比例項 |
| `cm.passivity_scale` | 受動性の強制で掛けた倍率(1.0 なら縮小なし) |

---

## 3. `StaticNonlinearGP` — 非線形素子(静的 I-V)

I = I0·sinh(z(V)) とおき、z を GP(1次の平均関数+RBF)で学習します。指数的な I-V も1つの滑らかな関数で扱えます。

### コンストラクタ
```python
dg = StaticNonlinearGP(I0=1e-6, bounds=None, reml=False)
```
| 引数 | 意味 |
|---|---|
| `I0` | asinh 変換のスケール。扱う電流範囲の下の方(例:1 µA)に選ぶ。これより十分小さい電流は線形に、大きい電流は対数的に扱われる |
| `bounds` | 入力電圧の範囲 `[(min, max)]`。`suggest()` で候補点を省略したときに使う |
| `reml` | `True` なら制限付き最尤(REML)で超パラメータを推定する。既定は最尤 |

### メソッド
| メソッド | 引数 | 処理 / 戻り値 |
|---|---|---|
| `fit(V, I, optimize=True)` | `V`:`(N,)` または `(N, D)`、`I`:`(N,)` | 学習する。戻り値は `self` |
| `current(v)` | 電圧 `(n,)` | `(i, di/dv)`。平均の電流と微分コンダクタンス。**`PortCircuitSimulator` の `nl` にそのまま渡せる** |
| `current_band(v, k=2.0)` | 電圧 | `(下限, 上限)`。±kσ の電流の帯 |
| `z_mean(v, grad=False)` | 電圧 | z の平均(`grad=True` なら dz/dV も返す) |
| `z_var(v, mean_uncertainty=True)` | 電圧 | z の事後分散。`True` なら平均関数の係数の不確かさも含める(外挿で大きくなる) |
| `suggest(cand=None, n_cand=401)` | 候補電圧 | 次に測るべき電圧を1点返す(分散 × tanh²(z) が最大の点)。`cand` を省略すると `bounds` から候補点を作る |
| `sample_functions(vgrid, n, seed=0)` | 評価グリッドとサンプル数 | 事後分布から I(V) を n 本サンプルし、`SplineSamples` を返す(1入力のみ) |
| `spice_bsource(name, n_plus, n_minus, digits=10)` | 素子名とノード名 | SPICE の B 素子(電流源)の1行を文字列で返す(1入力のみ) |

能動学習の例:
```python
dg = StaticNonlinearGP(I0=1e-6, bounds=[(-1.0, 0.9)]).fit(V, I)
for _ in range(10):
    vn = dg.suggest()
    V, I = np.r_[V, vn], np.r_[I, measure(vn)]
    dg.fit(V, I)
```

---

## 4. `SplineSamples` — 非線形素子の事後サンプル

`dg.sample_functions()` の戻り値です。グリッドの内側は3次スプライン、外側は z の1次外挿です。

| 使い方 | 処理 |
|---|---|
| `samples[j](v)` | j 番目のサンプルで `(i, di/dv)` を返す(`nl` / `nl_funcs` に渡せる) |
| `samples(v)` | `v` の形は `(n,)`。j 番目のサンプルを `v[j]` で評価する(`run_ensemble` 用) |
| `len(samples)`, `for s in samples` | サンプル数、1本ずつ取り出す |

---

## 5. `active_learning` — 線形素子の能動学習

```python
gp, thetas, hist = active_learning(simulate, bounds, n_init=4, n_iter=8, n_cand=2048, seed=0,
                                   model=None, refit_hyper_every=2, mode="imse", verbose=True)
```
| 引数 | 意味 |
|---|---|
| `simulate` | `simulate(theta) -> (f, S)` を返す関数。`S` は `(F, P, P)`。**f は毎回同じでなければならない**(違うと例外) |
| `bounds` | θ の範囲 `[(min, max), ...]` |
| `n_init` | 初期点の数(ラテン超方格) |
| `n_iter` | 追加する点の数 |
| `n_cand` | 候補点の数(Sobol 列) |
| `model` | 設定済みの `ParametricRationalGP` を使いたいときに渡す(`noise_bounds` などを変えたい場合) |
| `refit_hyper_every` | 何回ごとに超パラメータを最適化し直すか(それ以外は固定のまま再学習) |
| `mode` | `"imse"` または `"var"`(`acquisition` を参照) |

戻り値:
- `gp`:学習済みの GP
- `thetas`:`(n_init+n_iter, D)` 使った θ
- `hist`:`[(追加した θ, 評価値), ...]`

---

## 6. `PortCircuitSimulator` — 過渡解析

線形 N ポート(S パラメータの状態空間、基準インピーダンス z0)の各ポートに終端をつないだ回路を、台形則+Newton 法で解きます。線形部分の LU 分解は1回だけで、非線形素子は Woodbury の公式による低ランク更新で扱います。

### コンストラクタ
```python
sim = PortCircuitSimulator(A, B, C, D, z0, terms)
```
`A, B, C, D` は `cm.state_space()` の出力です(任意の S パラメータ状態空間も可)。`terms` にはポートの数だけ終端を、ポートの順に並べます。

| 終端の書き方 | 意味 |
|---|---|
| `("source", vs)` | 内部抵抗 z0 の電圧源。`vs(t)` は時刻 [s] → 電圧 [V] |
| `("source", dict(v=vs, R=Rs))` | 内部抵抗 Rs の電圧源(`R=0` で理想電圧源) |
| `("load", dict(R=..., C=..., nl=...))` | R ∥ C ∥ 非線形素子。`R` の既定は ∞、`C` の既定は 0(C=0 は代数方程式として扱う)。`nl` は `fun(v) -> (i, di/dv)`(例:`dg.current`)または `None` |

非線形素子の電流は、ポート端子から負荷の中へ流れる向きを正とします。

### `run(t_end, h, nl_funcs=None, tol=1e-9, max_newton=50, dv_max=0.1, init="zero")`
固定刻み `h` で `0 … t_end` を解きます。
- 戻り値は `t`(`(T,)`)と `V`(`(T, P)`、全ポートの電圧)です。
- `nl_funcs`:負荷ごとの非線形関数のリスト。`terms` の `nl` を一時的に差し替えます(例:事後サンプル1本)。
- `dv_max`:Newton 法の1反復あたりの電圧ステップ制限 [V]。指数的な素子が発散するのを防ぎます。
- `init`:`"zero"`(全状態 0)、`"dc"`(t=0 の DC 動作点から開始)、または初期状態ベクトル `(n+nL,)`。
- 実行後、`sim.n_newton`(Newton 反復の総数)と `sim.n_fail`(収束しなかったステップの数、0 でなければ警告)が見られます。

### `run_ensemble(t_end, h, nl_batch, n_members, ..., init="zero")`
非線形素子だけが違う `n_members` 個の回路を、LU 分解を共有してまとめて解きます。モンテカルロや不確かさの伝播に使い、個別に `run` するより数倍速くなります。
- `nl_batch`:負荷ごとの関数のリスト。各関数は `v (n_members,)` を受け取り、j 番目の要素を j 番目のメンバーの素子で評価した `(i, di/dv)` を返します。`SplineSamples` がそのまま使えます。
- 戻り値は `t`(`(T,)`)と `V`(`(n_members, T, P)`)です。

### `dc_op(nl_funcs=None, t0=0.0, n_members=1, gmin=1e-12, ...)`
時刻 `t0` の電源値で DC 動作点を求めます(Newton 法。失敗したら電源ステッピング)。戻り値は状態 `z (n+nL, n_members)` です。

### 制限
- 負荷をつなぐポートで DUT の高周波極限が短絡になる場合(ポートに直結したシャント容量など、D の対応する固有値が −1)、定式化が特異になるため例外を出します。負荷側に直列抵抗を入れるなどしてください。
- 時間刻みは固定です。比例項 e≠0 のモデルは状態空間にできません。

---

## 7. `write_touchstone` — 学習結果を Touchstone ファイルに出力

```python
path = write_touchstone(filename, model, theta=None, f=None, z0=None, form="ri")
```
学習した線形素子の S パラメータを Touchstone 形式(`.sNp`、N はポート数)で書き出し、書き出したファイルの絶対パスを返します。

| 引数 | 意味 |
|---|---|
| `filename` | 出力ファイル名。拡張子 `.sNp` が無ければ付ける(`"out/pred"` → `out/pred.s2p`)。存在しないディレクトリは作る |
| `model` | `ParametricRationalGP`:`theta` での予測平均を出力する。`CompactModel`:縮約モデルの応答を出力する |
| `theta` | `model` が GP のとき必須。出力する設計パラメータ `(D,)` |
| `f` | 出力する周波数 [Hz]。省略すると、GP は学習周波数、`CompactModel` は縮約に使った周波数(帯域外にも出力できる) |
| `z0` | 基準インピーダンス [Ω]。省略すると、GP は 50 Ω、`CompactModel` は `cm.z0` |
| `form` | `"ri"`(実部・虚部)、`"ma"`(振幅・角度)、`"db"`(dB・角度) |

- ファイルの先頭にはコメントが入ります。GP なら θ の値(`!mlrf ParametricRationalGP mean, L_nH=2.2, Cc_pF=0.7`)、`CompactModel` なら次数と受動性の縮小倍率です。
- 予測の σ(不確かさ)は Touchstone に入りません。
- `StaticNonlinearGP` は S パラメータを持たないので `TypeError` になります。
- メソッドとしても呼べます:`gp.write_touchstone(filename, theta, ...)`、`cm.write_touchstone(filename, ...)`。

```python
write_touchstone("results/filter_theta1", gp, theta=[2.2, 0.7])                 # results/filter_theta1.s2p
write_touchstone("compact.s2p", cm, f=np.linspace(0.1e9, 6e9, 601), form="db")
for i, th in enumerate(thetas):                                                # θ ごとに一括出力
    gp.write_touchstone(f"sweep/theta_{i:02d}", th)
```

---

## 8. `write_ltspice` — 学習結果を LTspice 用モデルに変換

```python
path = write_ltspice(filename, models, names=None, theta=None, zmax=300.0,
                     symbols=True, ref_pin=False, model_file="absolute", **compact_kw)
```
学習したモデルを LTspice のサブサーキット(`.lib`)と、モデルごとのシンボル(`<NAME>.asy`)に書き出します。同じモデルを **AC 解析(周波数解析)と過渡解析(.tran)の両方** に使えます。戻り値は書き出した `.lib` の絶対パスです。シンボルは `.lib` と同じフォルダに置かれます。

| 引数 | 意味 |
|---|---|
| `filename` | 出力ファイル名。拡張子が無ければ `.lib` を付ける。存在しないディレクトリは作る |
| `models` | 1つのモデル、またはモデルのリスト(1ファイルにまとめて書き出す) |
| `names` | サブサーキット名(文字列またはリスト)。省略すると `mlrf_lin1`、`mlrf_nl1` などになる |
| `theta` | `ParametricRationalGP` を渡すときの設計パラメータ。内部で `compact(theta)` してから書き出す |
| `zmax` | 非線形素子の sinh の引数の上限(オーバーフロー防止) |
| `symbols` | `True`(既定)なら `<NAME>.asy` も書き出す |
| `ref_pin` | `True` なら線形素子にポートの基準端子 `ref` を付ける(シンボルでは下側のピン)。`False` ならポートの基準はグラウンド 0 |
| `model_file` | シンボルの `ModelFile` 属性に書く `.lib` のパス。`"absolute"`(絶対パス、既定)または `"name"`(ファイル名だけ。`.lib` を回路図と同じフォルダか LTspice のライブラリ検索パスに置く場合) |
| `**compact_kw` | GP を渡したときに `compact()` に渡す引数(`n_dense`、`enforce_passivity` など) |

モデルの種類ごとの出力:

| モデル | サブサーキット | 中身 | シンボル |
|---|---|---|---|
| `CompactModel`(または GP + `theta`) | `.subckt NAME p1 … pP [ref]` | 状態空間を R・C・E・F・G だけで実現した等価回路。ポートの基準はグラウンド 0(`ref_pin=True` なら `ref`)、基準インピーダンスは `cm.z0` | 箱型。奇数ポートは左、偶数ポートは右、`ref` は下 |
| `StaticNonlinearGP`(1入力) | `.subckt NAME a k` | B 素子1個。a→k に流れる電流 `I0*sinh(z(V(a,k)))`。学習範囲の外は z を1次式で外挿する | ダイオード型。`a` が上、`k` が下 |

LTspice での使い方:

**回路図(シンボルを使う)**
1. 書き出した `<NAME>.asy` を、回路図(`.asc`)と同じフォルダにコピーします。複数の回路図で使う場合は、Settings → Search Paths(Control Panel → Sym. & Lib. Search Paths)の Symbol 欄に `.asy` のフォルダを追加します。
2. 部品の追加(F2)で `NAME` を選んで配置します。
3. `.lib` はシンボルの `ModelFile` 属性で自動的に読み込まれます(`.include` は不要)。`.lib` を移動したら `write_ltspice` で書き出し直すか、`model_file="name"` で書き出してください。

**ネットリスト**
```spice
.include models.lib
Vs src 0 SIN(0 2 2G) AC 1
Rs src p1 50
X1 p1 p2 filt          ; 線形素子 (2 ポート)
RL p2 0 50
X2 p2 0 dml            ; 非線形素子 (アノード p2, カソード 0)
.ac dec 200 100Meg 10G ; 周波数解析 (非線形素子は動作点で線形化される)
* .tran 0 8n 0 2p      ; 過渡解析
```

- 線形モデルは、状態変数を極の大きさで正規化して書き出します。素子の係数は O(1) に、内部ノードの電圧は V オーダーになるので、LTspice の既定の許容誤差でも精度が落ちません。
- 線形モデルの精度は `compact()` の縮約精度で決まります。Python で `cm.response(f)` を確認してから書き出してください。
- 非線形素子は静的な I-V だけです。接合容量などの電荷 Q(V) は含まないので、必要なら外付けの C を並列に入れてください。
- 式中の2乗は `^` ではなく積で書いています(LTspice の B 素子では `^` がべき乗ではないため)。ngspice 向けには `StaticNonlinearGP.spice_bsource()` / `CompactModel.write_spice()` を使ってください。

**検証**(`verify_ltspice.py`・`verify_ltspice_symbols.py`、LTspice 24 のバッチモードで実行):

| 比較 | 結果 |
|---|---|
| AC 解析の S パラメータ vs `cm.response` | 最大差 7e-15 |
| ダイオードの DC スイープ vs `dg.current` | 最大相対誤差 8e-9 |
| フィルタ → ダイオードリミッタの過渡解析 vs `PortCircuitSimulator` | RMS 差 0.002 mV(振幅 0.57 V) |
| シンボルを置いた回路図(`.asc`)の過渡解析 vs 同じ回路のネットリスト | 差 0.0000 mV(LTspice がピンの順序どおりに X 行と `.lib` 行を生成) |
| `ref` ピン付きシンボル(ref=GND)vs `PortCircuitSimulator` | RMS 差 0.0013 mV |

LTspice の場所が既定(`%LOCALAPPDATA%\Programs\ADI\LTspice\LTspice.exe`)と違う場合は、環境変数 `LTSPICE` で指定してください。

---

## 9. 学習データ(周波数特性・I-V 特性)のファイルを指定する

### 周波数特性:`load_touchstone` / `ParametricRationalGP.fit_touchstone`
θ(設計パラメータ)ごとに1つの Touchstone ファイル(`.s1p`、`.s2p` など)を用意し、ファイル群と各ファイルの θ を指定します。

```python
# (a) θ をファイル名から取り出す: 正規表現の各グループ (...) が θ の1成分 (bounds と同じ順)
gp = ParametricRationalGP([(1.5, 3.0), (0.3, 1.2)], names=["L_nH", "Cc_pF"])
gp.fit_touchstone("data/*.s2p", pattern=r"L([\d.]+)nH_Cc([\d.]+)pF")
#   data/filter_L2.000nH_Cc0.600pF.s2p → θ = (2.0, 0.6)

# (b) θ を明示する (ファイルの順と同じ順に並べる)
files = ["meas/run01.s2p", "meas/run02.s2p", "meas/run03.s2p"]
gp.fit_touchstone(files, thetas=[[1.5, 0.3], [2.0, 0.6], [3.0, 1.2]])

# (c) 読み込みだけ行い、配列として使う
thetas, f, S = load_touchstone("data", pattern=r"L([\d.]+)nH_Cc([\d.]+)pF")
gp.fit(thetas, f, S)
```

| 引数 | 意味 |
|---|---|
| `files` | ファイルのリスト、ワイルドカード(`"data/*.s2p"`)、またはフォルダ(中の `.sNp` / `.ts` を名前順に読む) |
| `thetas` | 各ファイルの θ `(Nθ, D)`。ワイルドカードやフォルダの場合は名前順に対応する |
| `pattern` | `thetas` を省略したとき、ファイル名から θ を取り出す正規表現 |
| `f` | 学習に使う周波数 [Hz]。省略すると、全ファイルが同じ周波数ならそれを使う。違う場合は1つ目のファイルの周波数のうち全ファイルに共通する範囲を使い、他のファイルは3次補間する(外挿はしない) |
| `z0` | 基準インピーダンス(既定 50 Ω)。ファイルの z0 が違えばこの値に変換する |

- 戻り値は `thetas (Nθ, D)`、`f (F,)`、`S (Nθ, F, P, P)` です。
- f ≤ 0 の点(DC)は自動で除きます。
- `fit_touchstone(..., **fit_kw)` の `fit_kw`(`n_restarts` など)は `fit` に渡されます。

能動学習(§5)でファイルを使う場合は、`simulate` の中で「θ に対応するファイルを作る(EM シミュレータを呼ぶなど)→ 読む」処理を書きます:
```python
def simulate(theta):
    path = run_em_simulation(theta)          # θ の Touchstone ファイルを作る (利用者の処理)
    _, f, S = load_touchstone([path], thetas=[theta])
    return f, S[0]
```

### I-V 特性:`load_iv` / `StaticNonlinearGP.fit_file`
CSV、TSV、空白区切りのテキストを読みます。LTspice の波形エクスポート(File → Export data as text)もそのまま読めます。

```python
dg = StaticNonlinearGP(I0=1e-6).fit_file("diode_iv.csv", v_col="Vd [V]", i_col="Id [mA]", i_scale=1e-3)
V, I = load_iv("dc_sweep.txt", v_col=0, i_col=1)        # 読み込みだけ
dg = StaticNonlinearGP(I0=1e-6).fit(V, I)
```

| 引数 | 意味 |
|---|---|
| `path` | ファイルのパス |
| `v_col` | 電圧の列。列番号(0 始まり)またはヘッダの列名。リストにすると多入力 `(N, D)` |
| `i_col` | 電流の列。列番号またはヘッダの列名 |
| `delimiter` | 区切り文字。省略すると `,`・タブ・`;`・空白から自動判定する |
| `v_scale`, `i_scale` | 単位換算の倍率(例:電流が mA なら `i_scale=1e-3`) |

- 数値として読めない行はヘッダとして扱い、最後のヘッダ行を列名に使います。空行と `#`・`!`・`*` で始まる行は飛ばします。
- 文字コードは UTF-8 / Shift_JIS を自動判定します。
- 能動学習で測定点を増やす場合は、`dg.suggest()` で次の電圧を決め、測定した点を追加して `fit` し直します(§3 の例)。

---

## 補助関数
- `szego_real(w1, w2, a)`:[Re; Im] に実数化した Szegö カーネル。周波数は正規化角周波数。
- `rbf(X1, X2, ell)`:ARD-RBF カーネル。

## 関連ファイル
| ファイル | 内容 |
|---|---|
| `demo_mlrf.py` | フィルタ+ダイオードリミッタの通しの例(ngspice があれば比較もする) |
| `test_mlrf.py` | 解析解・常微分方程式の数値解との比較テスト(`python test_mlrf.py [旧版 mlrf.py]`) |
| `verify_ltspice.py` | `write_ltspice` の出力を LTspice で解いて Python と比較する(結果は `ltspice_check/`) |
| `verify_ltspice_symbols.py` | `load_touchstone`・`load_iv` と、`.asy` シンボルを置いた回路図の LTspice 実行を検証する |
| `mlrf_review.md` | 良い点・問題点・改善内容のまとめ |
