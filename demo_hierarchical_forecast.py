# -*- coding: utf-8 -*-
"""
需求预测 Demo：门店(Store) × 产品(Product) 层级概率销量预测
================================================================
覆盖三大考点：
  1. 层次化预测与调和  —— bottom-up / OLS / MinT 三种调和，对比未调和的独立预测
  2. 间歇性需求        —— Croston vs AutoARIMA 在零膨胀 SKU 上的表现
  3. 概率预测          —— 分位数(Pinball loss) + CRPS 评估

数据：合成的「区域→门店→产品」日销量（含周末效应、周季节、趋势、间歇性零膨胀）。
      换真实数据时，把 build_data() 替换为 M5 / Rossmann 加载即可，
      其余调和 + 概率逻辑完全复用。

运行：python demo_hierarchical_forecast.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, CrostonOptimized

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
N_TRAIN = 365          # 训练天数
H = 28                 # 预测步长
N_TOTAL = N_TRAIN + H
SEED = 42

# 层级结构：区域 -> 门店 -> 产品（SKU）
REGIONS = {"R1": ["S1", "S2"], "R2": ["S3"]}
STORE_PRODUCTS = {
    "S1": ["p1", "p2", "p3", "p4"],
    "S2": ["p5", "p6", "p7", "p8"],
    "S3": ["p9", "p10", "p11", "p12"],
}
STORES = ["S1", "S2", "S3"]
PRODUCTS = [p for ps in STORE_PRODUCTS.values() for p in ps]
STORE_OF = {s: r for r, ss in REGIONS.items() for s in ss}
PRODUCT_STORE = {p: s for s, ps in STORE_PRODUCTS.items() for p in ps}

# 全部层级序列 id：产品(底层) + 门店 + 区域 + 总量
BOTTOM_IDS = PRODUCTS
AGG_IDS = PRODUCTS + STORES + list(REGIONS.keys()) + ["total"]
N_BOTTOM = len(PRODUCTS)
N_SERIES = len(AGG_IDS)
ID_TO_ROW = {a: i for i, a in enumerate(AGG_IDS)}

# 每只产品的参数：base=基础量, trend=趋势, amp=周季节幅度, weekend=周末增量,
#               noise=噪声, inter=间歇性零占比(0~1)
PROD_PARAMS = {
    "p1":  dict(base=40, trend=0.05, amp=8, weekend=25, noise=4, inter=0.0),
    "p2":  dict(base=30, trend=0.03, amp=5, weekend=15, noise=3, inter=0.0),
    "p3":  dict(base=8,  trend=0.0,  amp=0, weekend=0,  noise=2, inter=0.70),
    "p4":  dict(base=25, trend=0.02, amp=4, weekend=12, noise=3, inter=0.0),
    "p5":  dict(base=35, trend=0.04, amp=7, weekend=20, noise=4, inter=0.0),
    "p6":  dict(base=28, trend=0.02, amp=5, weekend=14, noise=3, inter=0.0),
    "p7":  dict(base=9,  trend=0.0,  amp=0, weekend=0,  noise=2, inter=0.72),
    "p8":  dict(base=22, trend=0.02, amp=4, weekend=11, noise=3, inter=0.0),
    "p9":  dict(base=45, trend=0.05, amp=9, weekend=28, noise=5, inter=0.0),
    "p10": dict(base=33, trend=0.03, amp=6, weekend=18, noise=4, inter=0.0),
    "p11": dict(base=27, trend=0.02, amp=5, weekend=13, noise=3, inter=0.0),
    "p12": dict(base=7,  trend=0.0,  amp=0, weekend=0,  noise=2, inter=0.75),
}
INTERMITTENT = [p for p, pr in PROD_PARAMS.items() if pr["inter"] > 0]


# ----------------------------------------------------------------------
# 数据生成（换真实数据只需替换此函数）
# ----------------------------------------------------------------------
def build_data():
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2022-01-01", periods=N_TOTAL, freq="D")
    dow = dates.dayofweek.values
    weekend = (dow >= 5).astype(float)
    t = np.arange(N_TOTAL)

    bottom = {}
    for p in PRODUCTS:
        pr = PROD_PARAMS[p]
        y = (pr["base"] + pr["trend"] * t / 365
             + pr["amp"] * np.sin(2 * np.pi * t / 7.0)
             + pr["weekend"] * weekend
             + pr["noise"] * rng.standard_normal(N_TOTAL))
        y = np.maximum(y, 0.0)
        if pr["inter"] > 0:                      # 零膨胀 -> 间歇性需求
            mask = rng.random(N_TOTAL) > pr["inter"]
            y = np.where(mask, y, 0.0)
        bottom[p] = np.round(y, 2)

    full = pd.DataFrame({"ds": dates})
    for p in PRODUCTS:
        full[p] = bottom[p]
    for s in STORES:                              # 门店 = Σ 产品
        full[s] = full[[p for p in STORE_PRODUCTS[s]]].sum(axis=1)
    for r, ss in REGIONS.items():                 # 区域 = Σ 门店
        full[r] = full[ss].sum(axis=1)
    full["total"] = full[PRODUCTS].sum(axis=1)    # 总量 = Σ 全部
    return full


def to_long(df, ids):
    return pd.concat(
        [pd.DataFrame({"unique_id": i, "ds": df["ds"].values, "y": df[i].values})
         for i in ids],
        ignore_index=True,
    )


# ----------------------------------------------------------------------
# 评估指标
# ----------------------------------------------------------------------
def mae(a, f):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(f))))


def pinball(a, q, tau):
    d = np.asarray(a) - np.asarray(q)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def crps_from_quantiles(a, qs, taus):
    """CRPS ≈ 2 × 各分位数 Pinball loss 的均值。"""
    return 2.0 * np.mean([pinball(a, qs[i], taus[i]) for i in range(len(taus))])


# ----------------------------------------------------------------------
# 调和核心
# ----------------------------------------------------------------------
def build_summing_matrix():
    S = np.zeros((N_SERIES, N_BOTTOM))
    for j, p in enumerate(PRODUCTS):
        s = PRODUCT_STORE[p]
        r = STORE_OF[s]
        for a in (p, s, r, "total"):
            S[ID_TO_ROW[a], j] = 1.0
    return S


def reconcile(y_hat, S, method="bu", W_inv=None):
    """y_hat: (H, N_SERIES) 各层级 base forecast；返回调和后 (H, N_SERIES)。"""
    if method == "bu":
        b = y_hat[:, :N_BOTTOM]                    # 只取底层预测，向上加总
        return (S @ b.T).T
    if method == "ols":
        G = np.linalg.pinv(S.T @ S) @ S.T          # G_OLS = (S'S)⁻¹ S'
    else:  # mint
        G = np.linalg.pinv(S.T @ W_inv @ S) @ S.T @ W_inv  # G = (S'W⁻¹S)⁻¹ S'W⁻¹
    return (S @ G @ y_hat.T).T


def estimate_W(train_df, period=7):
    """用季节性 naive 残差估计 base forecast 误差协方差 W。

    收缩到对角线（保留异方差）+ 微扰，保证可逆且异方差信息不丢。
    注意：当误差同方差且不相关时，MinT 会退化为 OLS。
    """
    resid = np.zeros((N_TRAIN - period, N_SERIES))
    for k, a in enumerate(AGG_IDS):
        y = train_df[a].values
        resid[:, k] = y[period:] - y[:-period]
    Sigma = np.cov(resid.T)
    lmbd = 0.1
    Sigma_shrunk = (1 - lmbd) * Sigma + lmbd * np.diag(np.diag(Sigma)) \
        + 1e-6 * np.eye(N_SERIES)
    return np.linalg.pinv(Sigma_shrunk)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("需求预测 Demo：门店 × 产品 层级概率销量预测")
    print("=" * 70)

    full = build_data()
    train = full.iloc[:N_TRAIN].copy()
    test = full.iloc[N_TRAIN:].copy()

    # ---- 拟合 base 模型（全部 18 条序列一起拟合） ----
    levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    train_long = to_long(train, AGG_IDS)
    sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=1)
    sf.fit(train_long)
    fc = sf.predict(h=H, level=levels)

    # 提取点预测矩阵 (H, N_SERIES)
    y_hat = fc.pivot(index="ds", columns="unique_id", values="AutoARIMA")[AGG_IDS].values

    S = build_summing_matrix()
    W_inv = estimate_W(train)
    y_bu = reconcile(y_hat, S, "bu")
    y_ols = reconcile(y_hat, S, "ols")
    y_mint = reconcile(y_hat, S, "mint", W_inv)

    t_idx = ID_TO_ROW["total"]
    actual_total = test["total"].values

    # ---- 考点 1：层次化调和（对比各层级 MAE） ----
    print("\n[考点1] 层次化调和 —— 未来 28 天各层级 MAE 对比")
    print(f"{'层级':<8}{'独立(未调和)':>14}{'bottom-up':>12}{'OLS':>10}{'MinT':>10}")
    print("-" * 56)
    methods = {
        "独立(未调和)": y_hat,
        "bottom-up": y_bu,
        "OLS": y_ols,
        "MinT": y_mint,
    }
    for aid in ["total"] + list(REGIONS.keys()) + STORES:
        i = ID_TO_ROW[aid]
        a = test[aid].values
        vals = [mae(a, y[:, i]) for y in methods.values()]
        print(f"{aid:<8}" + "".join(f"{v:>14.2f}" for v in vals))

    # 一致性说明
    store_sum_indep = y_hat[:, [ID_TO_ROW[s] for s in STORES]].sum(axis=1)
    tot_indep = y_hat[:, ID_TO_ROW["total"]]
    print(f"\n独立预测：门店之和 ≠ 总销量（累计差异 {np.abs(store_sum_indep - tot_indep).sum():.1f}）")
    print("调和后：bottom-up / OLS / MinT 均保证 门店之和 == 总销量（层级一致）")
    print("说明：总销量层级是所有一致化方法的共同总和，差异体现在门店/区域层级；")
    print("      MinT 在误差同方差不相关时会退化为 OLS，异方差明显时才有加权优势。")

    # ---- 考点 3：概率预测（总销量的分位数 + CRPS） ----
    print("\n[考点3] 概率预测 —— 总销量分位数评估")
    fc_total = fc[fc["unique_id"] == "total"].sort_values("ds")
    actual = test.set_index("ds").loc[fc_total["ds"].values, "total"].values

    # 分位数映射：level=L 的区间，下界 tau=(100-L)/200，上界 tau=(100+L)/200
    taus, qs = [], []
    for L in levels:
        taus.append((100 - L) / 200.0)
        qs.append(fc_total[f"AutoARIMA-lo-{L}"].values)
        taus.append((100 + L) / 200.0)
        qs.append(fc_total[f"AutoARIMA-hi-{L}"].values)
    taus.append(0.50)                       # 点预测 = 中位数
    qs.append(fc_total["AutoARIMA"].values)

    crps = crps_from_quantiles(actual, qs, taus)
    pinball_low = pinball(actual, fc_total["AutoARIMA-lo-80"].values, 0.10)   # 10% 分位
    pinball_high = pinball(actual, fc_total["AutoARIMA-hi-80"].values, 0.90)  # 90% 分位
    print(f"  10% 分位 Pinball loss : {pinball_low:.2f}")
    print(f"  90% 分位 Pinball loss : {pinball_high:.2f}")
    print(f"  CRPS（分位数近似）     : {crps:.2f}")
    print("  （Pinball loss 惩罚分位数方向的不对称误差；CRPS 衡量整体概率预测质量）")

    # ---- 考点 2：间歇性需求（Croston vs AutoARIMA vs 零预测） ----
    print("\n[考点2] 间歇性需求 —— 零膨胀 SKU：点预测 MAE 会「骗人」，要看累计销量")
    print(f"{'产品':<6}{'零占比':>8}{'MAE(ARIMA)':>12}{'MAE(Croston)':>13}{'MAE(全0)':>11}")
    print("-" * 52)
    inter_long = to_long(train, INTERMITTENT)
    sf_inter = StatsForecast(
        models=[AutoARIMA(season_length=7), CrostonOptimized()], freq="D", n_jobs=1
    )
    sf_inter.fit(inter_long)
    fc_inter = sf_inter.predict(h=H)
    for p in INTERMITTENT:
        actual = test[p].values
        f_arima = fc_inter[fc_inter["unique_id"] == p]["AutoARIMA"].values
        f_crost = fc_inter[fc_inter["unique_id"] == p]["CrostonOptimized"].values
        zero_ratio = (train[p] == 0).mean()
        print(f"{p:<6}{zero_ratio:>8.2f}{mae(actual, f_arima):>12.2f}"
              f"{mae(actual, f_crost):>13.2f}{mae(actual, np.zeros(H)):>11.2f}")

    print("\n  28 天累计销量对比（零预测严重低估，无法用于备货）：")
    print(f"{'产品':<6}{'实际累计':>10}{'AutoARIMA':>12}{'Croston':>10}{'全0预测':>10}")
    print("-" * 52)
    for p in INTERMITTENT:
        actual = test[p].values
        f_arima = fc_inter[fc_inter["unique_id"] == p]["AutoARIMA"].values
        f_crost = fc_inter[fc_inter["unique_id"] == p]["CrostonOptimized"].values
        print(f"{p:<6}{actual.sum():>10.1f}{f_arima.sum():>12.1f}{f_crost.sum():>10.1f}{0.0:>10.1f}")
    print("  （MAE 对零膨胀序列不敏感——中位数≈0；备货要看累计销量或高分位，Croston 给出无偏需求量）")

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = test["ds"].values
    ax.plot(x, actual_total, label="Actual total sales", color="black", lw=2)
    ax.plot(x, y_hat[:, t_idx], label="Independent (incoherent)", ls="--", alpha=0.7)
    ax.plot(x, y_bu[:, t_idx], label="bottom-up", alpha=0.85)
    ax.plot(x, y_mint[:, t_idx], label="MinT reconciliation", alpha=0.9, lw=2)
    ax.set_title("Total sales: actual vs reconciliation (next 28 days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("total_forecast_comparison.png", dpi=110)
    print("\n已保存对比图：total_forecast_comparison.png")


if __name__ == "__main__":
    main()
