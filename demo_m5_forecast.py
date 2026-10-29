# -*- coding: utf-8 -*-
"""
需求预测 Demo（真实数据 M5）：门店 × 商品 层级概率销量预测
================================================================
数据：Kaggle M5（Walmart 门店×商品层级销量，2011-2016 日粒度）。
      层级映射到餐厅场景：state(区域) -> store(门店) -> item(产品/SKU)。
覆盖三大考点：
  1. 层次化调和  —— bottom-up / OLS / MinT，对比独立预测
  2. 间歇性需求  —— Croston vs AutoARIMA 在零膨胀 SKU 上
  3. 概率预测    —— 分位数(Pinball loss) + CRPS

运行：python demo_m5_forecast.py   （首次会自动下载 M5 数据并缓存到 ./data）
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, CrostonOptimized
from datasetsforecast.m5 import M5

from demo_hierarchical_forecast import mae, pinball, crps_from_quantiles, to_long

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
N_TRAIN = 365
H = 28
STATE = "CA"            # 选一个州（CA 有 3 家门店 CA_1/CA_2/CA_3）
TOP_ITEMS = 20          # 取该州销量最大的前 N 个商品，控制运行规模


# ----------------------------------------------------------------------
# 加载并切子集
# ----------------------------------------------------------------------
def load_subset():
    Y, X, S = M5.load(directory="data")

    # 只保留目标州，并把静态属性并进来（category -> str，避免 groupby 混入 NaN）
    S_cols = ["unique_id", "item_id", "store_id", "state_id"]
    df = Y.merge(S[S_cols], on="unique_id")
    for c in S_cols:
        df[c] = df[c].astype(str)
    df = df[df["state_id"] == STATE]

    # 取销量最大的前 N 个 item（保证序列有信号）
    top_items = df.groupby("item_id")["y"].sum().nlargest(TOP_ITEMS).index
    df = df[df["item_id"].isin(top_items)]

    # 只取最后 N_TRAIN+H 天
    last_days = sorted(df["ds"].unique())[-(N_TRAIN + H):]
    df = df[df["ds"].isin(last_days)]

    bottom_ids = sorted(df["unique_id"].unique())
    store_of = df.groupby("unique_id")["store_id"].first().to_dict()
    state_of = df.groupby("unique_id")["state_id"].first().to_dict()
    stores = sorted({s for s in store_of.values()})
    states = sorted({s for s in state_of.values()})

    # 宽表：ds × bottom_id
    full = df.pivot(index="ds", columns="unique_id", values="y").fillna(0.0)
    for s in stores:
        cols = [u for u in bottom_ids if store_of[u] == s]
        full[s] = full[cols].sum(axis=1)
    for st in states:
        cols = [u for u in bottom_ids if state_of[u] == st]
        full[st] = full[cols].sum(axis=1)
    full["total"] = full[bottom_ids].sum(axis=1)
    full = full.reset_index().sort_values("ds")

    agg_ids = bottom_ids + stores + states + ["total"]
    return full, agg_ids, bottom_ids, store_of, state_of, stores


# ----------------------------------------------------------------------
# 调和核心（泛化版）
# ----------------------------------------------------------------------
def build_S(agg_ids, bottom_ids, store_of, state_of):
    n, m = len(agg_ids), len(bottom_ids)
    S = np.zeros((n, m))
    row = {a: i for i, a in enumerate(agg_ids)}
    for j, b in enumerate(bottom_ids):
        S[row[b], j] = 1.0
        S[row[store_of[b]], j] = 1.0
        S[row[state_of[b]], j] = 1.0
        S[row["total"], j] = 1.0
    return S


def reconcile(y_hat, S, n_bottom, method="bu", W_inv=None):
    if method == "bu":
        return (S @ y_hat[:, :n_bottom].T).T
    if method == "ols":
        G = np.linalg.pinv(S.T @ S) @ S.T
    else:  # mint
        G = np.linalg.pinv(S.T @ W_inv @ S) @ S.T @ W_inv
    return (S @ G @ y_hat.T).T


def estimate_W(train_df, agg_ids, period=7):
    resid = np.zeros((N_TRAIN - period, len(agg_ids)))
    for k, a in enumerate(agg_ids):
        y = train_df[a].values
        resid[:, k] = y[period:] - y[:-period]
    Sigma = np.cov(resid.T)
    lmbd = 0.1
    Sigma = (1 - lmbd) * Sigma + lmbd * np.diag(np.diag(Sigma)) + 1e-6 * np.eye(len(agg_ids))
    return np.linalg.pinv(Sigma)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"需求预测 Demo（真实数据 M5）：门店 × 商品 层级概率销量预测（州={STATE}）")
    print("=" * 70)

    full, agg_ids, bottom_ids, store_of, state_of, stores = load_subset()
    states = sorted(set(state_of.values()))
    n_bottom = len(bottom_ids)
    n_series = len(agg_ids)
    print(f"底层序列(item×store)={n_bottom}，门店={len(stores)}，总序列={n_series}")

    train = full.iloc[:N_TRAIN].copy()
    test = full.iloc[N_TRAIN:].copy()
    id_to_row = {a: i for i, a in enumerate(agg_ids)}

    # ---- 拟合 base 模型（全部序列一起） ----
    levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=-1)
    sf.fit(to_long(train, agg_ids))
    fc = sf.predict(h=H, level=levels)
    y_hat = fc.pivot(index="ds", columns="unique_id", values="AutoARIMA")[agg_ids].values

    S = build_S(agg_ids, bottom_ids, store_of, state_of)
    W_inv = estimate_W(train, agg_ids)
    y_bu = reconcile(y_hat, S, n_bottom, "bu")
    y_ols = reconcile(y_hat, S, n_bottom, "ols")
    y_mint = reconcile(y_hat, S, n_bottom, "mint", W_inv)

    # ---- 考点 1：层次化调和 ----
    print("\n[考点1] 层次化调和 —— 未来 28 天各层级 MAE 对比")
    print(f"{'层级':<10}{'独立(未调和)':>14}{'bottom-up':>12}{'OLS':>10}{'MinT':>10}")
    print("-" * 58)
    methods = {"独立(未调和)": y_hat, "bottom-up": y_bu, "OLS": y_ols, "MinT": y_mint}
    show_ids = ["total"] + states + stores
    for aid in show_ids:
        i = id_to_row[aid]
        a = test[aid].values
        vals = [mae(a, y[:, i]) for y in methods.values()]
        print(f"{aid:<10}" + "".join(f"{v:>14.2f}" for v in vals))

    store_sum = y_hat[:, [id_to_row[s] for s in stores]].sum(axis=1)
    tot_indep = y_hat[:, id_to_row["total"]]
    print(f"\n独立预测：门店之和 ≠ 总销量（累计差异 {np.abs(store_sum - tot_indep).sum():.1f}）")
    print("调和后：bottom-up / OLS / MinT 保证 门店之和 == 总销量（层级一致）")

    # ---- 考点 3：概率预测 ----
    print("\n[考点3] 概率预测 —— 总销量分位数评估")
    fc_total = fc[fc["unique_id"] == "total"].sort_values("ds")
    actual = test.set_index("ds").loc[fc_total["ds"].values, "total"].values
    taus, qs = [], []
    for L in levels:
        taus.append((100 - L) / 200.0); qs.append(fc_total[f"AutoARIMA-lo-{L}"].values)
        taus.append((100 + L) / 200.0); qs.append(fc_total[f"AutoARIMA-hi-{L}"].values)
    taus.append(0.50); qs.append(fc_total["AutoARIMA"].values)
    print(f"  10% 分位 Pinball loss : {pinball(actual, fc_total['AutoARIMA-lo-80'].values, 0.10):.2f}")
    print(f"  90% 分位 Pinball loss : {pinball(actual, fc_total['AutoARIMA-hi-80'].values, 0.90):.2f}")
    print(f"  CRPS（分位数近似）     : {crps_from_quantiles(actual, qs, taus):.2f}")

    # ---- 考点 2：间歇性需求 ----
    print("\n[考点2] 间歇性需求 —— 零膨胀 SKU（item×store）")
    zero_ratio = (train[bottom_ids] == 0).mean().sort_values(ascending=False)
    inter_ids = list(zero_ratio[(zero_ratio > 0.3) & (zero_ratio < 0.95)].head(5).index)
    print(f"  选取零占比 0.3~0.95 的 SKU {len(inter_ids)} 个：")
    inter_long = to_long(train, inter_ids)
    sf_inter = StatsForecast(
        models=[AutoARIMA(season_length=7), CrostonOptimized()], freq="D", n_jobs=-1
    )
    sf_inter.fit(inter_long)
    fc_inter = sf_inter.predict(h=H)
    print(f"{'SKU':<22}{'零占比':>8}{'MAE(ARIMA)':>12}{'MAE(Croston)':>13}{'MAE(全0)':>11}")
    print("-" * 66)
    for u in inter_ids:
        a = test[u].values
        fa = fc_inter[fc_inter["unique_id"] == u]["AutoARIMA"].values
        fcrost = fc_inter[fc_inter["unique_id"] == u]["CrostonOptimized"].values
        zr = zero_ratio[u]
        print(f"{u:<22}{zr:>8.2f}{mae(a, fa):>12.2f}{mae(a, fcrost):>13.2f}{mae(a, np.zeros(H)):>11.2f}")
    print("  （零膨胀序列 MAE 会偏向「预测 0」；备货要看累计销量/高分位，Croston 给无偏需求量）")

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = test["ds"].values
    t_idx = id_to_row["total"]
    ax.plot(x, test["total"].values, label="Actual total sales", color="black", lw=2)
    ax.plot(x, y_hat[:, t_idx], label="Independent (incoherent)", ls="--", alpha=0.7)
    ax.plot(x, y_bu[:, t_idx], label="bottom-up", alpha=0.85)
    ax.plot(x, y_mint[:, t_idx], label="MinT reconciliation", alpha=0.9, lw=2)
    ax.set_title("M5 total sales: actual vs reconciliation (next 28 days)")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("m5_total_forecast_comparison.png", dpi=110)
    print("\n已保存对比图：m5_total_forecast_comparison.png")


if __name__ == "__main__":
    main()
