# -*- coding: utf-8 -*-
"""
概率预测对比：经典 AutoARIMA vs 深度学习 NHITS（全局模型）
================================================================
在 M5 真实数据的底层 SKU（item×store）序列上，比较两者的概率预测质量：
  - 点预测 MAE
  - 概率预测 CRPS（分位数近似）

NHITS 是「全局模型」：用一个网络同时学所有 SKU，靠序列间共享模式提升精度，
尤其适合 SKU 多、单序列历史短的零售需求预测场景。

运行：python demo_neural_forecast.py
"""

import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.utils import PredictionIntervals

from demo_m5_forecast import load_subset, to_long
from demo_hierarchical_forecast import mae, crps_from_quantiles

N_TRAIN = 365
H = 28
LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]


def _quantiles_and_taus(fcu, model):
    taus, qs = [], []
    for L in LEVELS:
        taus.append((100 - L) / 200.0)
        qs.append(fcu[f"{model}-lo-{L}"].values)
        taus.append((100 + L) / 200.0)
        qs.append(fcu[f"{model}-hi-{L}"].values)
    taus.append(0.5)
    qs.append(fcu[model].values)
    return qs, taus


def avg_crps(fc, model, bottom_ids, test):
    vals = []
    for u in bottom_ids:
        fcu = fc[fc["unique_id"] == u].sort_values("ds")
        actual = test.set_index("ds").loc[fcu["ds"].values, u].values
        qs, taus = _quantiles_and_taus(fcu, model)
        vals.append(crps_from_quantiles(actual, qs, taus))
    return float(np.mean(vals))


def avg_mae(fc, model, bottom_ids, test):
    vals = []
    for u in bottom_ids:
        fcu = fc[fc["unique_id"] == u].sort_values("ds")
        actual = test.set_index("ds").loc[fcu["ds"].values, u].values
        vals.append(mae(actual, fcu[model].values))
    return float(np.mean(vals))


def main():
    print("=" * 70)
    print("概率预测对比：AutoARIMA vs NHITS（M5 底层 SKU，未来 28 天）")
    print("=" * 70)

    full, agg_ids, bottom_ids, store_of, state_of, stores = load_subset()
    train = full.iloc[:N_TRAIN].copy()
    test = full.iloc[N_TRAIN:].copy()
    bottom_train = to_long(train, bottom_ids)
    print(f"底层 SKU 数量 = {len(bottom_ids)}，训练天数 = {N_TRAIN}")

    # ---- 经典：AutoARIMA ----
    sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=-1)
    sf.fit(bottom_train)
    fc_sf = sf.predict(h=H, level=LEVELS)

    # ---- 深度学习：NHITS 全局模型 ----
    nhits = NHITS(h=H, input_size=28, max_steps=400)
    nf = NeuralForecast(models=[nhits], freq="D")
    nf.fit(bottom_train, prediction_intervals=PredictionIntervals(n_windows=2))
    fc_nf = nf.predict(level=LEVELS)

    print("\n[结果] 底层 SKU 平均概率预测质量（越低越好）")
    print(f"{'模型':<14}{'平均 MAE':>12}{'平均 CRPS':>12}")
    print("-" * 40)
    print(f"{'AutoARIMA':<14}{avg_mae(fc_sf, 'AutoARIMA', bottom_ids, test):>12.2f}"
          f"{avg_crps(fc_sf, 'AutoARIMA', bottom_ids, test):>12.2f}")
    print(f"{'NHITS':<14}{avg_mae(fc_nf, 'NHITS', bottom_ids, test):>12.2f}"
          f"{avg_crps(fc_nf, 'NHITS', bottom_ids, test):>12.2f}")
    print("\n说明：NHITS 是全局模型，SKU 越多、单序列历史越短，共享模式的收益越大；")
    print("      CRPS 同时衡量点预测与不确定性，比单独看 MAE 更全面。")


if __name__ == "__main__":
    main()
