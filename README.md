# Hierarchical & Probabilistic Demand Forecasting

A portfolio project demonstrating three advanced techniques for retail demand forecasting, built on the Kaggle **M5** dataset (Walmart store × item sales). The hierarchy maps naturally to a restaurant scenario: **region → store → product (SKU)**.

The three techniques cover the core of production demand-forecasting systems:

1. **Hierarchical forecasting & reconciliation** — bottom-up / OLS / MinT, producing coherent forecasts across levels
2. **Intermittent & cold-start demand** — handling zero-inflated SKUs with Croston
3. **Probabilistic forecasting** — quantile regression (Pinball loss) and CRPS evaluation

---

## Why these three

| Technique | Problem it solves | Production context |
|---|---|---|
| Hierarchical reconciliation | Independent forecasts at each level don't add up (store sums ≠ total) | Inventory, staffing, and planning decisions must be mutually consistent |
| Intermittent demand | Many SKUs have long zero stretches; MSE/MAE are misleading | Replenishment and safety-stock for slow-moving products |
| Probabilistic forecasting | A point forecast hides uncertainty | Safety-stock and risk decisions need quantiles, not a single number |

---

## Repository structure

| File | Description | Data |
|---|---|---|
| `demo_hierarchical_forecast.py` | Self-contained intro: reconciliation + intermittent + probabilistic, with MinT implemented from scratch | Synthetic (runs in seconds) |
| `demo_m5_forecast.py` | Real-data version: full reconciliation + intermittent + probabilistic pipeline | M5 (auto-downloaded & cached) |
| `demo_neural_forecast.py` | Classical vs deep learning probabilistic comparison (AutoARIMA vs NHITS global model) | M5 |

---

## Key results (M5, CA state, 80 bottom-level SKUs)

### 1. Hierarchical reconciliation

| Level | Independent (incoherent) | bottom-up | OLS | MinT |
|---|---|---|---|---|
| S1 | 5.10 | **4.77** | 5.06 | 4.92 |
| R1 | 7.47 | **7.10** | 7.33 | 7.20 |
| total | 8.39 | 8.30 | 8.40 | 8.30 |

- Independent forecasts were incoherent (store sums ≠ total, cumulative gap **13.8** on synthetic; **234** on M5).
- Reconciliation guarantees coherence across all levels.
- On noisy real data, `bottom-up` can *underperform* direct aggregate forecasting — reconciliation's value is optimally combining information across levels (MinT/OLS), not blindly summing the bottom level.

### 2. Probabilistic forecasting (classical vs deep learning)

| Model | Avg. MAE | Avg. CRPS |
|---|---|---|
| AutoARIMA (per-series) | 8.84 | **6.43** |
| NHITS (global model) | **8.62** | 7.12 |

NHITS (a global deep model) is competitive on point accuracy; AutoARIMA's analytic intervals are better calibrated out of the box. NHITS's advantage grows with more series and shorter per-series history.

---

## Getting started

### Requirements
- Python 3.14+ (also works on 3.9–3.13)

### Install
```bash
pip install statsforecast neuralforecast datasetsforecast utilsforecast pandas numpy matplotlib
```

### Run
```bash
python demo_hierarchical_forecast.py   # intro, synthetic data
python demo_m5_forecast.py             # real M5 data (first run downloads ~50 MB)
python demo_neural_forecast.py         # AutoARIMA vs NHITS
```

---

## Techniques in detail

### Hierarchical forecasting & reconciliation

Forecasts must satisfy the summing constraint across the hierarchy (region → store → SKU). After fitting base models, reconcile with:

- **bottom-up**: `ỹ = S·b` (sum the bottom level)
- **OLS**: `G = (SᵀS)⁻¹Sᵀ`
- **MinT** (optimal): `G = (SᵀW⁻¹S)⁻¹SᵀW⁻¹`, minimizing the trace of the reconciled error covariance

`W` is estimated from base-forecast residuals with shrinkage. MinT reduces to OLS when errors are homoscedastic and uncorrelated.

### Intermittent & cold-start demand

For zero-inflated SKUs, MAE is misleading — predicting all zeros can yield the lowest MAE while being useless for replenishment. Croston/SBA model "demand size" and "inter-arrival interval" separately, giving an unbiased demand rate. Note: plain Croston assumes no seasonality; intermittent + seasonal demand needs seasonal intermittent models or seasonal adjustment.

### Probabilistic forecasting

Quantile forecasts are trained with Pinball loss:

`L_τ(y, q) = max(τ·(y − q), (τ − 1)·(y − q))`

Evaluated with **CRPS** (Continuous Ranked Probability Score), approximated from a dense grid of quantiles.

---

## Honest notes / limitations

- This is a learning-and-portfolio project, not a production system; results use small subsets for speed.
- The synthetic data generator mimics restaurant daily sales (weekend effect, weekly seasonality, intermittent SKUs) and is fully deterministic.
- Swap in real data by replacing the data-loading function; the reconciliation and probabilistic logic is reusable as-is.
