# Cryptocurrency Volatility Forecasting
### Machine Learning Approaches to BTC-USD Realized Volatility Prediction
---

## Overview

This project applies four forecasting models to predict one-step-ahead **realized volatility** of Bitcoin (BTC-USD) at hourly frequency. The models range from a simple persistence baseline to deep learning, allowing a structured comparison of complexity versus predictive gain.

| Model | Type | Library |
|---|---|---|
| Naive Baseline | Persistence (σ̂ₜ₊₁ = σₜ) | — |
| Random Forest | Ensemble / Tree-based | `scikit-learn` |
| LSTM | Recurrent Neural Network | `TensorFlow / Keras` |
| XGBoost | Gradient Boosting | `xgboost` |

---

## Results Summary

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Naive Baseline | 0.0314 | 0.0148 | 0.9216 |
| Random Forest | 0.0277 | 0.0154 | 0.9390 |
| LSTM | 0.0280 | 0.0177 | 0.9374 |
| **XGBoost** | **0.0258** | **0.0147** | **0.9469** |

XGBoost achieves the best performance across all three metrics. All ML models improve on the naive baseline, though the margin is modest — consistent with the strong volatility persistence (ARCH effects) present in high-frequency crypto data.

---

## Repository Structure

```
.
├── volatility_forecasting.ipynb   # Main Jupyter notebook (all models + plots)
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## Data

- **Asset:** BTC-USD
- **Source:** Yahoo Finance via `yfinance`
- **Frequency:** Hourly (`interval="1h"`)
- **Period:** 1 July 2024 → 1 September 2025 (18 months)
- **Split:** 14 months training · 4 months test (78/22)

No data files are stored in this repository. The notebook downloads data automatically at runtime via the `yfinance` API.

---

## Methodology

### Target Variable

One-step-ahead **realized volatility**, defined as the rolling standard deviation of log returns over a 30-hour window, annualised for hourly data:

$$\sigma_t^{RV} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} r_{t-i}^2} \cdot \sqrt{24 \times 365}$$

where $r_t = \ln(P_t / P_{t-1})$ and $n = 30$.

### Feature Engineering

The following features are constructed from the training data and used as model inputs:

- **Lagged log returns** — `returns_lag_1` to `returns_lag_5`
- **Lagged squared returns** — `sq_returns_lag_1` to `sq_returns_lag_5`
- **Lagged realized volatility** — `vol_lag_1` to `vol_lag_5`
- **Rolling volatility means** — 5-hour and 10-hour moving averages of realized vol
- **Rolling return statistics** — 5-hour mean and standard deviation of log returns
- **Garman-Klass volatility** — intra-bar range-based estimator (current + 5-hour MA)

All features are constructed using only information available at time $t$ to avoid look-ahead bias. Scalers are fit exclusively on training data.

### Garman-Klass Estimator

The Garman-Klass (1980) volatility estimator is used as an additional feature, leveraging OHLC bar data:

$$\sigma_{GK}^2 = 0.5 \cdot \left[\ln\frac{H}{L}\right]^2 - (2\ln 2 - 1) \cdot \left[\ln\frac{C}{O}\right]^2$$

Negative values (caused by bid-ask bounce at hourly resolution) are clamped to zero before taking the square root.

### Naive Baseline

The persistence model assumes volatility is unchanged from the prior period:

$$\hat{\sigma}_{t+1} = \sigma_t$$

This is equivalent to using `vol_lag_1` directly as the forecast. It serves as the lower-bound benchmark — any ML model that cannot beat it provides no value over a trivially simple rule.

### Random Forest

A `RandomForestRegressor` with 100 trees and maximum depth 10, trained directly on the tabular feature matrix. Feature importance is measured by mean decrease in impurity (MDI).

### LSTM

A single-layer LSTM with 50 hidden units, followed by a dropout layer (rate = 0.2) and two dense layers. Input sequences span a 24-hour lookback window. All features are scaled to [0, 1] using `MinMaxScaler` fit on training data only. Trained for 20 epochs with Adam optimiser and MSE loss.

### XGBoost

Gradient boosted trees with histogram-based splitting (`tree_method="hist"`), early stopping after 50 rounds of no improvement on the test set evaluation. Learning rate 0.05, max depth 6, subsample 0.8.

---

## Key Findings

**Volatility persistence dominates.** The naive baseline achieves R² = 0.92, reflecting the well-known clustering property of financial volatility. All ML models operate in the narrow improvement band above this floor.

**XGBoost is the best model.** It achieves the lowest RMSE (0.0258) and highest R² (0.9469), with a ~18% RMSE improvement over the naive baseline. Its regularisation allows it to distribute importance across multiple lag features rather than collapsing onto `vol_lag_1` alone.

**LSTM underperforms tree-based models.** Despite higher architectural complexity, the LSTM yields worse MAE (0.0177) than even the naive baseline. This is a consequence of MSE loss training making the model risk-averse — it predicts close to the conditional mean and systematically undershoots volatility spikes. This is a common finding in the literature for short-horizon volatility forecasting.

**`vol_lag_1` is the dominant feature.** Both Random Forest and XGBoost rank lagged realized volatility as the most important feature by a wide margin, consistent with GARCH(1,1) theory. GK volatility features contribute marginal additional signal.

**All models underpredict extreme events.** Volatility spikes (σ > 0.6) are consistently underpredicted across all models — a structural limitation of MSE-trained models and a direction for future work using asymmetric or quantile loss functions.

---

## Reproducibility

All results are fully reproducible. Random seeds are fixed globally before any model is instantiated:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.random.set_seed(SEED)
```

Data is downloaded at runtime from Yahoo Finance. Minor variation in LSTM results across hardware/OS is possible due to GPU non-determinism in TensorFlow, but results should be consistent on CPU.

---


### requirements.txt

```
yfinance>=0.2.40
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
tensorflow>=2.13
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

---

## Limitations & Future Work

- **MSE loss.** All models are trained to minimise squared error, which penalises large errors symmetrically and produces conservative forecasts during spike regimes. Quantile regression or asymmetric loss functions could improve tail performance.
- **No GARCH benchmark.** A GARCH(1,1) or GJR-GARCH model would be the natural econometric baseline alongside the naive model, and would strengthen the paper's claim that ML adds value.
- **Feature set.** Order book data, funding rates, or on-chain metrics could provide predictive signal not captured by price history alone.
- **Hyperparameter tuning.** RF and XGBoost use fixed hyperparameters. Time-series cross-validation (e.g. walk-forward) with a grid search would produce more robust model selection.

