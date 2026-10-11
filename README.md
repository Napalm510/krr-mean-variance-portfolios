# When Better Return Forecasts Do Not Produce Better Portfolios

Group project for **Machine Learning in Asset Pricing** (MSc Quantitative Finance and Insurance, University of Turin, 2025/26).

Machine-learning models for returns are usually ranked by forecast accuracy (MSE). This project asks a different question: **do better forecasts make better portfolios?** Stock-level Kernel Ridge Regression (KRR) forecasts are fed into a regularised mean-variance (MVE) allocation and compared with the naive 1/N portfolio.

## Method

- **Data:** Gu–Kelly–Xiu (2020) monthly U.S. equity characteristics from January 1990. The 50 largest stocks by market equity at the start date are used, with no survivorship filter, and 10 characteristics: size, value, momentum, change in momentum, profitability, investment, beta, idiosyncratic volatility, illiquidity and turnover.
- **Preprocessing:** missing values are filled with the cross-sectional median, characteristics are standardised cross-sectionally each month, and the sample is split chronologically 60 / 20 / 20 into train, validation and test.
- **Forecasting:** one KRR model per asset with **linear, polynomial and RBF kernels**. The ridge penalty λᵢ and the kernel parameters are tuned per asset on validation MSE.
- **Allocation:** `w = (Σ̂ + γI)⁻¹ R̂`, rescaled to unit gross exposure. γ is selected on the validation set.
- **Evaluation:** static out-of-sample test plus a rolling-window robustness check, against the 1/N benchmark.

## Key findings

| Kernel     | Test MSE | Test RMSE | Test MAE |
|------------|---------:|----------:|---------:|
| Linear     | 0.005134 | 0.071649  | 0.050385 |
| Polynomial | 0.005261 | 0.072532  | 0.051131 |
| RBF        | 0.005220 | 0.072252  | 0.051057 |

- RBF wins the validation stage (lowest mean validation MSE; chosen for 31 stocks vs 2 for polynomial and 0 for linear), yet the **linear kernel** has the best test accuracy and the best KRR-MVE portfolio.
- Small differences in forecast error turn into large differences in weights once Σ̂ is inverted.
- **No KRR-MVE strategy beats 1/N** in final wealth or Sharpe ratio. 1/N needs no expected-return estimate and no covariance inversion.

## Files

| File | Content |
|---|---|
| `KRR_MVE.ipynb` | Full pipeline: data, KRR, model selection, MVE portfolios, rolling check, γ sensitivity (executed, outputs included) |
| `Report.pdf` | Project report (11 pages) |

## Running it

The dataset is not included because it is about 3 GB. Download `datashare.csv` from the Gu–Kelly–Xiu replication data on [Dacheng Xiu's website](https://dachxiu.chicagobooth.edu/), place it in `data/datashare.csv`, then:

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook KRR_MVE.ipynb
```

## Authors

Mattia Vizzino, Ludovica Durante, Paolo Ponticiello, Marta Brusini, Eleonora Ferinando.
