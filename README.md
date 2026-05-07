# Cross-Asset Attention for Multi-Asset Financial Forecasting

DATA 612 — Deep Learning · Final Project · University of Maryland

**Eniyan Ezhilan Sumathi · Shri Varshan Periyaswamy · Dhanush Sambasivam · Madhumitha Rajagopal**

---

## What this is

A **Cross-Asset Transformer** that predicts next-day log returns for 20 financial
assets, trained jointly so the model learns relationships *between* assets
through self-attention. Compared against six baselines (Ridge, ARIMA, Gradient
Boosting, LSTM, GRU, TCN) on out-of-sample 2023 data.

### Headline results

| Metric | Transformer | Notes |
|---|---|---|
| Macro F1 | **0.495** | Tied for best of 7 models |
| Directional Accuracy | **52.27%** | Real edge above 50% coin-flip baseline |
| Backtest Sharpe | **+1.12** | Best of all *active* strategies |
| DM test vs Ridge / ARIMA / LSTM / TCN | **p < 0.001** | Statistically better on overall error |

### Novelty

We show that the Transformer's learned cross-asset attention captures a
**fundamentally different** kind of relationship than classical rolling
correlation: Spearman ρ = **−0.37** between the two matrices, with attention
being **25× more stable** day-to-day. Combined with a custom **Direction-Aware
+ Class-Weighted Loss** that flipped backtest Sharpe from −0.61 to +1.12,
this is the project's main contribution.

---

## Live demo

The Streamlit dashboard has 5 tabs:

1. **Overview** — elevator pitch + project pipeline
2. **Leaderboard** — test-set metrics across all 7 models
3. **Backtest** — equity curves and Sharpe ratios from real data
4. **Attention vs Correlation** — interactive heatmap comparison (the novelty)
5. **Live Prediction** — pick a date, the trained Transformer runs forward

### Run locally

```bash
git clone <this-repo>
cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Reproducing the results

```bash
python scripts/01_build_data.py            # download + featurize + split
python scripts/02_train_all.py             # train all 7 models with Combined Loss
python scripts/03_evaluate.py              # RMSE / MAE / DA / Diebold-Mariano
python scripts/04_backtest.py              # long/short top-5 strategy
python scripts/05_extract_attention.py     # save (T, N, N) cross-asset attention
python scripts/06_classification_report.py # UP/DOWN F1, precision, recall
python scripts/07_threshold_calibration.py # tune decision threshold on val set
```

All artifacts land in `artifacts/`. The Streamlit app reads from that folder.
