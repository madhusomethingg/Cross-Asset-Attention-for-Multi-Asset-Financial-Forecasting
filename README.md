# Cross-Asset Attention for Multi-Asset Financial Forecasting

A deep learning system that predicts next-day log returns for multiple financial assets using a Cross-Asset Transformer, enabling the model to learn interdependencies between assets through self-attention.

---

## Overview

This project builds a unified forecasting framework for 20 financial assets, where all assets are modeled jointly instead of independently.

The key idea is that financial assets are interconnected, and capturing these relationships improves predictive performance. The Transformer architecture leverages self-attention to learn these cross-asset dependencies.

The model is evaluated against six strong baselines:

- Ridge Regression  
- ARIMA  
- Gradient Boosting  
- LSTM  
- GRU  
- TCN  

All models are tested on out-of-sample 2023 data.

---

## Results

| Metric | Transformer | Notes |
|--------|------------|------|
| Macro F1 | 0.495 | Among the best across all models |
| Directional Accuracy | 52.27% | Above random baseline |
| Backtest Sharpe | +1.12 | Best-performing strategy |
| Diebold-Mariano Test | p < 0.001 | Statistically superior |

---

## Core Contributions

### Cross-Asset Attention vs Correlation

The Transformer learns relationships between assets that differ significantly from traditional correlation-based approaches.

- Spearman correlation between attention and correlation matrices: ρ = −0.37  
- Attention patterns are 25× more stable over time  

This indicates that deep learning captures non-linear and hidden dependencies beyond classical statistics.

---

### Direction-Aware Class-Weighted Loss

A custom loss function was introduced to improve directional prediction:

- Focuses on UP/DOWN classification  
- Improves trading performance  
- Converts Sharpe from −0.61 to +1.12  

---

## Live Demo

The project includes a Streamlit dashboard with the following modules:

- Overview — Project pipeline and summary  
- Leaderboard — Comparison across all models  
- Backtest — Equity curves and Sharpe ratios  
- Attention vs Correlation — Interactive visualization  
- Live Prediction — Run model on selected dates  

---

## Setup

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git  
cd YOUR_REPO_NAME  

python -m venv .venv  
source .venv/bin/activate  

pip install -r requirements.txt  

---

## Run the App

streamlit run app/streamlit_app.py  

---

## Author

Madhumitha Rajagopal
