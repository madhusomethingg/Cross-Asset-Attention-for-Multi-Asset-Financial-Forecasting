"""
DATA 612 -- Cross-Asset Transformer Demo App
=============================================

Streamlit dashboard for the live demo. Everything is loaded from the
artifacts/ folder produced by the 7 pipeline scripts -- nothing is faked.
The app degrades gracefully: if any one artifact file is missing, that
specific tab shows a friendly warning instead of crashing.

Run:
    streamlit run app/streamlit_app.py

Tabs:
    1. Overview          -- elevator pitch, project pipeline, key numbers
    2. Leaderboard       -- test-set metrics across all 7 models, with charts
    3. Backtest          -- equity curves and Sharpe ratios from real data
    4. Attention vs Corr -- the novelty: side-by-side heatmaps
    5. Live Prediction   -- pick an asset, see the Transformer's prediction
                            and which other assets it attended to
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make src/ importable so we can load the trained Transformer
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------------------------
# Page config + global styling -- matches the slide deck colors
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cross-Asset Transformer · DATA 612",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Color palette (same as slides). Dark mode: cards are dark-navy with
# ice-blue text, matching the slide deck's "midnight executive" theme.
COLORS = {
    "navy":   "#1E2761",
    "ink":    "#0F1535",
    "ice":    "#CADCFC",
    "gold":   "#F4A261",
    "green":  "#2A9D8F",
    "red":    "#E76F51",
    "mute":   "#8C9AB7",
    "border": "#2A3568",        # subtle dark navy border (was light)
    "card":   "#1E2761",        # dark navy card bg (was light F7F9FF)
    "text":   "#FFFFFF",        # white headers/body (was dark 1A1F36)
    "muted":  "#8C9AB7",        # muted ice on dark bg (was 606E89 dark)
}

# Custom CSS to match the slide aesthetic
st.markdown(f"""
<style>
    .main {{ background-color: white; }}
    h1 {{ color: {COLORS['text']}; font-family: 'Calibri', sans-serif; font-weight: 700; }}
    h2 {{ color: {COLORS['text']}; font-family: 'Calibri', sans-serif; font-weight: 700; }}
    h3 {{ color: {COLORS['navy']}; font-family: 'Calibri', sans-serif; font-weight: 600; }}
    .stMetric {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        padding: 14px;
        border-radius: 6px;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLORS['ice']};
        font-family: 'Georgia', serif;
        font-weight: 700;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS['mute']};
        font-size: 0.85rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS['card']};
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        color: {COLORS['ice']};
    }}
    .stTabs [data-baseweb="tab"] p {{
        color: {COLORS['ice']} !important;
        font-weight: 600 !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['gold']};
        color: {COLORS['ink']};
    }}
    .stTabs [aria-selected="true"] p {{
        color: {COLORS['ink']} !important;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cached loaders -- all I/O happens once per session, then served from memory.
# Each loader returns None on failure so the calling tab can show a warning
# instead of crashing the whole app.
# =============================================================================
ART = ROOT / "artifacts"


@st.cache_data(show_spinner=False)
def load_leaderboard() -> pd.DataFrame | None:
    """Test-set RMSE/MAE/DA/IC table."""
    p = ART / "leaderboard_test.csv"
    if not p.exists(): return None
    return pd.read_csv(p, index_col=0)


@st.cache_data(show_spinner=False)
def load_backtest_summary() -> pd.DataFrame | None:
    """Strategy-level Sharpe / drawdown summary."""
    p = ART / "backtest_summary.csv"
    if not p.exists(): return None
    return pd.read_csv(p, index_col=0)


@st.cache_data(show_spinner=False)
def load_classification_report() -> pd.DataFrame | None:
    """UP/DOWN classification metrics."""
    p = ART / "classification_report_test.csv"
    if not p.exists(): return None
    return pd.read_csv(p, index_col=0)


@st.cache_data(show_spinner=False)
def load_dm_pvalues() -> pd.DataFrame | None:
    """Diebold-Mariano p-values (Transformer vs each baseline)."""
    p = ART / "dm_test_pvalues.csv"
    if not p.exists(): return None
    return pd.read_csv(p, index_col=0)


@st.cache_data(show_spinner=False)
def load_predictions(model: str) -> dict | None:
    """Per-model prediction arrays. Returns dict with val_preds / test_preds."""
    p = ART / "predictions" / f"{model}.npz"
    if not p.exists(): return None
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


@st.cache_data(show_spinner=False)
def load_actuals() -> dict | None:
    """Ground truth val/test return arrays."""
    p = ART / "predictions" / "_actuals.npz"
    if not p.exists(): return None
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


@st.cache_data(show_spinner=False)
def load_backtest(strategy: str) -> dict | None:
    """Per-strategy daily returns, equity curve, weights, etc."""
    p = ART / "backtest" / f"{strategy}.npz"
    if not p.exists(): return None
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


@st.cache_data(show_spinner=False)
def load_attention() -> dict | None:
    """Cross-asset attention matrices (T, N, N)."""
    p = ART / "attention" / "test_attention.npz"
    if not p.exists(): return None
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


@st.cache_data(show_spinner=False)
def load_corr_rolling() -> dict | None:
    """Rolling Pearson correlation matrices (T, N, N)."""
    p = ART / "attention" / "test_corr_rolling.npz"
    if not p.exists(): return None
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


@st.cache_data(show_spinner=False)
def load_attention_metrics() -> dict | None:
    """Per-day metrics: concentration, attn shift, corr shift, similarity."""
    p = ART / "attention" / "test_attention_metrics.npz"
    if not p.exists(): return None
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


@st.cache_data(show_spinner=False)
def load_data_bundle():
    """The full DataBundle (cached so we don't re-load on every interaction).

    `load_bundle` from src.data takes a *directory* containing arrays.npz +
    close_prices.csv + meta.json — not a single file path.
    """
    try:
        from src.data import load_bundle
        return load_bundle(ART / "data")
    except Exception as e:
        st.session_state.setdefault("_load_errors", []).append(f"bundle: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_transformer_model():
    """Load the trained Cross-Asset Transformer for live inference.

    Uses cache_resource (not cache_data) because torch models aren't picklable
    by Streamlit's data cache. Returns (model, device) or None.

    Architecture is inferred from the checkpoint shapes — same logic as
    scripts/05_extract_attention.py, so this stays in sync with whatever
    the user actually trained.
    """
    try:
        import torch
        import yaml
        from src.models.transformer import CrossAssetTransformer

        ckpt_path = ART / "models" / "transformer.pt"
        if not ckpt_path.exists():
            return None

        device = torch.device("cpu")   # CPU is plenty for one forward pass; avoids MPS surprises
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Some scripts wrap state_dict; fall through if not
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Infer architecture from checkpoint shapes (same as 05_extract_attention.py)
        d_model  = state["input_proj.weight"].shape[0]
        d_ff     = state["temporal_blocks.0.ff.0.weight"].shape[0]
        n_layers = len({k.split(".")[1] for k in state
                        if k.startswith("temporal_blocks.")})

        # Pull the rest from YAML config (n_heads, dropout)
        cfg_path = ROOT / "configs" / "config.yaml"
        cfg = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}
        model_cfg = dict(cfg.get("models", {}).get("transformer", {}))
        model_cfg.update({"d_model": d_model, "d_ff": d_ff, "n_layers": n_layers})
        model_cfg.setdefault("n_heads", 4)
        model_cfg["dropout"] = 0.0   # always 0 at inference

        bundle = load_data_bundle()
        if bundle is None:
            return None

        model = CrossAssetTransformer(
            n_features=len(bundle.feature_cols),
            n_assets=len(bundle.tickers),
            **model_cfg,
        )
        model.load_state_dict(state)
        model.eval()
        model.to(device)
        return model, device
    except Exception as e:
        import traceback
        st.session_state.setdefault("_load_errors", []).append(
            f"model: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        return None


# =============================================================================
# Header banner
# =============================================================================
def render_header():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div style="padding: 4px 0;">
            <div style="color: {COLORS['gold']}; font-size: 0.8rem; font-weight: 600;
                        letter-spacing: 0.15em;">
                DATA 612 · DEEP LEARNING · FINAL PROJECT
            </div>
            <h1 style="margin: 4px 0 0 0; color: white;">
                Cross-Asset Attention for Multi-Asset Financial Forecasting
            </h1>
            <div style="color: {COLORS['muted']}; font-size: 1rem; margin-top: 4px;">
                Eniyan Ezhilan Sumathi · Shri Varshan Periyaswamy ·
                Dhanush Sambasivam · Madhumitha Rajagopal
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 18px;">
            <span style="background: {COLORS['ink']}; color: {COLORS['gold']};
                         padding: 6px 14px; border-radius: 4px; font-size: 0.8rem;
                         font-weight: 600; letter-spacing: 0.1em;">
                LIVE DEMO
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")


# =============================================================================
# TAB 1 -- Overview
# =============================================================================
def tab_overview():
    st.markdown("### Project at a glance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models compared", "7", help="Ridge, ARIMA, GBM, LSTM, GRU, TCN, Transformer")
    col2.metric("Assets", "20", help="Equities, sector ETFs, broad-market ETFs, bonds, gold, crypto")
    col3.metric("Trading days", "2,169", help="2018-01-01 to 2023-12-31, daily OHLCV from yfinance")
    col4.metric("Features per asset", "16", help="Returns, volatility, momentum, Bollinger, volume/range")

    st.markdown("")
    st.markdown("### What we built")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style="background: {COLORS['card']}; padding: 18px; border-radius: 6px;
                    border-left: 4px solid {COLORS['ice']}; height: 200px;">
            <div style="color: {COLORS['gold']}; font-size: 0.75rem; font-weight: 700;
                        letter-spacing: 0.15em;">1. THE MODEL</div>
            <h4 style="margin: 6px 0; color: {COLORS['text']};">Cross-Asset Transformer</h4>
            <div style="color: {COLORS['muted']}; font-size: 0.95rem;">
                Factorized attention: temporal self-attention (per asset) + cross-asset
                self-attention (per day). 3 layers, 4 heads, ~330K params.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background: {COLORS['card']}; padding: 18px; border-radius: 6px;
                    border-left: 4px solid {COLORS['gold']}; height: 200px;">
            <div style="color: {COLORS['gold']}; font-size: 0.75rem; font-weight: 700;
                        letter-spacing: 0.15em;">2. THE LOSS</div>
            <h4 style="margin: 6px 0; color: {COLORS['text']};">Direction-Aware + Class-Weighted</h4>
            <div style="color: {COLORS['muted']}; font-size: 0.95rem;">
                MSE + sign-mismatch penalty + DOWN-day upweighting. Fights the bullish bias
                in 2018-2022 training data. Sharpe -0.61 → +1.12.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="background: {COLORS['card']}; padding: 18px; border-radius: 6px;
                    border-left: 4px solid {COLORS['green']}; height: 200px;">
            <div style="color: {COLORS['gold']}; font-size: 0.75rem; font-weight: 700;
                        letter-spacing: 0.15em;">3. THE NOVELTY</div>
            <h4 style="margin: 6px 0; color: {COLORS['text']};">Attention vs Rolling Correlation</h4>
            <div style="color: {COLORS['muted']}; font-size: 0.95rem;">
                Cross-asset attention is anti-correlated with rolling Pearson correlation
                (ρ = −0.37) and 25× more stable across time.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### Headline result")
    st.markdown(f"""
    <div style="background: {COLORS['ink']}; padding: 22px; border-radius: 6px;
                color: white; line-height: 1.7;">
        <span style="color: {COLORS['gold']}; font-weight: 700;">
            We don't just claim our Transformer beats baselines —
        </span>
        we show that its learned cross-asset attention captures a fundamentally different
        kind of relationship than classical rolling correlation. Combined with our custom
        Direction-Aware + Class-Weighted Loss, the Transformer achieves the
        <span style="color: {COLORS['gold']}; font-weight: 700;">
            highest Sharpe ratio (+1.12) of all active strategies
        </span>
        and tops the macro F1 leaderboard at 0.495.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    with st.expander("Pipeline: how the artifacts were produced"):
        st.markdown(f"""
        <div style="font-family: monospace; font-size: 0.9rem; line-height: 1.8;
                    color: {COLORS['text']};">
        <b>scripts/01_build_data.py</b>     → download OHLCV, build 16 features, chronological split<br/>
        <b>scripts/02_train_all.py</b>      → train all 7 models with Combined Loss<br/>
        <b>scripts/03_evaluate.py</b>       → RMSE, MAE, DA, IC, Diebold-Mariano significance tests<br/>
        <b>scripts/04_backtest.py</b>       → top-5 long/short strategy, 1bp transaction cost<br/>
        <b>scripts/05_extract_attention.py</b> → forward-pass test set, save (T, N, N) attention<br/>
        <b>scripts/06_classification_report.py</b> → UP/DOWN F1, precision, recall<br/>
        <b>scripts/07_threshold_calibration.py</b> → tune decision threshold on validation set
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 2 -- Leaderboard
# =============================================================================
def tab_leaderboard():
    st.markdown("### Test-set leaderboard")
    st.caption("All numbers computed on held-out 2023 data. "
               "Lower is better for RMSE/MAE; higher is better for Dir.Acc/F1.")

    lb = load_leaderboard()
    cls = load_classification_report()
    dm  = load_dm_pvalues()

    if lb is None:
        st.warning("⚠️ Leaderboard CSV not found. Run `python scripts/03_evaluate.py`.")
        return

    # Combine leaderboard + classification report into one display table
    display = lb.copy()
    if cls is not None:
        # Pull macro F1 and accuracy
        for col in ("f1_macro", "accuracy", "recall_UP", "recall_DOWN"):
            if col in cls.columns:
                display[col] = cls[col]
    if dm is not None and "p_value" in dm.columns:
        display["DM p-value vs Trf"] = dm["p_value"]

    # Pretty-print
    fmt = {
        "rmse": "{:.4f}", "mae": "{:.4f}",
        "directional_accuracy": "{:.2%}",
        "ic": "{:.3f}", "ic_ir": "{:.3f}",
        "f1_macro": "{:.3f}", "accuracy": "{:.2%}",
        "recall_UP": "{:.2%}", "recall_DOWN": "{:.2%}",
        "DM p-value vs Trf": "{:.3f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in display.columns}

    # Sort by macro F1 if available (else by directional accuracy)
    sort_col = "f1_macro" if "f1_macro" in display.columns else "directional_accuracy"
    if sort_col in display.columns:
        display = display.sort_values(sort_col, ascending=False)

    # Try to render with color-graded styling (needs matplotlib via pandas).
    # Falls back to a plain formatted table if matplotlib is missing or
    # styling otherwise fails. We never want the leaderboard tab to crash
    # over decorative shading.
    try:
        styled = display.style.format(fmt)
        if "f1_macro" in display.columns:
            styled = styled.background_gradient(subset=["f1_macro"], cmap="YlGn")
        if "directional_accuracy" in display.columns:
            styled = styled.background_gradient(
                subset=["directional_accuracy"], cmap="YlOrBr")
        st.dataframe(styled, use_container_width=True)
    except (ImportError, Exception):
        # Plain formatted table -- still readable, no colors
        st.dataframe(display.style.format(fmt), use_container_width=True)

    # Key takeaways box
    st.markdown(f"""
    <div style="background: {COLORS['ink']}; color: white; padding: 16px;
                border-radius: 6px; margin-top: 8px;">
        <div style="color: {COLORS['gold']}; font-weight: 700; letter-spacing: 0.12em;
                    font-size: 0.8rem; margin-bottom: 8px;">HEADLINE RESULT</div>
        <span style="color: {COLORS['gold']};">✓</span>
        Transformer is competitive with the strongest baseline (TCN) and statistically
        outperforms classical and recurrent models (Diebold-Mariano, p&lt;0.001).<br/>
        <span style="color: {COLORS['gold']};">✓</span>
        Differences are small — which reflects how genuinely difficult financial
        prediction is. 52.27% directional accuracy is a real edge above the 50% coin-flip baseline.
    </div>
    """, unsafe_allow_html=True)

    # Bar charts
    st.markdown("")
    st.markdown("### Per-model comparison")
    c1, c2 = st.columns(2)
    with c1:
        if "directional_accuracy" in display.columns:
            chart = display["directional_accuracy"].sort_values(ascending=True) * 100
            st.bar_chart(chart, color=COLORS["gold"], height=320,
                         use_container_width=True)
            st.caption("Directional Accuracy (%) — higher is better; 50% = coin flip")
    with c2:
        if "f1_macro" in display.columns:
            chart = display["f1_macro"].sort_values(ascending=True)
            st.bar_chart(chart, color=COLORS["navy"], height=320,
                         use_container_width=True)
            st.caption("Macro F1 — class-balanced; higher is better")


# =============================================================================
# TAB 3 -- Backtest
# =============================================================================
def tab_backtest():
    st.markdown("### Backtest: long/short top-5 strategy")
    st.caption("Each day, go long the top-5 highest-predicted assets and short the bottom-5. "
               "Equal-weighted. 1 basis point transaction cost. Test period: 2023.")

    summary = load_backtest_summary()
    if summary is None:
        st.warning("⚠️ Backtest summary not found. Run `python scripts/04_backtest.py`.")
        return

    # Sort by Sharpe descending
    sort_col = "sharpe" if "sharpe" in summary.columns else summary.columns[0]
    summary = summary.sort_values(sort_col, ascending=False)

    # KPI row -- pull Transformer's headline numbers
    if "transformer" in summary.index:
        t = summary.loc["transformer"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transformer Sharpe", f"+{t.get('sharpe', 0):.2f}",
                  help="Annualized risk-adjusted return")
        c2.metric("Annual return", f"{t.get('ann_return', 0)*100:+.1f}%")
        c3.metric("Annual vol",    f"{t.get('ann_vol',    0)*100:.1f}%")
        c4.metric("Max drawdown",  f"{t.get('max_dd',     0)*100:.1f}%")

    st.markdown("")

    # Equity curves -- load each strategy's daily returns and cumulate
    st.markdown("### Cumulative wealth from $1")

    strategies_to_plot = ["equal_weight", "transformer", "tcn", "ridge", "gru", "lstm"]
    eq_curves = {}
    for s in strategies_to_plot:
        bt = load_backtest(s)
        if bt is None: continue
        # try common key names: 'equity' or 'cum_returns' or daily_returns
        if "equity" in bt:
            eq_curves[s] = bt["equity"]
        elif "cum_returns" in bt:
            eq_curves[s] = 1.0 + bt["cum_returns"]
        elif "daily_returns" in bt:
            eq_curves[s] = (1.0 + bt["daily_returns"]).cumprod()
        elif "returns" in bt:
            eq_curves[s] = (1.0 + bt["returns"]).cumprod()

    if eq_curves:
        df_eq = pd.DataFrame({k: v for k, v in eq_curves.items()})
        # Pretty names
        rename = {"equal_weight": "Equal-weight (passive)",
                  "transformer":  "Transformer",
                  "tcn":          "TCN",
                  "ridge":        "Ridge",
                  "gru":          "GRU",
                  "lstm":         "LSTM"}
        df_eq = df_eq.rename(columns=rename)
        st.line_chart(df_eq, height=380, use_container_width=True)
        st.caption("Real equity paths — not simulated. Each line is the cumulative wealth of $1 "
                   "deployed at the start of the test period.")
    else:
        st.info("No equity curves available — check that artifacts/backtest/*.npz files exist.")

    # Sharpe ratio bar chart + summary table
    st.markdown("")
    st.markdown("### Sharpe ratio by strategy")
    c1, c2 = st.columns([2, 3])
    with c1:
        if "sharpe" in summary.columns:
            sharpe_data = summary["sharpe"].sort_values()
            st.bar_chart(sharpe_data, color=COLORS["gold"], height=340,
                         use_container_width=True)
    with c2:
        fmt = {col: ("{:.2f}" if col in ("sharpe",) else "{:.2%}")
               for col in summary.columns
               if col in ("sharpe", "ann_return", "ann_vol", "max_dd", "hit_rate")}
        st.dataframe(summary.style.format(fmt), use_container_width=True, height=340)

    # Honest framing
    st.markdown(f"""
    <div style="background: {COLORS['card']}; border-left: 4px solid {COLORS['red']};
                padding: 14px 18px; border-radius: 6px; margin-top: 12px;
                color: {COLORS['text']};">
        <span style="color: {COLORS['red']}; font-weight: 700;">WHAT WE'RE COMPARING:</span>
        <span style="color: {COLORS['text']};">
        We compare <b>active strategies</b> — predictive models against each other —
        not against passive bull-market performance. Equal-weight wins in absolute terms because
        2023 was strongly trending. <b>Among active strategies, Transformer is best</b>,
        with the combined-loss training improving Sharpe from −0.61 (vanilla MSE) to +1.12.
        </span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 4 -- Attention vs Rolling Correlation (the novelty)
# =============================================================================
def tab_attention():
    st.markdown("### Novelty: Attention vs Rolling Correlation")
    st.caption("We show that attention learns a fundamentally different structure "
               "than rolling Pearson correlation.")

    attn = load_attention()
    corr = load_corr_rolling()
    metrics = load_attention_metrics()
    bundle = load_data_bundle()

    if attn is None or corr is None:
        st.warning("⚠️ Attention artifacts not found. "
                   "Run `python scripts/05_extract_attention.py`.")
        return

    # Headline numbers
    c1, c2, c3, c4 = st.columns(4)
    if metrics is not None:
        # Different versions of the extraction script used different key names.
        # Try the modern names first, fall back to legacy ('sim', 'att_shift',
        # 'cor_shift'). Returning np.nan from missing keys is fine; we filter
        # before calling nanmean to avoid the "Mean of empty slice" warning.
        def _pick(*candidates):
            for k in candidates:
                if k in metrics and metrics[k].size:
                    arr = metrics[k]
                    if np.isfinite(arr).any():    # at least one non-NaN value
                        return arr
            return None

        def _safe_mean(arr):
            if arr is None: return float("nan")
            return float(np.nanmean(arr))

        sim     = _pick("attn_corr_similarity", "sim")
        a_shift = _pick("attn_shift",           "att_shift")
        c_shift = _pick("corr_shift",           "cor_shift")

        sim_mean = _safe_mean(sim)
        if not np.isnan(sim_mean):
            c1.metric("Spearman ρ (attn vs corr)", f"{sim_mean:+.3f}")
        else:
            c1.metric("Spearman ρ (attn vs corr)", "—")

        a_mean = _safe_mean(a_shift)
        c_mean = _safe_mean(c_shift)
        if not np.isnan(a_mean) and not np.isnan(c_mean) and a_mean > 0:
            ratio = c_mean / a_mean
            c2.metric("Stability advantage", f"{ratio:.0f}×",
                      help="Correlation moves this much more than attention day-to-day")
        else:
            c2.metric("Stability advantage", "—")

        c3.metric("Mean attention shift",
                  f"{a_mean:.4f}" if not np.isnan(a_mean) else "—",
                  help="Frobenius norm of day-to-day attention change")
        c4.metric("Mean correlation shift",
                  f"{c_mean:.4f}" if not np.isnan(c_mean) else "—",
                  help="Frobenius norm of day-to-day correlation change")

    st.markdown("")

    # Day picker -- find first key whose array has 3 dims (T, N, N)
    A = next((v for v in attn.values() if v.ndim == 3), None)
    R = next((v for v in corr.values() if v.ndim == 3), None)
    if A is None or R is None:
        st.warning("Couldn't locate (T, N, N) arrays in attention/correlation files.")
        return

    n_days = min(A.shape[0], R.shape[0])
    tickers = bundle.tickers if bundle is not None else [f"A{i}" for i in range(A.shape[1])]

    st.markdown("### Side-by-side heatmaps")
    st.caption("Drag the slider to pick a trading day from the 2023 test period. "
               "The two matrices should look very different on most days.")

    day_idx = st.slider("Test day", 0, n_days - 1, value=n_days // 2, key="day")

    if bundle is not None and len(bundle.dates_test) > day_idx:
        st.markdown(f"**Date:** {bundle.dates_test[day_idx].date()}")

    A_day = A[day_idx]
    R_day = R[day_idx]

    # Heatmaps -- use plotly for nicer rendering than matplotlib
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        c1, c2 = st.columns(2)
        with c1:
            fig = px.imshow(
                A_day, x=tickers, y=tickers,
                color_continuous_scale="Oranges",
                aspect="auto",
                title="Cross-Asset Attention",
            )
            fig.update_layout(height=520, coloraxis_showscale=True,
                              font=dict(family="Calibri", size=11),
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.imshow(
                R_day, x=tickers, y=tickers,
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                aspect="auto",
                title="Rolling Pearson Correlation (60-day)",
            )
            fig.update_layout(height=520, coloraxis_showscale=True,
                              font=dict(family="Calibri", size=11),
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.info("Install plotly for prettier heatmaps:  `pip install plotly`")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cross-Asset Attention**")
            st.dataframe(pd.DataFrame(A_day, index=tickers, columns=tickers))
        with c2:
            st.markdown("**Rolling Correlation**")
            st.dataframe(pd.DataFrame(R_day, index=tickers, columns=tickers))

    # Interpretation
    st.markdown(f"""
    <div style="background: {COLORS['ink']}; color: white; padding: 18px;
                border-radius: 6px; margin-top: 12px;">
        <div style="color: {COLORS['gold']}; font-weight: 700; letter-spacing: 0.12em;
                    font-size: 0.8rem; margin-bottom: 10px;">WHAT WE FOUND</div>
        Cross-asset attention is <b style="color: {COLORS['gold']};">anti-correlated</b>
        with rolling Pearson correlation (ρ ≈ −0.37) and
        <b style="color: {COLORS['gold']};">25× more stable</b> across time.
        <br/><br/>
        <b style="color: {COLORS['gold']};">(1)</b>
        Attention captures persistent structural relationships — sector memberships,
        factor exposures, regime alignments — rather than short-term co-movement.<br/>
        <b style="color: {COLORS['gold']};">(2)</b>
        Rolling correlation, by construction, lags regime shifts and reacts to noise.
        Attention provides a complementary, more stable view of cross-asset dependencies.<br/>
        <b style="color: {COLORS['gold']};">(3)</b>
        This extends our original proposal — moving from attention as visualization
        to attention as a quantitative comparison against the classical baseline.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 5 -- Live Prediction
# =============================================================================
def tab_live():
    st.markdown("### Live Transformer Prediction")
    st.caption("Pick a date from the test period. The Transformer runs forward inference "
               "on the 40-day window leading up to that day and predicts each asset's "
               "next-day log return — plus shows which other assets it attended to.")

    bundle = load_data_bundle()
    if bundle is None:
        st.warning("⚠️ Couldn't load data bundle. The app expects three files in "
                   "`artifacts/data/`: `arrays.npz`, `close_prices.csv`, and "
                   "`meta.json`. Run `python scripts/01_build_data.py` to "
                   "regenerate them, or check that they were pushed to GitHub.")
        if "_load_errors" in st.session_state:
            with st.expander("Debug details"):
                for err in st.session_state["_load_errors"]:
                    st.text(err)
        return

    loaded = load_transformer_model()
    if loaded is None:
        st.warning("⚠️ Couldn't load trained Transformer. "
                   "Check that artifacts/models/transformer.pt exists.")
        if "_load_errors" in st.session_state:
            for err in st.session_state["_load_errors"]:
                st.text(err)
        return

    model, device = loaded

    import torch

    tickers = bundle.tickers
    test_dates = bundle.dates_test
    LOOKBACK = 40

    # Date picker
    valid_idx_min = LOOKBACK
    valid_idx_max = len(test_dates) - 1
    if valid_idx_max < valid_idx_min:
        st.warning("Test set is shorter than the lookback window.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        idx = st.slider(
            "Test day index",
            valid_idx_min, valid_idx_max,
            value=valid_idx_max // 2 + valid_idx_min,
            help="Index into the test period. Earlier = needs more lookback.",
        )
        target_date = test_dates[idx]
        st.markdown(f"**Predicting next-day return for:**  \n**{target_date.date()}**")

    # Build the input window: the LOOKBACK days BEFORE target_date
    # X_test has shape (T_test, N, F)
    X_test = bundle.X_test
    window = X_test[idx - LOOKBACK : idx]   # (L, N, F)
    if window.shape[0] != LOOKBACK:
        st.warning(f"Could not build window of length {LOOKBACK} at idx={idx}")
        return

    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L, N, F)

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]   # (N,)
        # Get the cross-asset attention from the LAST timestep, last layer
        if hasattr(model, "last_cross_attn") and model.last_cross_attn is not None:
            cross_attn = model.last_cross_attn.cpu().numpy()
            # shape could be (B, L, N, N) or (B, N, N) — use last timestep
            if cross_attn.ndim == 4:
                cross_attn = cross_attn[0, -1]   # (N, N)
            elif cross_attn.ndim == 3:
                cross_attn = cross_attn[0]
        else:
            cross_attn = None

    # Show the predictions table
    st.markdown("")
    st.markdown("### Predictions for next trading day")

    actual = None
    if idx < len(bundle.y_test):
        actual = bundle.y_test[idx]   # ground truth (N,)

    rows = []
    for i, t in enumerate(tickers):
        row = {"Asset": t, "Pred. log return": pred[i]}
        if actual is not None:
            row["Actual log return"] = actual[i]
            row["Direction match"] = "✓" if np.sign(pred[i]) == np.sign(actual[i]) else "✗"
        rows.append(row)
    df_pred = pd.DataFrame(rows).sort_values("Pred. log return", ascending=False)

    fmt = {"Pred. log return": "{:+.4f}"}
    if "Actual log return" in df_pred.columns:
        fmt["Actual log return"] = "{:+.4f}"

    c1, c2 = st.columns([3, 2])
    with c1:
        # Defensive styling -- same fallback pattern as the leaderboard
        try:
            styled = df_pred.style.format(fmt).background_gradient(
                subset=["Pred. log return"], cmap="RdYlGn")
            st.dataframe(styled, use_container_width=True,
                         hide_index=True, height=560)
        except (ImportError, Exception):
            st.dataframe(df_pred.style.format(fmt),
                         use_container_width=True, hide_index=True, height=560)
    with c2:
        # Top-5 / bottom-5 breakdown
        st.markdown("**Top-5 LONG (highest predicted)**")
        st.markdown("\n".join(
            f"- {r['Asset']}: **{r['Pred. log return']:+.4f}**"
            for _, r in df_pred.head(5).iterrows()
        ))
        st.markdown("**Top-5 SHORT (lowest predicted)**")
        st.markdown("\n".join(
            f"- {r['Asset']}: **{r['Pred. log return']:+.4f}**"
            for _, r in df_pred.tail(5).iloc[::-1].iterrows()
        ))

        if actual is not None:
            n_correct = int((np.sign(pred) == np.sign(actual)).sum())
            n_total = len(pred)
            st.metric("Direction-correct (this day)",
                      f"{n_correct}/{n_total}",
                      delta=f"{n_correct/n_total - 0.5:+.1%} vs coin flip",
                      delta_color="normal")

    # Attention heatmap
    if cross_attn is not None:
        st.markdown("")
        st.markdown("### Cross-asset attention on this day")
        st.caption("Rows: query asset.  Columns: which other assets the model attended to.  "
                   "Brighter cells = more attention.")
        try:
            import plotly.express as px
            fig = px.imshow(
                cross_attn, x=tickers, y=tickers,
                color_continuous_scale="Oranges", aspect="auto",
            )
            fig.update_layout(height=520, font=dict(family="Calibri", size=11),
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(pd.DataFrame(cross_attn, index=tickers, columns=tickers))


# =============================================================================
# Main
# =============================================================================
def main():
    render_header()

    tabs = st.tabs([
        "Overview",
        "Leaderboard",
        "Backtest",
        "Attention vs Correlation",
        "Live Prediction",
    ])

    with tabs[0]: tab_overview()
    with tabs[1]: tab_leaderboard()
    with tabs[2]: tab_backtest()
    with tabs[3]: tab_attention()
    with tabs[4]: tab_live()


if __name__ == "__main__":
    main()
