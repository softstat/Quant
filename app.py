import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")

SIGNALS_PARQUET = os.path.join(MODELS_DIR, "latest_signals.parquet")
SIGNALS_CSV = os.path.join(MODELS_DIR, "latest_signals.csv")
PORTFOLIO_PARQUET = os.path.join(MODELS_DIR, "thematic_portfolio.parquet")
PORTFOLIO_CSV = os.path.join(MODELS_DIR, "thematic_portfolio.csv")
TEST_METRICS_JSON = os.path.join(MODELS_DIR, "test_metrics.json")
ABLATION_CSV = os.path.join(BASE_DIR, "ablation_results.csv")
BACKTEST_JSON = os.path.join(BASE_DIR, "data", "backtest_summary.json")
BACKTEST_RETURNS = os.path.join(BASE_DIR, "data", "backtest_strategy_returns.csv")

st.set_page_config(page_title="Quant Survival Dashboard", layout="wide")
st.title("Quant Survival × GNN × LLaMA Dashboard")

def run_command(cmd):
    proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_lines = []
    box = st.empty()
    for line in proc.stdout:
        output_lines.append(line)
        box.code("".join(output_lines))
    proc.wait()
    return proc.returncode

def load_any(parquet_path, csv_path):
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

with st.sidebar:
    st.header("Run")
    mode = st.selectbox("Mode", ["test", "sp500", "kospi", "custom"])
    epochs = st.number_input("Epochs", min_value=5, max_value=300, value=30, step=5)
    lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
    target_return = st.number_input("Target Return", min_value=0.01, max_value=0.50, value=0.10, step=0.01, format="%.2f")
    stop_loss = st.number_input("Stop Loss", min_value=-0.50, max_value=-0.01, value=-0.05, step=0.01, format="%.2f")
    custom_tickers = st.text_input("Custom tickers", "AAPL,MSFT,NVDA,TSLA")
    groq_key = st.text_input("GROQ_API_KEY", value=os.environ.get("GROQ_API_KEY", ""), type="password")

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    if st.button("Run Training"):
        cmd = [sys.executable, "train.py", "--mode", mode, "--epochs", str(epochs), "--lr", str(lr), "--target-return", str(target_return), "--stop-loss", str(stop_loss)]
        if mode == "custom":
            tickers = [t.strip() for t in custom_tickers.split(",") if t.strip()]
            if tickers:
                cmd += ["--tickers"] + tickers
        st.subheader("Training Log")
        code = run_command(cmd)
        st.success("Training finished." if code == 0 else "Training failed.")

    if st.button("Run Ablation"):
        cmd = [sys.executable, "ablation_study.py", "--mode", mode, "--epochs", str(epochs), "--lr", str(lr)]
        st.subheader("Ablation Log")
        code = run_command(cmd)
        st.success("Ablation finished." if code == 0 else "Ablation failed.")

    if st.button("Run Backtest"):
        cmd = [sys.executable, "advanced_backtest.py"]
        st.subheader("Backtest Log")
        code = run_command(cmd)
        st.success("Backtest finished." if code == 0 else "Backtest failed.")

tab1, tab2, tab3, tab4 = st.tabs(["Signals", "Thematic Portfolio", "Ablation", "Backtest"])

with tab1:
    df = load_any(SIGNALS_PARQUET, SIGNALS_CSV)
    st.subheader("Latest Signals")
    if df is not None and len(df) > 0:
        st.dataframe(df, use_container_width=True)
        chart_col = "expected_return" if "expected_return" in df.columns else ("entry_score" if "entry_score" in df.columns else None)
        if chart_col:
            st.bar_chart(df.set_index("ticker")[chart_col].head(20))
    else:
        st.info("No signal file found.")

    if os.path.exists(TEST_METRICS_JSON):
        with open(TEST_METRICS_JSON) as f:
            st.json(json.load(f))

with tab2:
    pdf = load_any(PORTFOLIO_PARQUET, PORTFOLIO_CSV)
    st.subheader("Thematic Portfolio")
    if pdf is not None and len(pdf) > 0:
        st.dataframe(pdf, use_container_width=True)
        if "weight" in pdf.columns:
            st.bar_chart(pdf.set_index("ticker")["weight"])
    else:
        st.info("No thematic portfolio file found.")

with tab3:
    st.subheader("Ablation Results")
    if os.path.exists(ABLATION_CSV):
        adf = pd.read_csv(ABLATION_CSV)
        st.dataframe(adf, use_container_width=True)
    else:
        st.info("No ablation_results.csv found.")

with tab4:
    st.subheader("Backtest")
    if os.path.exists(BACKTEST_JSON):
        with open(BACKTEST_JSON) as f:
            st.json(json.load(f))
    else:
        st.info("No backtest summary found.")

    if os.path.exists(BACKTEST_RETURNS):
        rdf = pd.read_csv(BACKTEST_RETURNS)
        if "strategy_return" in rdf.columns:
            rdf["equity"] = (1 + rdf["strategy_return"]).cumprod()
            st.line_chart(rdf.set_index("date")["equity"])
