import json
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("DataFrame must have exactly one column to convert to Series.")
        return x.iloc[:, 0].astype(float)
    return pd.Series(x, dtype=float)


def align_series(strategy_returns, benchmark_returns=None) -> Tuple[pd.Series, Optional[pd.Series]]:
    s = _to_series(strategy_returns).dropna()
    if benchmark_returns is None:
        return s, None
    b = _to_series(benchmark_returns).dropna()
    df = pd.concat([s.rename("strategy"), b.rename("benchmark")], axis=1).dropna()
    return df["strategy"], df["benchmark"]


def calc_equity_curve(returns: pd.Series) -> pd.Series:
    returns = _to_series(returns).fillna(0.0)
    return (1.0 + returns).cumprod()


def calc_total_return(returns: pd.Series) -> float:
    returns = _to_series(returns).dropna()
    return float((1.0 + returns).prod() - 1.0)


def calc_cagr(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    returns = _to_series(returns).dropna()
    n = len(returns)
    if n == 0:
        return np.nan
    total = float((1.0 + returns).prod())
    years = n / trading_days
    if years <= 0 or total <= 0:
        return np.nan
    return float(total ** (1.0 / years) - 1.0)


def calc_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    returns = _to_series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    return float(np.sqrt(trading_days) * returns.std(ddof=1))


def calc_sharpe(returns: pd.Series, rf_daily: float = 0.0, trading_days: int = TRADING_DAYS) -> float:
    returns = _to_series(returns).dropna() - rf_daily
    vol = returns.std(ddof=1)
    if len(returns) < 2 or vol == 0 or np.isnan(vol):
        return np.nan
    return float(np.sqrt(trading_days) * returns.mean() / vol)


def calc_sortino(returns: pd.Series, rf_daily: float = 0.0, trading_days: int = TRADING_DAYS) -> float:
    returns = _to_series(returns).dropna() - rf_daily
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1)
    if len(downside) < 2 or downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float(np.sqrt(trading_days) * returns.mean() / downside_std)


def calc_mdd(returns: pd.Series) -> float:
    equity = calc_equity_curve(returns)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calc_calmar(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    cagr = calc_cagr(returns, trading_days=trading_days)
    mdd = calc_mdd(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(cagr / abs(mdd))


def calc_win_rate(returns: pd.Series) -> float:
    returns = _to_series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    return float((returns > 0).mean())


def calc_turnover(weights_df: pd.DataFrame) -> pd.Series:
    if weights_df is None or len(weights_df) == 0:
        return pd.Series(dtype=float)
    weights_df = weights_df.fillna(0.0).sort_index()
    turnover = weights_df.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights_df.iloc[0].abs().sum()
    return turnover.rename("turnover")


def calc_information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    s, b = align_series(strategy_returns, benchmark_returns)
    active = s - b
    te = active.std(ddof=1)
    if len(active) < 2 or te == 0 or np.isnan(te):
        return np.nan
    return float(np.sqrt(trading_days) * active.mean() / te)


def calc_alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series, rf_daily: float = 0.0, trading_days: int = TRADING_DAYS) -> Tuple[float, float]:
    s, b = align_series(strategy_returns, benchmark_returns)
    y = (s - rf_daily).to_numpy(dtype=float)
    x = (b - rf_daily).to_numpy(dtype=float)
    if len(y) < 2:
        return np.nan, np.nan
    x_var = np.var(x, ddof=1)
    if x_var == 0 or np.isnan(x_var):
        return np.nan, np.nan
    beta = np.cov(x, y, ddof=1)[0, 1] / x_var
    alpha_daily = y.mean() - beta * x.mean()
    alpha_annual = (1.0 + alpha_daily) ** trading_days - 1.0 if alpha_daily > -1 else np.nan
    return float(alpha_annual), float(beta)


def summarize_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    turnover_series: Optional[pd.Series] = None,
    rf_daily: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, float]:
    s = _to_series(strategy_returns).dropna()
    metrics = {
        "Total Return": calc_total_return(s),
        "CAGR": calc_cagr(s, trading_days=trading_days),
        "Sharpe": calc_sharpe(s, rf_daily=rf_daily, trading_days=trading_days),
        "Sortino": calc_sortino(s, rf_daily=rf_daily, trading_days=trading_days),
        "Max Drawdown": calc_mdd(s),
        "Calmar Ratio": calc_calmar(s, trading_days=trading_days),
        "Win Rate": calc_win_rate(s),
        "Volatility": calc_volatility(s, trading_days=trading_days),
    }

    if turnover_series is not None and len(turnover_series) > 0:
        t = _to_series(turnover_series).dropna()
        metrics["Average Turnover"] = float(t.mean())
        metrics["Median Turnover"] = float(t.median())
    else:
        metrics["Average Turnover"] = np.nan
        metrics["Median Turnover"] = np.nan

    if benchmark_returns is not None:
        b = _to_series(benchmark_returns).dropna()
        metrics["Information Ratio"] = calc_information_ratio(s, b, trading_days=trading_days)
        alpha, beta = calc_alpha_beta(s, b, rf_daily=rf_daily, trading_days=trading_days)
        metrics["Alpha"] = alpha
        metrics["Beta"] = beta
        metrics["Benchmark CAGR"] = calc_cagr(b, trading_days=trading_days)
    else:
        metrics["Information Ratio"] = np.nan
        metrics["Alpha"] = np.nan
        metrics["Beta"] = np.nan
        metrics["Benchmark CAGR"] = np.nan

    return metrics


def save_summary_json(metrics: Dict[str, float], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: (None if pd.isna(v) else float(v)) for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)
