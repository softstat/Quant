import argparse
import os
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_returns(csv_path: str, return_col: str = "daily_return") -> pd.Series:
    df = pd.read_csv(csv_path)

    if return_col not in df.columns:
        raise ValueError(f"'{return_col}' 컬럼이 CSV에 없습니다.")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    returns = pd.to_numeric(df[return_col], errors="coerce").dropna()

    if len(returns) < 30:
        raise ValueError("수익률 데이터가 너무 짧습니다. 최소 30개 이상 필요합니다.")

    return returns.reset_index(drop=True)


def calc_equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def calc_cagr(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    n = len(returns)
    total_return = float((1.0 + returns).prod())
    years = n / trading_days
    if years <= 0 or total_return <= 0:
        return np.nan
    return total_return ** (1.0 / years) - 1.0


def calc_sharpe(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    vol = returns.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return np.sqrt(trading_days) * returns.mean() / vol


def calc_sortino(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return np.sqrt(trading_days) * returns.mean() / downside_std


def calc_mdd(returns: pd.Series) -> float:
    equity = calc_equity_curve(returns)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calc_calmar(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    cagr = calc_cagr(returns, trading_days)
    mdd = calc_mdd(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr / abs(mdd)


def summarize_performance(returns: pd.Series) -> dict:
    return {
        "num_days": len(returns),
        "total_return": float((1.0 + returns).prod() - 1.0),
        "cagr": calc_cagr(returns),
        "sharpe": calc_sharpe(returns),
        "sortino": calc_sortino(returns),
        "mdd": calc_mdd(returns),
        "calmar": calc_calmar(returns),
        "volatility": float(np.sqrt(TRADING_DAYS) * returns.std(ddof=1)),
        "avg_daily_return": float(returns.mean()),
        "win_rate": float((returns > 0).mean()),
    }


def moving_block_bootstrap(
    returns: pd.Series,
    block_size: int = 20,
    n_boot: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    arr = returns.to_numpy()
    n = len(arr)

    if block_size <= 0:
        raise ValueError("block_size는 1 이상이어야 합니다.")
    if block_size > n:
        raise ValueError("block_size가 데이터 길이보다 클 수 없습니다.")

    starts = np.arange(0, n - block_size + 1)
    results = []

    for i in range(n_boot):
        sampled = []

        while len(sampled) < n:
            s = rng.choice(starts)
            block = arr[s:s + block_size]
            sampled.extend(block.tolist())

        sampled = np.array(sampled[:n], dtype=float)
        sampled_returns = pd.Series(sampled)

        results.append({
            "bootstrap_id": i + 1,
            "cagr": calc_cagr(sampled_returns),
            "sharpe": calc_sharpe(sampled_returns),
            "sortino": calc_sortino(sampled_returns),
            "mdd": calc_mdd(sampled_returns),
            "calmar": calc_calmar(sampled_returns),
            "total_return": float((1.0 + sampled_returns).prod() - 1.0),
            "volatility": float(np.sqrt(TRADING_DAYS) * sampled_returns.std(ddof=1)),
            "win_rate": float((sampled_returns > 0).mean()),
        })

    return pd.DataFrame(results)


def ci_summary(bootstrap_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    lower = alpha / 2
    upper = 1 - alpha / 2

    rows = []
    metric_cols = [c for c in bootstrap_df.columns if c != "bootstrap_id"]

    for col in metric_cols:
        series = bootstrap_df[col].dropna()
        rows.append({
            "metric": col,
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
            "median": float(series.median()),
            "ci_lower": float(series.quantile(lower)),
            "ci_upper": float(series.quantile(upper)),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Block bootstrap backtest summary")
    parser.add_argument("--input", type=str, required=True, help="CSV path with daily returns")
    parser.add_argument("--return-col", type=str, default="daily_return", help="Return column name")
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap simulations")
    parser.add_argument("--block-size", type=int, default=20, help="Block size for bootstrap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="bootstrap_output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    returns = load_returns(args.input, args.return_col)
    base_summary = summarize_performance(returns)
    base_df = pd.DataFrame([base_summary])

    boot_df = moving_block_bootstrap(
        returns=returns,
        block_size=args.block_size,
        n_boot=args.n_boot,
        random_state=args.seed,
    )

    ci_df = ci_summary(boot_df)

    base_path = os.path.join(args.output_dir, "base_performance.csv")
    boot_path = os.path.join(args.output_dir, "bootstrap_metrics.csv")
    ci_path = os.path.join(args.output_dir, "bootstrap_ci_summary.csv")

    base_df.to_csv(base_path, index=False)
    boot_df.to_csv(boot_path, index=False)
    ci_df.to_csv(ci_path, index=False)

    print("=== Base Performance ===")
    print(base_df.to_string(index=False))

    print("\n=== Bootstrap CI Summary ===")
    print(ci_df.to_string(index=False))

    print(f"\nSaved:")
    print(f"- {base_path}")
    print(f"- {boot_path}")
    print(f"- {ci_path}")


if __name__ == "__main__":
    main()