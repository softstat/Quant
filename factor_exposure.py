import argparse
import numpy as np
import pandas as pd


def annualize_return(monthly_returns: pd.Series) -> float:
    if len(monthly_returns) == 0:
        return np.nan
    total = (1 + monthly_returns).prod()
    years = len(monthly_returns) / 12
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1 / years) - 1


def annualize_vol(monthly_returns: pd.Series) -> float:
    return monthly_returns.std(ddof=1) * np.sqrt(12)


def sharpe_ratio(monthly_returns: pd.Series) -> float:
    vol = annualize_vol(monthly_returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return annualize_return(monthly_returns) / vol


def ols_fit(X: np.ndarray, y: np.ndarray):
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    residuals = y - X @ beta
    n, k = X.shape
    sigma2 = (residuals.T @ residuals) / (n - k)
    var_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    r2 = 1 - (residuals.T @ residuals) / np.sum((y - y.mean()) ** 2)
    return beta, se_beta, t_stats, r2, residuals


def load_strategy_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "strategy_return" not in df.columns:
        raise ValueError("strategy CSV에는 'date', 'strategy_return' 컬럼이 필요합니다.")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def load_factor_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = ["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"factor CSV에 필요한 컬럼이 없습니다: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def run_factor_regression(strategy_path: str, factor_path: str, output_prefix: str = "factor"):
    strategy = load_strategy_returns(strategy_path)
    factors = load_factor_returns(factor_path)

    df = pd.merge(strategy, factors, on="date", how="inner").dropna()
    if len(df) < 24:
        raise ValueError("회귀에 필요한 데이터가 너무 적습니다. 최소 24개 이상 권장합니다.")

    df["excess_return"] = df["strategy_return"] - df["RF"]
    factor_cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    X = df[factor_cols].to_numpy(dtype=float)
    y = df["excess_return"].to_numpy(dtype=float)
    X_with_const = np.column_stack([np.ones(len(X)), X])

    beta, se_beta, t_stats, r2, residuals = ols_fit(X_with_const, y)
    names = ["alpha"] + factor_cols
    result_df = pd.DataFrame({
        "term": names,
        "coef": beta,
        "std_err": se_beta,
        "t_stat": t_stats,
    })

    monthly_alpha = float(beta[0])
    annual_alpha = (1 + monthly_alpha) ** 12 - 1 if monthly_alpha > -1 else np.nan
    market_beta = float(beta[1])

    summary_df = pd.DataFrame([{
        "n_obs": len(df),
        "monthly_alpha": monthly_alpha,
        "annual_alpha": annual_alpha,
        "beta": market_beta,
        "r_squared": float(r2),
        "strategy_ann_return": annualize_return(df["strategy_return"]),
        "strategy_ann_vol": annualize_vol(df["strategy_return"]),
        "strategy_sharpe": sharpe_ratio(df["strategy_return"]),
    }])

    merged_path = f"{output_prefix}_merged_data.csv"
    coef_path = f"{output_prefix}_regression_results.csv"
    summary_path = f"{output_prefix}_summary.csv"

    df.to_csv(merged_path, index=False)
    result_df.to_csv(coef_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print("\n=== Coefficients ===")
    print(result_df.to_string(index=False))
    print(f"\nSaved:\n- {merged_path}\n- {coef_path}\n- {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True, help="CSV with date, strategy_return")
    parser.add_argument("--factors", type=str, required=True, help="CSV with Fama-French 5 factors + RF")
    parser.add_argument("--output-prefix", type=str, default="factor")
    args = parser.parse_args()

    run_factor_regression(
        strategy_path=args.strategy,
        factor_path=args.factors,
        output_prefix=args.output_prefix,
    )
