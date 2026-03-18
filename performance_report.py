import argparse
import os
import pandas as pd

from quant_metrics import summarize_metrics, save_summary_json


def load_input(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


def performance_metrics(returns, benchmark_returns=None, turnover_series=None):
    return summarize_metrics(returns, benchmark_returns=benchmark_returns, turnover_series=turnover_series)


def main():
    parser = argparse.ArgumentParser(description="Full performance report for quant strategy")
    parser.add_argument("--input", required=True, help="CSV/Parquet with strategy returns")
    parser.add_argument("--return-col", default="strategy_return")
    parser.add_argument("--benchmark-col", default="benchmark_return")
    parser.add_argument("--turnover-col", default="turnover")
    parser.add_argument("--output-dir", default="performance_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_input(args.input)

    if args.return_col not in df.columns:
        raise ValueError(f"Missing return column: {args.return_col}")

    benchmark = df[args.benchmark_col] if args.benchmark_col in df.columns else None
    turnover = df[args.turnover_col] if args.turnover_col in df.columns else None

    metrics = summarize_metrics(df[args.return_col], benchmark_returns=benchmark, turnover_series=turnover)
    summary_df = pd.DataFrame([metrics])

    out_csv = os.path.join(args.output_dir, "performance_summary.csv")
    out_json = os.path.join(args.output_dir, "performance_summary.json")
    summary_df.to_csv(out_csv, index=False)
    save_summary_json(metrics, out_json)

    print(summary_df.to_string(index=False))
    print(f"\nSaved:\n- {out_csv}\n- {out_json}")


if __name__ == "__main__":
    main()
