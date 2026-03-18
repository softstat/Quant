"""
backtest.py
-----------
Professional Backtester for GAT Survival Portfolio (Standalone)
Usage:
    python backtest.py --portfolio results/portfolio.parquet --start 2025-01-01
    python backtest.py --portfolio results/portfolio.parquet --benchmark QQQ --batch-size 50
"""

import argparse
import logging
import os
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────

DARK_STYLE = {
    "figure.facecolor": "#131722",
    "axes.facecolor":   "#131722",
    "text.color":       "#d1d4dc",
    "axes.labelcolor":  "#d1d4dc",
    "xtick.color":      "#d1d4dc",
    "ytick.color":      "#d1d4dc",
    "grid.color":       "#2a2e39",
}


# ─────────────────────────────────────────────
# Core Class
# ─────────────────────────────────────────────

class ProfessionalBacktester:
    """
    Parameters
    ----------
    portfolio_path : str
        parquet 파일 경로. 컬럼: ['ticker', 'weight']
    start_date : str
        백테스트 시작일 (YYYY-MM-DD)
    benchmark : str
        벤치마크 티커 (default: SPY)
    batch_size : int
        yfinance 배치 크기. Rate limit 방지용 (default: 100)
    sleep_sec : float
        배치 간 대기 시간(초) (default: 0.5)
    """

    def __init__(
        self,
        portfolio_path: str,
        start_date: str = "2025-01-01",
        benchmark: str  = "SPY",
        batch_size: int = 100,
        sleep_sec: float = 0.5,
    ):
        self.portfolio_df = pd.read_parquet(portfolio_path)
        self._validate_portfolio()

        self.start_date = start_date
        self.benchmark  = benchmark
        self.batch_size = batch_size
        self.sleep_sec  = sleep_sec
        self.tickers    = self.portfolio_df["ticker"].tolist()

        plt.rcParams.update(DARK_STYLE)

    # ── Validation ──────────────────────────
    def _validate_portfolio(self):
        required = {"ticker", "weight"}
        missing  = required - set(self.portfolio_df.columns)
        if missing:
            raise ValueError(f"portfolio 파일에 필수 컬럼 없음: {missing}")
        if self.portfolio_df["weight"].isna().any():
            raise ValueError("weight 컬럼에 NaN 존재")
        if not np.isclose(self.portfolio_df["weight"].sum(), 1.0, atol=1e-3):
            logger.warning("weight 합이 1이 아님 → 자동 재정규화")
            self.portfolio_df["weight"] /= self.portfolio_df["weight"].sum()

    # ── Download ────────────────────────────
    def _download_prices(self, tickers: list) -> pd.DataFrame:
        """배치 분할 + rate limit 대응 다운로드"""
        batches    = [tickers[i:i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]
        all_frames = []

        for idx, batch in enumerate(batches):
            try:
                raw = yf.download(
                    batch,
                    start=self.start_date,
                    auto_adjust=True,   # Adj Close → Close 통합
                    progress=False,
                    threads=True,
                )
                # yfinance 버전별 MultiIndex 처리
                if isinstance(raw.columns, pd.MultiIndex):
                    price = raw["Close"]
                else:
                    # 단일 티커 케이스
                    price = raw[["Close"]].rename(columns={"Close": batch[0]})

                all_frames.append(price)
                logger.info(f"Batch {idx+1}/{len(batches)} downloaded ({len(batch)} tickers)")

            except Exception as e:
                logger.warning(f"Batch {idx+1} failed: {e}")

            time.sleep(self.sleep_sec)

        if not all_frames:
            raise RuntimeError("모든 배치 다운로드 실패 — 네트워크/티커 목록 확인 필요")

        combined = pd.concat(all_frames, axis=1)
        # 중복 컬럼 제거 (배치 경계에서 동일 티커 중복 방지)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined

    # ── Main Run ────────────────────────────
    def run(self, output_dir: str = "./results") -> dict:
        logger.info(f"백테스트 시작: {len(self.tickers)}개 티커 / 시작일: {self.start_date}")

        # 가격 수집
        price_data = self._download_prices(self.tickers)
        bench_data = yf.download(
            self.benchmark, start=self.start_date,
            auto_adjust=True, progress=False,
        )["Close"]
        bench_data.name = self.benchmark

        # 실제 수집된 티커만 사용
        available_tickers = [t for t in self.tickers if t in price_data.columns]
        failed_tickers    = sorted(set(self.tickers) - set(available_tickers))

        if failed_tickers:
            logger.warning(f"{len(failed_tickers)}개 티커 수집 실패 → 제외: {failed_tickers}")

        if not available_tickers:
            raise RuntimeError("사용 가능한 티커가 없습니다.")

        # 가중치 재정규화 (실패 티커 제거 반영)
        weights = self.portfolio_df.set_index("ticker")["weight"]
        weights = weights.loc[available_tickers]
        weights = weights / weights.sum()

        # 수익률 계산
        port_returns  = price_data[available_tickers].pct_change().dropna()
        bench_returns = bench_data.pct_change().dropna()

        # 날짜 정렬
        common_idx    = port_returns.index.intersection(bench_returns.index)
        port_returns  = port_returns.loc[common_idx]
        bench_returns = bench_returns.loc[common_idx]

        # 가중 포트폴리오 수익률
        weighted_returns = (port_returns * weights).sum(axis=1)

        # 누적 수익률
        cum_port  = (1 + weighted_returns).cumprod()
        cum_bench = (1 + bench_returns).cumprod()

        # MDD
        peak      = cum_port.cummax()
        drawdown  = (cum_port - peak) / peak
        mdd       = float(drawdown.min())

        # 추가 통계
        stats = self._compute_stats(weighted_returns, bench_returns, cum_port, mdd,
                                     available_tickers, failed_tickers)

        # 시각화
        self._plot_report(cum_port, cum_bench, drawdown, mdd, stats, output_dir)

        logger.info(f"\n{'─'*45}")
        logger.info(f"  Total Return  : {stats['total_return']:>8.2%}")
        logger.info(f"  Ann. Return   : {stats['ann_return']:>8.2%}")
        logger.info(f"  Ann. Vol      : {stats['ann_vol']:>8.2%}")
        logger.info(f"  Sharpe        : {stats['sharpe']:>8.2f}")
        logger.info(f"  MDD           : {stats['mdd']:>8.2%}")
        logger.info(f"  Available     : {stats['available_tickers']} / {len(self.tickers)}")
        logger.info(f"{'─'*45}")

        return stats

    # ── Statistics ──────────────────────────
    def _compute_stats(self, port_ret, bench_ret, cum_port, mdd,
                        available_tickers, failed_tickers) -> dict:
        n_days     = len(port_ret)
        ann_factor = 252

        ann_return = (1 + port_ret.mean()) ** ann_factor - 1
        ann_vol    = port_ret.std() * np.sqrt(ann_factor)
        sharpe     = ann_return / (ann_vol + 1e-12)

        # 벤치마크 대비 Alpha (간단 계산)
        bench_ann  = (1 + bench_ret.mean()) ** ann_factor - 1
        alpha      = ann_return - bench_ann

        return {
            "total_return":      float(cum_port.iloc[-1] - 1),
            "ann_return":        float(ann_return),
            "ann_vol":           float(ann_vol),
            "sharpe":            float(sharpe),
            "mdd":               float(mdd),
            "alpha_vs_bench":    float(alpha),
            "available_tickers": len(available_tickers),
            "failed_tickers":    failed_tickers,
        }

    # ── Visualization ───────────────────────
    def _plot_report(self, cum_port, cum_bench, drawdown, mdd, stats, output_dir):
        fig = plt.figure(figsize=(15, 10))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

        # ── (1) Equity Curve ──
        ax0 = plt.subplot(gs[0])
        ax0.plot(cum_port,  color="#2962ff", linewidth=2.5, label="GNN Survival Strategy")
        ax0.plot(cum_bench, color="#ff9800", linewidth=1.5, linestyle="--",
                 alpha=0.7, label=f"Benchmark ({self.benchmark})")
        ax0.fill_between(cum_port.index, cum_port, 1, color="#2962ff", alpha=0.1)

        # 주요 지표 텍스트 박스
        info = (
            f"Total Return: {stats['total_return']:+.2%}  |  "
            f"Ann. Return: {stats['ann_return']:+.2%}  |  "
            f"Sharpe: {stats['sharpe']:.2f}  |  "
            f"Alpha: {stats['alpha_vs_bench']:+.2%}"
        )
        ax0.set_title(
            f"GNN Survival Strategy  —  Performance Report\n{info}",
            fontsize=13, fontweight="bold", pad=15,
        )
        ax0.legend(frameon=False, loc="upper left", fontsize=11)
        ax0.set_ylabel("Cumulative Return")
        ax0.grid(True, linestyle=":", alpha=0.5)
        ax0.tick_params(labelbottom=False)

        # ── (2) Drawdown ──
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.fill_between(drawdown.index, drawdown, 0, color="#ef5350", alpha=0.3)
        ax1.plot(drawdown, color="#ef5350", linewidth=1.0)
        ax1.axhline(mdd, color="#ef5350", linestyle="--", linewidth=0.8,
                    label=f"MDD {mdd:.2%}")
        ax1.axhline(0,   color="white",   linewidth=0.5, alpha=0.8)
        ax1.set_ylabel("Drawdown")
        ax1.set_ylim(mdd * 1.3, 0.05)
        ax1.legend(frameon=False, loc="lower left", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.5)

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "backtest_report.png")
        plt.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Chart saved → {save_path}")


# ─────────────────────────────────────────────
# Callable wrapper (train.py / main.py에서 import 가능)
# ─────────────────────────────────────────────

def run_backtest(
    portfolio_path: str,
    output_dir: str   = "./results",
    start_date: str   = "2025-01-01",
    benchmark: str    = "SPY",
    batch_size: int   = 100,
) -> dict:
    tester = ProfessionalBacktester(portfolio_path, start_date, benchmark, batch_size)
    stats  = tester.run(output_dir)
    return {"portfolio_stats": stats}


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GNN Survival Portfolio Backtester")
    parser.add_argument("--portfolio",   required=True,          help="portfolio.parquet 경로")
    parser.add_argument("--output-dir",  default="./results",    help="결과 저장 폴더")
    parser.add_argument("--start",       default="2025-01-01",   help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--benchmark",   default="SPY",          help="벤치마크 티커")
    parser.add_argument("--batch-size",  type=int, default=100,  help="yfinance 배치 크기")
    args = parser.parse_args()

    run_backtest(
        portfolio_path=args.portfolio,
        output_dir=args.output_dir,
        start_date=args.start,
        benchmark=args.benchmark,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()