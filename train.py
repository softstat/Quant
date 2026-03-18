"""
train.py
--------
GAT Survival Model Training Pipeline (Standalone)
Usage:
    python train.py --mode sp500 --epochs 20 --retrain
    python train.py --mode kospi --epochs 10 --train-limit-dates 60
"""

import argparse
import logging
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch

from config import get_test_config, get_sp500_config, get_kospi_config
from data_pipeline import DataPipeline, FundamentalCollector
from earnings_collector import EarningsCollector, MarketContextCollector
from feature_assembler import FeatureAssembler
from graph_builder import build_full_graph
from gat_survival_model import GATSurvivalModel, GATSurvivalTrainer
from macro_collector import MacroCollector
from llama_engine import LLaMAClient, FeatureEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Temporal Split Utilities
# ─────────────────────────────────────────────

def make_temporal_splits(
    survival_labels: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = survival_labels.copy()
    labels["entry_date"] = pd.to_datetime(labels["entry_date"])
    labels = labels.sort_values("entry_date").reset_index(drop=True)

    unique_dates = sorted(labels["entry_date"].dt.normalize().unique())
    n_dates = len(unique_dates)
    if n_dates < 10:
        raise ValueError(f"Too few unique dates for temporal split: {n_dates}")

    train_end = max(1, int(n_dates * train_ratio))
    val_end   = max(train_end + 1, int(n_dates * (train_ratio + val_ratio)))
    val_end   = min(val_end, n_dates - 1)

    train_dates = set(unique_dates[:train_end])
    val_dates   = set(unique_dates[train_end:val_end])
    test_dates  = set(unique_dates[val_end:])

    train_df = labels[labels["entry_date"].dt.normalize().isin(train_dates)].copy()
    val_df   = labels[labels["entry_date"].dt.normalize().isin(val_dates)].copy()
    test_df  = labels[labels["entry_date"].dt.normalize().isin(test_dates)].copy()

    logger.info(f"Split sizes → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def build_training_snapshot_dates(split_df: pd.DataFrame) -> List[str]:
    return (
        pd.to_datetime(split_df["entry_date"])
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )


# ─────────────────────────────────────────────
# 2. Snapshot & Feature Assembly
# ─────────────────────────────────────────────

def build_context_features(cfg, tickers, data):
    earn_collector    = EarningsCollector(cfg)
    earnings_raw      = earn_collector.collect_earnings(tickers)
    earnings_features = earn_collector.build_earnings_features(earnings_raw, data["prices_with_ta"])

    market_collector = MarketContextCollector(cfg)
    market_raw       = market_collector.collect_indices(cfg.data.start_date)
    market_features  = market_collector.build_market_features(market_raw)

    macro_collector = MacroCollector(cfg)
    macro_raw       = macro_collector.collect_all(cfg.data.start_date)
    macro_features  = macro_collector.build_macro_features(macro_raw)

    return earnings_features, market_features, macro_features


def _reorder_snapshot_to_graph(
    snapshot_x: np.ndarray,
    snapshot_meta: pd.DataFrame,
    graph_tickers: List[str],
):
    meta    = snapshot_meta.copy().reset_index(drop=True)
    row_map = dict(zip(meta["ticker"], np.arange(len(meta))))

    available_tickers = [t for t in graph_tickers if t in row_map]
    if not available_tickers:
        raise ValueError("No overlap between snapshot_meta and graph_tickers.")

    order          = [row_map[t] for t in available_tickers]
    reordered_x    = snapshot_x[order]
    reordered_meta = meta.set_index("ticker").loc[available_tickers].reset_index()
    return reordered_x, reordered_meta


def assemble_snapshot_for_date(
    assembler: FeatureAssembler,       # ✅ 인스턴스를 외부에서 주입
    cfg, date_str, graph_tickers, data,
    earnings_f, market_f, macro_f, llama_embs,
):
    x_raw, _, _, meta = assembler.assemble_snapshot(
        date_str, graph_tickers,
        data["prices_with_ta"], data["fundamentals"],
        earnings_f, market_f, macro_f, llama_embs,
    )
    return _reorder_snapshot_to_graph(x_raw, meta, graph_tickers)


def _build_label_tensors_for_anchor_date(split_df: pd.DataFrame, meta: pd.DataFrame):
    df      = split_df.sort_values("entry_date").groupby("ticker").tail(1)
    aligned = meta[["ticker"]].merge(df, on="ticker", how="left")

    dur_col = next((c for c in ["duration", "durations", "time"]  if c in df.columns), None)
    evt_col = next((c for c in ["event_type", "event", "status"]  if c in df.columns), None)

    aligned["durations"]   = pd.to_numeric(aligned[dur_col], errors="coerce").fillna(1.0)
    aligned["event_types"] = pd.to_numeric(aligned[evt_col], errors="coerce").fillna(0).astype(int)
    aligned["events"]      = (aligned["event_types"] != 0).astype(float)

    return (
        torch.FloatTensor(aligned["durations"].values),
        torch.FloatTensor(aligned["events"].values),
        torch.LongTensor(aligned["event_types"].values),
    )


# ─────────────────────────────────────────────
# 3. Scaler Fitting (train only)
# ─────────────────────────────────────────────

def fit_scaler_on_train(
    assembler: FeatureAssembler,
    train_dates: List[str],
    graph_tickers: List[str],
    data, cfg, earn_f, mkt_f, mac_f, llama_embs,
    scaler_path: str,
):
    """
    train 날짜들의 raw feature를 쌓아서 scaler를 fit한 뒤 저장.
    val/test에서는 이 scaler로 transform만 수행 → 미래 정보 차단.
    """
    logger.info(f"Fitting scaler on {len(train_dates)} train snapshots...")
    all_features = []

    for date_str in train_dates:
        try:
            # scale=False로 raw feature만 수집 (assembler._fitted=False 상태여야 함)
            x_raw, _, _, meta = assembler.assemble_snapshot(
                date_str, graph_tickers,
                data["prices_with_ta"], data["fundamentals"],
                earn_f, mkt_f, mac_f, llama_embs,
            )
            if len(x_raw) > 0:
                all_features.append(x_raw)
        except Exception as e:
            logger.warning(f"Scaler fit: {date_str} skipped ({e})")

    if not all_features:
        raise RuntimeError("Scaler fit 실패 — train snapshot이 모두 비어있습니다")

    combined = np.vstack(all_features)
    assembler.fit_scaler_on_data(combined)
    assembler.save_scaler(scaler_path)
    logger.info(f"Scaler fitted on {combined.shape[0]} samples, {combined.shape[1]} features")


# ─────────────────────────────────────────────
# 4. Dataset & Training Helpers
# ─────────────────────────────────────────────

def build_single_dataset_for_split(
    assembler, split_df, graph_data, data, cfg,
    earn_f, mkt_f, mac_f, llama_embs,
):
    anchor_date = pd.to_datetime(split_df["entry_date"]).max().strftime("%Y-%m-%d")
    x_np, meta  = assemble_snapshot_for_date(
        assembler, cfg, anchor_date, graph_data["tickers"],
        data, earn_f, mkt_f, mac_f, llama_embs,
    )
    durations, events, event_types = _build_label_tensors_for_anchor_date(split_df, meta)

    n    = len(meta)
    mask = torch.ones(n, dtype=torch.bool)

    return {
        "x":           torch.FloatTensor(x_np),
        "edge_index":  torch.LongTensor(graph_data["edge_index"]),
        "durations":   durations,
        "events":      events,
        "event_types": event_types,
        "train_mask":  mask,
        "val_mask":    mask,
        "meta":        meta,
        "date":        anchor_date,
    }


def train_model(
    assembler, trainer, graph_data,
    train_labels, val_labels,
    data, cfg, earn_f, mkt_f, mac_f, llama_embs,
    epochs, checkpoint_path,
):
    logger.info("Caching training snapshots...")
    train_dates = build_training_snapshot_dates(train_labels)

    train_batches = [
        build_single_dataset_for_split(
            assembler,
            train_labels[
                pd.to_datetime(train_labels["entry_date"]).dt.strftime("%Y-%m-%d") == d
            ],
            graph_data, data, cfg, earn_f, mkt_f, mac_f, llama_embs,
        )
        for d in train_dates
    ]

    # Validation: 최신 snapshot 단일 배치
    val_batch = build_single_dataset_for_split(
        assembler, val_labels, graph_data, data, cfg,
        earn_f, mkt_f, mac_f, llama_embs,
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_batches:
            history    = trainer.fit(batch, val_data=None, epochs=1)
            epoch_loss += history[-1]["total"]

        val_history = trainer.fit(val_batch, val_data=None, epochs=1)
        logger.info(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"Train Loss: {epoch_loss/max(len(train_batches),1):.4f} | "
            f"Val Loss: {val_history[-1]['total']:.4f}"
        )

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(trainer.model.state_dict(), checkpoint_path)
    logger.info(f"✅ Checkpoint saved → {checkpoint_path}")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GAT Survival Model Trainer")
    parser.add_argument("--mode",              choices=["test", "sp500", "kospi"], default="sp500")
    parser.add_argument("--epochs",            type=int, default=10)
    parser.add_argument("--train-limit-dates", type=int, default=60,
                        help="최근 N개 날짜만 사용 (0=전체)")
    parser.add_argument("--retrain",           action="store_true",
                        help="기존 체크포인트 무시하고 재학습")
    parser.add_argument("--portfolio-size",    type=int, default=20,
                        help="포트폴리오 편입 종목 수 (default: 20)")
    parser.add_argument("--max-per-sector",    type=int, default=2,
                        help="섹터별 최대 편입 종목 수 (default: 2)")
    args = parser.parse_args()

    # ── Config ──────────────────────────────
    cfg_map = {"sp500": get_sp500_config, "kospi": get_kospi_config, "test": get_test_config}
    cfg     = cfg_map[args.mode]()

    # ── Data ────────────────────────────────
    pipeline = DataPipeline(cfg)
    data     = pipeline.run(steps=["universe", "prices", "fundamentals", "technical", "survival_labels"])
    tickers  = data["tickers"]

    # ── Features ────────────────────────────
    earn_f, mkt_f, mac_f = build_context_features(cfg, tickers, data)

    sector_map    = FundamentalCollector(cfg).get_sector_industry_map(data["fundamentals"])
    _, graph_data = build_full_graph(sector_map, data["prices_with_ta"], tickers, cfg)
    llama_embs    = None

    # ── Splits ──────────────────────────────
    train_labels, val_labels, test_labels = make_temporal_splits(data["survival_labels"])

    if args.train_limit_dates:
        recent_dates = build_training_snapshot_dates(train_labels)[-args.train_limit_dates:]
        train_labels = train_labels[
            pd.to_datetime(train_labels["entry_date"])
            .dt.strftime("%Y-%m-%d")
            .isin(recent_dates)
        ]
        logger.info(f"Train limited to last {args.train_limit_dates} dates ({len(train_labels)} rows)")

    # ── FeatureAssembler 단일 인스턴스 생성 ──
    #    scaler는 train 데이터로만 fit → val/test는 transform만 수행
    assembler   = FeatureAssembler(cfg)
    scaler_path = os.path.join(cfg.data.data_dir, "models", f"scaler_{args.mode}.joblib")

    if args.retrain or not os.path.exists(scaler_path):
        # train snapshot 전체로 scaler fit 후 저장
        train_dates_for_scaler = build_training_snapshot_dates(train_labels)
        fit_scaler_on_train(
            assembler, train_dates_for_scaler, graph_data["tickers"],
            data, cfg, earn_f, mkt_f, mac_f, llama_embs,
            scaler_path,
        )
    else:
        # 기존 scaler 로드 → val/test는 동일 scaler로 transform
        assembler.load_scaler(scaler_path)
        logger.info(f"Scaler loaded ← {scaler_path}")

    # ── Model ───────────────────────────────
    ref_date = build_training_snapshot_dates(train_labels)[-1]
    ref_x, _ = assemble_snapshot_for_date(
        assembler, cfg, ref_date, graph_data["tickers"],
        data, earn_f, mkt_f, mac_f, llama_embs,
    )
    model_cfg = {
        "feature_dim": ref_x.shape[1],
        "gat_hidden":  128,
        "gat_out":     64,
        "num_heads":   8,
        "num_layers":  3,
        "device":      cfg.device,
    }
    model   = GATSurvivalModel(model_cfg)
    trainer = GATSurvivalTrainer(model, model_cfg)

    checkpoint_path = os.path.join(
        cfg.data.data_dir, "models", f"gat_survival_{args.mode}.pt"
    )

    # ── Train / Load ────────────────────────
    if args.retrain or not os.path.exists(checkpoint_path):
        train_model(
            assembler, trainer, graph_data,
            train_labels, val_labels,
            data, cfg, earn_f, mkt_f, mac_f, llama_embs,
            args.epochs, checkpoint_path,
        )
    else:
        logger.info(f"체크포인트 존재 → 로드만 수행: {checkpoint_path}")

    trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    logger.info("✅ Training pipeline completed.")

    # ── Inference (test 구간, 동일 assembler 사용) ──
    test_date = build_training_snapshot_dates(test_labels)[-1]
    test_x, test_meta = assemble_snapshot_for_date(
        assembler, cfg, test_date, graph_data["tickers"],
        data, earn_f, mkt_f, mac_f, llama_embs,
    )
    preds = trainer.predict({
        "x":          torch.FloatTensor(test_x),
        "edge_index": torch.LongTensor(graph_data["edge_index"]),
    })

    # ── Portfolio Construction ───────────────
    from portfolio_construction import construct_portfolio
    from gat_survival_model import MultiMarketSignalRanker

    rankings = MultiMarketSignalRanker({}).rank(preds, graph_data["tickers"])

    # test_meta의 sector/market_cap 병합 (Unknown 방지)
    rankings = rankings.merge(
        test_meta[["ticker", "sector", "market_cap"]],
        on="ticker",
        how="left",
        suffixes=("", "_meta"),
    )
    if "sector_meta" in rankings.columns:
        mask = rankings["sector"].isna() | (rankings["sector"] == "Unknown")
        rankings.loc[mask, "sector"] = rankings.loc[mask, "sector_meta"]
        rankings.drop(columns=["sector_meta"], inplace=True)
    if "market_cap_meta" in rankings.columns:
        rankings.drop(columns=["market_cap_meta"], inplace=True)

    portfolio = construct_portfolio(
        rankings,
        sector_map=sector_map,
        total_n=args.portfolio_size,
        max_per_sector=args.max_per_sector,
        max_per_industry=2,
        min_score_col="score",
        weighting="equal",
    )

    portfolio_path = os.path.join(cfg.data.data_dir, f"portfolio_{args.mode}.parquet")
    portfolio.to_parquet(portfolio_path, index=False)
    logger.info(f"Portfolio saved → {portfolio_path}")

    # ── test_labels 저장 (backtest.py 입력용) ──
    test_label_path = os.path.join(cfg.data.data_dir, f"test_labels_{args.mode}.parquet")
    test_labels.to_parquet(test_label_path, index=False)
    logger.info(f"Test labels saved → {test_label_path}")
    logger.info(f"\n▶ 백테스트 실행:")
    logger.info(f"  python backtest.py --portfolio {portfolio_path} --start {test_date}")


if __name__ == "__main__":
    main()