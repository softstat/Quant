"""
Module: Feature Assembler

Merges all feature sources into a unified node feature matrix for GNN:
  - Technical indicators (from data_pipeline)
  - Fundamental ratios (from data_pipeline)
  - Earnings features (from earnings_collector)
  - Market context features (SPY/QQQ/KOSPI regime)
  - Macro features (from macro_collector)
  - LLaMA embeddings (from llama_engine)

Additionally returns per-snapshot metadata for portfolio selection:
  - ticker
  - market_cap
  - sector
  - ret_20d
  - vol_20d
  - drawdown_60d
  - macro_benefit_score
  - cap_bucket

Output:
  - GNN-ready node feature matrix
  - valid tickers
  - feature names
  - snapshot metadata for thematic portfolio selection
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


# ============================================================
# Base Feature Lists
# ============================================================

TA_FEATURES = [
    "return_1d", "return_5d", "return_20d", "log_return",
    "volume_ratio",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "sma_ratio_50", "sma_ratio_200",
    "rsi", "macd", "macd_signal", "macd_histogram",
    "bb_width", "bb_position",
    "atr_ratio",
    "stoch_k", "stoch_d",
    "momentum_10", "momentum_20",
    "volatility_20", "volatility_60",
    "position_52w",
]

FUNDAMENTAL_FEATURES = [
    "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
    "returnOnEquity", "returnOnAssets", "debtToEquity",
    "operatingMargins", "profitMargins",
    "revenueGrowth", "earningsGrowth",
    "beta", "dividendYield",
]

EARNINGS_FEATURES = [
    "days_since_earnings", "days_to_earnings", "in_earnings_window",
    "last_eps_surprise_pct", "earnings_momentum", "post_earnings_drift",
]

MARKET_CONTEXT_FEATURES = [
    "spy_ret_1d", "spy_ret_5d", "spy_ret_20d",
    "qqq_ret_1d", "qqq_ret_5d", "qqq_ret_20d",
    "kospi_ret_1d", "kospi_ret_5d", "kospi_ret_20d",
    "spy_sma20_ratio", "spy_sma50_ratio",
    "spy_qqq_spread_20d", "us_kr_spread_20d",
    "market_breadth", "vix_level", "vix_regime", "market_regime",
    "spy_volatility_20d",
]

MACRO_FEATURES = [
    "wti_ret_5d", "wti_ret_20d", "wti_sma50_ratio", "wti_vol_20d",
    "brent_ret_20d",
    "gold_ret_5d", "gold_ret_20d", "gold_sma50_ratio",
    "copper_ret_5d", "copper_ret_20d",
    "gold_copper_ratio_z",
    "yield_curve_10y_3m", "us10y_level", "us10y_change_5d", "us10y_change_20d",
    "credit_spread_z", "inflation_breakeven_proxy",
    "tlt_ret_20d", "tlt_sma50_ratio",
    "dxy_ret_5d", "dxy_ret_20d", "dxy_level",
    "usdkrw_ret_5d", "usdkrw_ret_20d", "usdkrw_level", "krw_regime",
    "dxy_gold_corr_20d",
    "pmi_proxy_ret_20d", "pmi_proxy_sma50",
    "consumer_proxy_ret_20d", "transport_proxy_ret_20d",
    "semicon_proxy_ret_20d",
    "housing_proxy_ret_20d", "china_proxy_ret_20d",
    "em_bond_ret_20d", "em_equity_ret_20d", "em_dm_spread",
    "risk_appetite_index", "global_liquidity_proxy", "inflation_pressure_index",
]

SELECTION_META_COLUMNS = [
    "ticker",
    "market_cap",
    "sector",
    "ret_20d",
    "vol_20d",
    "drawdown_60d",
    "macro_benefit_score",
    "cap_bucket",
]


class FeatureAssembler:
    """Assemble all feature sources into GNN-ready format + selection metadata."""

    def __init__(self, config=None):
        self.config  = config
        self.scaler  = RobustScaler()
        self._fitted = False  # ✅ train 데이터로 fit된 후에만 True

    # ========================================================
    # Scaler 관리 (train only fit)
    # ========================================================

    def fit_scaler_on_data(self, node_features: np.ndarray) -> None:
        """train snapshot 전체를 쌓은 뒤 한 번에 fit. 외부(train.py)에서만 호출."""
        if len(node_features) > 10:
            self.scaler.fit(node_features)
            self._fitted = True
            logger.info(f"Scaler fitted: {node_features.shape[0]} samples, {node_features.shape[1]} features")

    def save_scaler(self, path: str) -> None:
        """fit된 scaler를 joblib으로 저장."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved → {path}")

    def load_scaler(self, path: str) -> None:
        """저장된 scaler를 로드. val/test에서 transform만 수행하기 위해 사용."""
        self.scaler  = joblib.load(path)
        self._fitted = True
        logger.info(f"Scaler loaded ← {path}")

    # ========================================================
    # Public APIs
    # ========================================================

    def assemble_snapshot(
        self,
        date: str,
        tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        fundamentals_df: pd.DataFrame,
        earnings_features: Dict[str, pd.DataFrame],
        market_features: pd.DataFrame,
        macro_features: Optional[pd.DataFrame] = None,
        llama_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str], Dict, pd.DataFrame]:
        """Assemble features for all tickers at a specific date.

        Returns:
            node_features: [num_tickers, feature_dim]
            valid_tickers: tickers that had valid features
            feature_names: dict with names/dim
            meta_df: downstream portfolio selection metadata
        """
        dt = pd.Timestamp(date)
        all_features  = []
        valid_tickers = []
        feature_names = {}
        meta_rows     = []

        # Static fundamental lookup by ticker
        fund_dict = {}
        if fundamentals_df is not None and len(fundamentals_df) > 0:
            for _, row in fundamentals_df.iterrows():
                tkr = row.get("ticker")
                if pd.notna(tkr):
                    fund_dict[str(tkr)] = row

        # Shared market / macro vectors for this date
        mkt_vector   = self._get_market_vector(market_features, dt)
        macro_vector = self._get_macro_vector(macro_features, dt)

        for i, ticker in enumerate(tickers):
            try:
                feat_vector = []
                names_this  = []

                # 1. Technical features
                ta_feat = self._get_ta_features(price_data.get(ticker), dt)
                feat_vector.extend(ta_feat)
                names_this.extend([f"ta_{f}" for f in TA_FEATURES])

                # 2. Fundamental features
                fund_row  = fund_dict.get(ticker)
                fund_feat = self._get_fund_features(fund_row)
                feat_vector.extend(fund_feat)
                names_this.extend([f"fund_{f}" for f in FUNDAMENTAL_FEATURES])

                # 3. Earnings features
                earn_feat = self._get_earnings_features(earnings_features.get(ticker), dt)
                feat_vector.extend(earn_feat)
                names_this.extend([f"earn_{f}" for f in EARNINGS_FEATURES])

                # 4. Market context
                feat_vector.extend(mkt_vector)
                names_this.extend([f"mkt_{f}" for f in MARKET_CONTEXT_FEATURES])

                # 5. Macro context
                feat_vector.extend(macro_vector)
                names_this.extend([f"macro_{f}" for f in MACRO_FEATURES])

                # 6. LLaMA embeddings
                if llama_embeddings is not None and i < len(llama_embeddings):
                    emb = np.asarray(llama_embeddings[i], dtype=np.float32)
                    feat_vector.extend(emb.tolist())
                    names_this.extend([f"emb_{j}" for j in range(len(emb))])

                feat_array = np.array(feat_vector, dtype=np.float32)

                if len(feat_array) == 0:
                    continue

                # Accept if less than 50% missing
                nan_ratio = np.isnan(feat_array).sum() / len(feat_array)
                if nan_ratio < 0.5:
                    feat_array = np.nan_to_num(
                        feat_array, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    all_features.append(feat_array)
                    valid_tickers.append(ticker)

                    if not feature_names:
                        feature_names = {"names": names_this, "dim": len(names_this)}

                    meta_rows.append(
                        self._build_selection_meta(
                            ticker=ticker,
                            dt=dt,
                            price_df=price_data.get(ticker),
                            fund_row=fund_row,
                            macro_df=macro_features,
                        )
                    )

            except Exception as e:
                logger.debug(f"{ticker} feature assembly failed at {date}: {e}")
                continue

        if len(all_features) == 0:
            return (
                np.zeros((0, 0), dtype=np.float32),
                [],
                {},
                pd.DataFrame(columns=SELECTION_META_COLUMNS),
            )

        node_features = np.stack(all_features)

        # ✅ scaler: train에서 fit된 경우에만 transform (미래 정보 차단)
        if self._fitted:
            node_features = self.scaler.transform(node_features)

        meta_df = pd.DataFrame(meta_rows)
        meta_df = self._add_cap_bucket(meta_df)

        return node_features.astype(np.float32), valid_tickers, feature_names, meta_df

    def assemble_temporal(
        self,
        dates: List[str],
        tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        fundamentals_df: pd.DataFrame,
        earnings_features: Dict[str, pd.DataFrame],
        market_features: pd.DataFrame,
        macro_features: Optional[pd.DataFrame] = None,
        llama_embeddings: Optional[np.ndarray] = None,
    ) -> Dict:
        """Assemble features across multiple dates for temporal modeling."""
        logger.info(f"Assembling temporal features for {len(dates)} dates, {len(tickers)} tickers")
        snapshots   = {}
        feature_dim = None

        for date in dates:
            feat, valid, names, meta = self.assemble_snapshot(
                date=date,
                tickers=tickers,
                price_data=price_data,
                fundamentals_df=fundamentals_df,
                earnings_features=earnings_features,
                market_features=market_features,
                macro_features=macro_features,
                llama_embeddings=llama_embeddings,
            )

            if len(valid) > 0:
                snapshots[date] = {"features": feat, "tickers": valid, "meta": meta}
                if feature_dim is None:
                    feature_dim = feat.shape[1]

        logger.info(f"Assembled {len(snapshots)} snapshots, feature_dim={feature_dim}")
        return {"snapshots": snapshots, "feature_dim": feature_dim}

    def get_feature_dim(self, has_llama: bool = False, has_macro: bool = True) -> int:
        dim = (
            len(TA_FEATURES)
            + len(FUNDAMENTAL_FEATURES)
            + len(EARNINGS_FEATURES)
            + len(MARKET_CONTEXT_FEATURES)
        )
        if has_macro:
            dim += len(MACRO_FEATURES)
        if has_llama:
            emb_dim = getattr(getattr(self.config, "features", None), "embedding_dim", 0)
            dim += emb_dim
        return dim

    # ========================================================
    # Feature getters
    # ========================================================

    def _get_ta_features(self, df: Optional[pd.DataFrame], dt: pd.Timestamp) -> List[float]:
        if df is None or len(df) == 0:
            return [0.0] * len(TA_FEATURES)
        row = self._get_row_asof(df, dt)
        if row is None:
            return [0.0] * len(TA_FEATURES)
        return [self._safe_float(row.get(f, np.nan), 0.0) for f in TA_FEATURES]

    def _get_fund_features(self, row) -> List[float]:
        if row is None:
            return [0.0] * len(FUNDAMENTAL_FEATURES)
        return [self._safe_float(row.get(f, np.nan), 0.0) for f in FUNDAMENTAL_FEATURES]

    def _get_earnings_features(self, df: Optional[pd.DataFrame], dt: pd.Timestamp) -> List[float]:
        if df is None or len(df) == 0:
            return [0.0] * len(EARNINGS_FEATURES)
        row = self._get_row_asof(df, dt)
        if row is None:
            return [0.0] * len(EARNINGS_FEATURES)
        return [self._safe_float(row.get(f, np.nan), 0.0) for f in EARNINGS_FEATURES]

    def _get_market_vector(self, mkt_df: Optional[pd.DataFrame], dt: pd.Timestamp) -> List[float]:
        if mkt_df is None or len(mkt_df) == 0:
            return [0.0] * len(MARKET_CONTEXT_FEATURES)
        row = self._get_row_asof(mkt_df, dt)
        if row is None:
            return [0.0] * len(MARKET_CONTEXT_FEATURES)
        return [
            self._safe_float(row.get(f, np.nan) if f in row.index else np.nan, 0.0)
            for f in MARKET_CONTEXT_FEATURES
        ]

    def _get_macro_vector(self, macro_df: Optional[pd.DataFrame], dt: pd.Timestamp) -> List[float]:
        if macro_df is None or len(macro_df) == 0:
            return [0.0] * len(MACRO_FEATURES)
        row = self._get_row_asof(macro_df, dt)
        if row is None:
            return [0.0] * len(MACRO_FEATURES)
        return [
            self._safe_float(row.get(f, np.nan) if f in row.index else np.nan, 0.0)
            for f in MACRO_FEATURES
        ]

    # ========================================================
    # Selection meta builders
    # ========================================================

    def _build_selection_meta(
        self,
        ticker: str,
        dt: pd.Timestamp,
        price_df: Optional[pd.DataFrame],
        fund_row,
        macro_df: Optional[pd.DataFrame],
    ) -> Dict:
        ret_20d     = 0.0
        vol_20d     = 0.0
        drawdown_60d = 0.0

        if price_df is not None and len(price_df) > 0:
            px = self._get_row_asof(price_df, dt)
            if px is not None:
                ret_20d = self._safe_float(px.get("return_20d", 0.0), 0.0)
                if "volatility_20" in px.index:
                    vol_20d = self._safe_float(px.get("volatility_20", 0.0), 0.0)
                else:
                    vol_20d = self._compute_realized_vol(price_df, dt, lookback=20)
            drawdown_60d = self._compute_drawdown(price_df, dt, lookback=60)

        market_cap = 0.0
        sector     = "Unknown"

        if fund_row is not None:
            market_cap  = self._safe_float(fund_row.get("marketCap", 0.0), 0.0)
            sector_val  = fund_row.get("sector", "Unknown")
            sector      = "Unknown" if pd.isna(sector_val) else str(sector_val)

        macro_benefit_score = self._compute_macro_benefit_score(
            sector=sector, macro_df=macro_df, dt=dt,
        )

        return {
            "ticker":             ticker,
            "market_cap":         market_cap,
            "sector":             sector,
            "ret_20d":            ret_20d,
            "vol_20d":            vol_20d,
            "drawdown_60d":       drawdown_60d,
            "macro_benefit_score": macro_benefit_score,
        }

    def _compute_macro_benefit_score(
        self,
        sector: str,
        macro_df: Optional[pd.DataFrame],
        dt: pd.Timestamp,
    ) -> float:
        if macro_df is None or len(macro_df) == 0:
            base = 0.0
        else:
            row = self._get_row_asof(macro_df, dt)
            if row is None:
                base = 0.0
            else:
                risk_appetite = self._safe_float(row.get("risk_appetite_index", 0.0), 0.0)
                liquidity     = self._safe_float(row.get("global_liquidity_proxy", 0.0), 0.0)
                semicon       = self._safe_float(row.get("semicon_proxy_ret_20d", 0.0), 0.0)
                dxy           = self._safe_float(row.get("dxy_ret_20d", 0.0), 0.0)
                us10y         = self._safe_float(row.get("us10y_change_20d", 0.0), 0.0)

                base = (
                    0.35 * risk_appetite
                    + 0.20 * liquidity
                    + 0.20 * semicon
                    - 0.15 * dxy
                    - 0.10 * us10y
                )

        cyclical_bonus_sectors = {
            "Industrials", "Materials", "Energy", "Financials",
            "Consumer Discretionary", "Technology",
        }
        if sector in cyclical_bonus_sectors:
            base += 0.25

        return float(base)

    def _add_cap_bucket(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        if meta_df is None or len(meta_df) == 0:
            return pd.DataFrame(columns=SELECTION_META_COLUMNS)

        out = meta_df.copy()
        cap = pd.to_numeric(out["market_cap"], errors="coerce")

        if cap.notna().sum() >= 3:
            try:
                bucket = pd.qcut(
                    cap, q=3,
                    labels=["small", "mid", "large"],
                    duplicates="drop",
                )
                out["cap_bucket"] = bucket.astype(str)
                out.loc[~out["cap_bucket"].isin(["small", "mid", "large"]), "cap_bucket"] = "mid"
            except Exception:
                out["cap_bucket"] = "mid"
        else:
            out["cap_bucket"] = "mid"

        for col in SELECTION_META_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        return out[SELECTION_META_COLUMNS]

    # ========================================================
    # Time-series helpers
    # ========================================================

    def _get_row_asof(self, df: Optional[pd.DataFrame], dt: pd.Timestamp):
        if df is None or len(df) == 0:
            return None

        tmp = df.copy()
        tmp.index = pd.to_datetime(tmp.index)

        if dt in tmp.index:
            row = tmp.loc[dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            return row

        mask = tmp.index <= dt
        if mask.any():
            row = tmp.loc[tmp.index[mask][-1]]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            return row

        return None

    def _compute_realized_vol(
        self,
        price_df: Optional[pd.DataFrame],
        dt: pd.Timestamp,
        lookback: int = 20,
    ) -> float:
        if price_df is None or len(price_df) == 0:
            return 0.0

        tmp = price_df.copy()
        tmp.index = pd.to_datetime(tmp.index)
        tmp = tmp.loc[tmp.index <= dt].sort_index().tail(lookback + 1)

        if len(tmp) < 5:
            return 0.0

        if "return_1d" in tmp.columns:
            rets = pd.to_numeric(tmp["return_1d"], errors="coerce").dropna()
        elif "close" in tmp.columns:
            close = pd.to_numeric(tmp["close"], errors="coerce").dropna()
            rets  = close.pct_change().dropna()
        else:
            return 0.0

        if len(rets) < 5:
            return 0.0

        return float(rets.std() * np.sqrt(252))

    def _compute_drawdown(
        self,
        price_df: Optional[pd.DataFrame],
        dt: pd.Timestamp,
        lookback: int = 60,
    ) -> float:
        if price_df is None or len(price_df) == 0:
            return 0.0

        tmp = price_df.copy()
        tmp.index = pd.to_datetime(tmp.index)
        tmp = tmp.loc[tmp.index <= dt].sort_index().tail(lookback)

        if len(tmp) == 0 or "close" not in tmp.columns:
            return 0.0

        close = pd.to_numeric(tmp["close"], errors="coerce").dropna()
        if len(close) == 0:
            return 0.0

        cur  = close.iloc[-1]
        peak = close.max()

        if peak <= 0:
            return 0.0

        return float(cur / peak - 1.0)

    # ========================================================
    # Generic helpers
    # ========================================================

    def _safe_float(self, x, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            if pd.isna(x):
                return default
            return float(x)
        except Exception:
            return default