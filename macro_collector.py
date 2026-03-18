"""
Module: macro_collector.py

매크로 경제지표 수집기
- yfinance 기반
- 외부 API 키 불필요
- 원자재 / 금리 / 채권 / FX / 매크로 프록시 / 글로벌 리스크 수집
- feature 생성 및 parquet 저장
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ============================================================
# 매크로 티커 정의
# ============================================================

MACRO_TICKERS = {
    # ── 원자재 (Commodities) ──
    "CL=F":     {"name": "WTI Crude Oil", "category": "commodity", "sub": "energy"},
    "BZ=F":     {"name": "Brent Crude Oil", "category": "commodity", "sub": "energy"},
    "NG=F":     {"name": "Natural Gas", "category": "commodity", "sub": "energy"},
    "GC=F":     {"name": "Gold", "category": "commodity", "sub": "precious"},
    "SI=F":     {"name": "Silver", "category": "commodity", "sub": "precious"},
    "HG=F":     {"name": "Copper", "category": "commodity", "sub": "industrial"},
    "ZW=F":     {"name": "Wheat", "category": "commodity", "sub": "agriculture"},
    "ZC=F":     {"name": "Corn", "category": "commodity", "sub": "agriculture"},

    # ── 금리 & 채권 (Rates & Bonds) ──
    "^TNX":     {"name": "US 10Y Treasury Yield", "category": "rates", "sub": "us_treasury"},
    "^IRX":     {"name": "US 13W T-Bill Yield", "category": "rates", "sub": "us_treasury"},
    "^FVX":     {"name": "US 5Y Treasury Yield", "category": "rates", "sub": "us_treasury"},
    "^TYX":     {"name": "US 30Y Treasury Yield", "category": "rates", "sub": "us_treasury"},
    "TLT":      {"name": "20+Y Treasury Bond ETF", "category": "rates", "sub": "bond_etf"},
    "SHY":      {"name": "1-3Y Treasury Bond ETF", "category": "rates", "sub": "bond_etf"},
    "IEF":      {"name": "7-10Y Treasury Bond ETF", "category": "rates", "sub": "bond_etf"},
    "LQD":      {"name": "IG Corporate Bond ETF", "category": "rates", "sub": "credit"},
    "HYG":      {"name": "HY Corporate Bond ETF", "category": "rates", "sub": "credit"},
    "TIP":      {"name": "TIPS ETF (Inflation)", "category": "rates", "sub": "inflation"},

    # ── Fed / 채권 변동성 proxy ──
    "ZQ=F":     {"name": "30D Fed Funds Futures", "category": "rates", "sub": "fedfunds"},
    "^MOVE":    {"name": "MOVE Index", "category": "risk", "sub": "rates_vol"},
    "^VIX":     {"name": "VIX Index", "category": "risk", "sub": "equity_vol"},

    # ── 통화 (FX) ──
    "DX-Y.NYB": {"name": "US Dollar Index (DXY)", "category": "fx", "sub": "dollar"},
    "KRW=X":    {"name": "USD/KRW", "category": "fx", "sub": "krw"},
    "JPY=X":    {"name": "USD/JPY", "category": "fx", "sub": "jpy"},
    "EURUSD=X": {"name": "EUR/USD", "category": "fx", "sub": "eur"},
    "CNY=X":    {"name": "USD/CNY", "category": "fx", "sub": "cny"},

    # ── 매크로 Proxy ETFs ──
    "XLI":      {"name": "Industrials ETF (PMI proxy)", "category": "macro_proxy", "sub": "pmi"},
    "XHB":      {"name": "Homebuilders ETF (Housing proxy)", "category": "macro_proxy", "sub": "housing"},
    "XRT":      {"name": "Retail ETF (Consumer proxy)", "category": "macro_proxy", "sub": "consumer"},
    "IYT":      {"name": "Transport ETF (Economy proxy)", "category": "macro_proxy", "sub": "transport"},
    "SMH":      {"name": "Semiconductor ETF", "category": "macro_proxy", "sub": "semicon"},
    "KWEB":     {"name": "China Internet ETF", "category": "macro_proxy", "sub": "china"},

    # ── 글로벌 리스크 ──
    "EMB":      {"name": "EM Bond ETF (EM Risk)", "category": "risk", "sub": "em"},
    "EEM":      {"name": "EM Equity ETF", "category": "risk", "sub": "em_equity"},
    "FXI":      {"name": "China Large Cap ETF", "category": "risk", "sub": "china"},
    "VXX":      {"name": "VIX Short-Term Futures ETN", "category": "risk", "sub": "vol"},
}


# ============================================================
# 피처 목록
# ============================================================

MACRO_FEATURE_LIST = [
    # 원자재
    "wti_ret_1d", "wti_ret_5d", "wti_ret_20d", "wti_ret_60d",
    "wti_sma20_ratio", "wti_sma50_ratio", "wti_vol_20d",
    "brent_ret_1d", "brent_ret_5d", "brent_ret_20d", "brent_ret_60d",
    "brent_sma20_ratio", "brent_sma50_ratio", "brent_vol_20d",
    "natgas_ret_1d", "natgas_ret_5d", "natgas_ret_20d", "natgas_ret_60d",
    "natgas_sma20_ratio", "natgas_sma50_ratio", "natgas_vol_20d",
    "gold_ret_1d", "gold_ret_5d", "gold_ret_20d", "gold_ret_60d",
    "gold_sma20_ratio", "gold_sma50_ratio", "gold_vol_20d",
    "silver_ret_1d", "silver_ret_5d", "silver_ret_20d", "silver_ret_60d",
    "silver_sma20_ratio", "silver_sma50_ratio", "silver_vol_20d",
    "copper_ret_1d", "copper_ret_5d", "copper_ret_20d", "copper_ret_60d",
    "copper_sma20_ratio", "copper_sma50_ratio", "copper_vol_20d",
    "brent_wti_spread",
    "gold_copper_ratio", "gold_copper_ratio_z",

    # 금리 & 채권
    "yield_curve_10y_3m", "yield_curve_10y_5y",
    "us10y_level", "us10y_change_1d", "us10y_change_5d", "us10y_change_20d",
    "us30y_level",
    "credit_spread_proxy", "credit_spread_z",
    "inflation_breakeven_proxy",
    "tlt_ret_5d", "tlt_ret_20d", "tlt_sma50_ratio",
    "fedfunds_proxy_level", "fedfunds_proxy_change_20d",
    "move_level", "move_z_60d",
    "vix_level", "vix_z_60d",

    # FX
    "dxy_ret_1d", "dxy_ret_5d", "dxy_ret_20d", "dxy_level", "dxy_vol_20d",
    "usdkrw_ret_1d", "usdkrw_ret_5d", "usdkrw_ret_20d", "usdkrw_level", "usdkrw_vol_20d",
    "usdjpy_ret_1d", "usdjpy_ret_5d", "usdjpy_ret_20d", "usdjpy_level", "usdjpy_vol_20d",
    "eurusd_ret_1d", "eurusd_ret_5d", "eurusd_ret_20d", "eurusd_level", "eurusd_vol_20d",
    "usdcny_ret_1d", "usdcny_ret_5d", "usdcny_ret_20d", "usdcny_level", "usdcny_vol_20d",
    "krw_regime",
    "dxy_gold_corr_20d",

    # 매크로 Proxy
    "pmi_proxy_ret_20d", "pmi_proxy_sma50",
    "consumer_proxy_ret_20d",
    "transport_proxy_ret_20d", "transport_proxy_sma50",
    "semicon_proxy_ret_20d", "semicon_proxy_vol",
    "housing_proxy_ret_20d",
    "china_proxy_ret_20d",

    # 글로벌 리스크
    "em_bond_ret_20d", "em_equity_ret_20d", "em_dm_spread",
    "vxx_ret_5d", "vxx_ret_20d",

    # 복합 지표
    "risk_appetite_index",
    "global_liquidity_proxy",
    "inflation_pressure_index",

    # 메타
    "data_timestamp",
]


class MacroCollector:
    """매크로 경제지표 수집 및 feature 생성기"""

    def __init__(self, config=None):
        self.config = config
        self.save_dir = getattr(getattr(config, "data", None), "parquet_dir", "data/parquet")
        os.makedirs(self.save_dir, exist_ok=True)

    # ========================================================
    # 내부 유틸
    # ========================================================

    def _normalize_download(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """yfinance 다운로드 결과 컬럼 정리"""
        if df is None or df.empty:
            logger.warning(f"{ticker}: empty dataframe before normalization")
            return pd.DataFrame()

        out = df.copy()

        # MultiIndex 컬럼 처리
        if isinstance(out.columns, pd.MultiIndex):
            flat_cols = []
            for col in out.columns:
                if isinstance(col, tuple):
                    flat_cols.append(str(col[0]))
                else:
                    flat_cols.append(str(col))
            out.columns = flat_cols

        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

        # close 우선순위 확보
        if "adj_close" in out.columns:
            price_col = "adj_close"
        elif "close" in out.columns:
            price_col = "close"
        else:
            logger.warning(f"{ticker}: no close column found. cols={list(out.columns)}")
            return pd.DataFrame()

        keep_cols = [price_col]
        if "open" in out.columns:
            keep_cols.append("open")
        if "high" in out.columns:
            keep_cols.append("high")
        if "low" in out.columns:
            keep_cols.append("low")
        if "volume" in out.columns:
            keep_cols.append("volume")

        out = out[keep_cols].copy()
        out = out.rename(columns={price_col: "close"})

        if "volume" not in out.columns:
            out["volume"] = np.nan

        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()].copy()
        out.index.name = "date"
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]

        # close 기준으로만 제거
        out = out[out["close"].notna()].copy()

        if out.empty:
            logger.warning(f"{ticker}: empty dataframe after normalization")
            return pd.DataFrame()

        logger.debug(f"{ticker}: normalized shape={out.shape}, cols={list(out.columns)}")
        return out

    @staticmethod
    def _safe_zscore(series: pd.Series, window: int = 60) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / (std + 1e-10)

    @staticmethod
    def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
        return series.pct_change(periods=periods, fill_method=None)

    # ========================================================
    # 1. 데이터 수집
    # ========================================================

    def collect_all(self, start_date: str, end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """전체 매크로 데이터 다운로드"""
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        tickers = list(MACRO_TICKERS.keys())

        logger.info(f"Collecting {len(tickers)} macro tickers from {start_date} to {end_date}")
        data: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="column",
                    interval="1d",
                )

                if raw is None or raw.empty:
                    logger.warning(f"{ticker}: download returned empty dataframe")
                    continue

                df = self._normalize_download(raw, ticker)

                if not df.empty:
                    data[ticker] = df
                    logger.info(f"{ticker}: collected {len(df)} rows")
                else:
                    logger.warning(f"{ticker}: empty dataframe after normalization")

            except Exception as e:
                logger.warning(f"{ticker} 수집 실패: {e}")

        logger.info(f"Collected {len(data)}/{len(tickers)} tickers successfully")
        return data

    # ========================================================
    # 2. Feature 생성
    # ========================================================

    def build_macro_features(self, macro_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """매크로 원천 데이터에서 feature 생성"""
        features: Dict[str, pd.Series] = {}

        # ─────────────────────────────────────
        # 1. 원자재
        # ─────────────────────────────────────
        for ticker, label in [
            ("CL=F", "wti"),
            ("BZ=F", "brent"),
            ("NG=F", "natgas"),
            ("GC=F", "gold"),
            ("SI=F", "silver"),
            ("HG=F", "copper"),
        ]:
            if ticker not in macro_data:
                continue

            close = macro_data[ticker]["close"]

            for period in [1, 5, 20, 60]:
                features[f"{label}_ret_{period}d"] = self._safe_pct_change(close, period)

            features[f"{label}_sma20_ratio"] = close / close.rolling(20).mean()
            features[f"{label}_sma50_ratio"] = close / close.rolling(50).mean()
            features[f"{label}_vol_20d"] = self._safe_pct_change(close, 1).rolling(20).std() * np.sqrt(252)

        if "CL=F" in macro_data and "BZ=F" in macro_data:
            wti = macro_data["CL=F"]["close"]
            brent = macro_data["BZ=F"]["close"]
            common = wti.index.intersection(brent.index)
            features["brent_wti_spread"] = brent.reindex(common) - wti.reindex(common)

        if "GC=F" in macro_data and "HG=F" in macro_data:
            gold = macro_data["GC=F"]["close"]
            copper = macro_data["HG=F"]["close"]
            common = gold.index.intersection(copper.index)
            ratio = gold.reindex(common) / copper.reindex(common)
            features["gold_copper_ratio"] = ratio
            features["gold_copper_ratio_z"] = self._safe_zscore(ratio, 60)

        # ─────────────────────────────────────
        # 2. 금리 & 채권
        # ─────────────────────────────────────
        if "^TNX" in macro_data:
            y10 = macro_data["^TNX"]["close"]
            features["us10y_level"] = y10
            features["us10y_change_1d"] = y10.diff(1)
            features["us10y_change_5d"] = y10.diff(5)
            features["us10y_change_20d"] = y10.diff(20)

        if "^TYX" in macro_data:
            features["us30y_level"] = macro_data["^TYX"]["close"]

        if "^TNX" in macro_data and "^IRX" in macro_data:
            y10 = macro_data["^TNX"]["close"]
            y3m = macro_data["^IRX"]["close"]
            common = y10.index.intersection(y3m.index)
            features["yield_curve_10y_3m"] = y10.reindex(common) - y3m.reindex(common)

        if "^TNX" in macro_data and "^FVX" in macro_data:
            y10 = macro_data["^TNX"]["close"]
            y5 = macro_data["^FVX"]["close"]
            common = y10.index.intersection(y5.index)
            features["yield_curve_10y_5y"] = y10.reindex(common) - y5.reindex(common)

        if "HYG" in macro_data and "LQD" in macro_data:
            hyg = macro_data["HYG"]["close"]
            lqd = macro_data["LQD"]["close"]
            common = hyg.index.intersection(lqd.index)
            spread_proxy = (
                self._safe_pct_change(lqd.reindex(common), 20) -
                self._safe_pct_change(hyg.reindex(common), 20)
            )
            features["credit_spread_proxy"] = spread_proxy
            features["credit_spread_z"] = self._safe_zscore(spread_proxy, 60)

        if "TIP" in macro_data and "IEF" in macro_data:
            tip = macro_data["TIP"]["close"]
            ief = macro_data["IEF"]["close"]
            common = tip.index.intersection(ief.index)
            features["inflation_breakeven_proxy"] = (
                self._safe_pct_change(tip.reindex(common), 20) -
                self._safe_pct_change(ief.reindex(common), 20)
            )

        if "TLT" in macro_data:
            tlt = macro_data["TLT"]["close"]
            features["tlt_ret_5d"] = self._safe_pct_change(tlt, 5)
            features["tlt_ret_20d"] = self._safe_pct_change(tlt, 20)
            features["tlt_sma50_ratio"] = tlt / tlt.rolling(50).mean()

        if "ZQ=F" in macro_data:
            zq = macro_data["ZQ=F"]["close"]
            features["fedfunds_proxy_level"] = zq
            features["fedfunds_proxy_change_20d"] = zq.diff(20)

        if "^MOVE" in macro_data:
            move = macro_data["^MOVE"]["close"]
            features["move_level"] = move
            features["move_z_60d"] = self._safe_zscore(move, 60)

        if "^VIX" in macro_data:
            vix = macro_data["^VIX"]["close"]
            features["vix_level"] = vix
            features["vix_z_60d"] = self._safe_zscore(vix, 60)

        # ─────────────────────────────────────
        # 3. 통화 (FX)
        # ─────────────────────────────────────
        for ticker, label in [
            ("DX-Y.NYB", "dxy"),
            ("KRW=X", "usdkrw"),
            ("JPY=X", "usdjpy"),
            ("EURUSD=X", "eurusd"),
            ("CNY=X", "usdcny"),
        ]:
            if ticker not in macro_data:
                continue

            close = macro_data[ticker]["close"]
            for period in [1, 5, 20]:
                features[f"{label}_ret_{period}d"] = self._safe_pct_change(close, period)
            features[f"{label}_level"] = close
            features[f"{label}_vol_20d"] = self._safe_pct_change(close, 1).rolling(20).std() * np.sqrt(252)

        if "KRW=X" in macro_data:
            krw = macro_data["KRW=X"]["close"]
            sma60 = krw.rolling(60).mean()
            features["krw_regime"] = pd.Series(
                np.where(krw > sma60 * 1.02, 2,
                np.where(krw < sma60 * 0.98, 0, 1)),
                index=krw.index,
                dtype=float,
            )

        if "DX-Y.NYB" in macro_data and "GC=F" in macro_data:
            dxy_ret = self._safe_pct_change(macro_data["DX-Y.NYB"]["close"], 1)
            gold_ret = self._safe_pct_change(macro_data["GC=F"]["close"], 1)
            common = dxy_ret.index.intersection(gold_ret.index)
            features["dxy_gold_corr_20d"] = dxy_ret.reindex(common).rolling(20).corr(gold_ret.reindex(common))

        # ─────────────────────────────────────
        # 4. 매크로 Proxy
        # ─────────────────────────────────────
        if "XLI" in macro_data:
            xli = macro_data["XLI"]["close"]
            features["pmi_proxy_ret_20d"] = self._safe_pct_change(xli, 20)
            features["pmi_proxy_sma50"] = xli / xli.rolling(50).mean()

        if "XRT" in macro_data:
            xrt = macro_data["XRT"]["close"]
            features["consumer_proxy_ret_20d"] = self._safe_pct_change(xrt, 20)

        if "IYT" in macro_data:
            iyt = macro_data["IYT"]["close"]
            features["transport_proxy_ret_20d"] = self._safe_pct_change(iyt, 20)
            features["transport_proxy_sma50"] = iyt / iyt.rolling(50).mean()

        if "SMH" in macro_data:
            smh = macro_data["SMH"]["close"]
            features["semicon_proxy_ret_20d"] = self._safe_pct_change(smh, 20)
            features["semicon_proxy_vol"] = self._safe_pct_change(smh, 1).rolling(20).std() * np.sqrt(252)

        if "XHB" in macro_data:
            xhb = macro_data["XHB"]["close"]
            features["housing_proxy_ret_20d"] = self._safe_pct_change(xhb, 20)

        if "KWEB" in macro_data:
            kweb = macro_data["KWEB"]["close"]
            features["china_proxy_ret_20d"] = self._safe_pct_change(kweb, 20)

        # ─────────────────────────────────────
        # 5. 글로벌 리스크
        # ─────────────────────────────────────
        if "EMB" in macro_data:
            emb = macro_data["EMB"]["close"]
            features["em_bond_ret_20d"] = self._safe_pct_change(emb, 20)

        if "EEM" in macro_data:
            eem = macro_data["EEM"]["close"]
            features["em_equity_ret_20d"] = self._safe_pct_change(eem, 20)

        if "EEM" in macro_data and "XLI" in macro_data:
            eem_ret = self._safe_pct_change(macro_data["EEM"]["close"], 20)
            dm_ret = self._safe_pct_change(macro_data["XLI"]["close"], 20)
            common = eem_ret.index.intersection(dm_ret.index)
            features["em_dm_spread"] = eem_ret.reindex(common) - dm_ret.reindex(common)

        if "VXX" in macro_data:
            vxx = macro_data["VXX"]["close"]
            features["vxx_ret_5d"] = self._safe_pct_change(vxx, 5)
            features["vxx_ret_20d"] = self._safe_pct_change(vxx, 20)

        # ─────────────────────────────────────
        # 6. 복합 지표
        # ─────────────────────────────────────
        risk_signals = []
        if "gold_copper_ratio_z" in features:
            risk_signals.append(-features["gold_copper_ratio_z"])
        if "tlt_ret_20d" in features:
            risk_signals.append(-features["tlt_ret_20d"])
        if "credit_spread_z" in features:
            risk_signals.append(-features["credit_spread_z"])
        if "move_z_60d" in features:
            risk_signals.append(-features["move_z_60d"])
        if "vix_z_60d" in features:
            risk_signals.append(-features["vix_z_60d"])

        if risk_signals:
            risk_df = pd.concat(risk_signals, axis=1)
            features["risk_appetite_index"] = risk_df.mean(axis=1)

        liquidity_signals = []
        if "tlt_ret_20d" in features:
            liquidity_signals.append(features["tlt_ret_20d"])
        if "gold_ret_20d" in features:
            liquidity_signals.append(features["gold_ret_20d"])
        if "dxy_ret_20d" in features:
            liquidity_signals.append(-features["dxy_ret_20d"])

        if liquidity_signals:
            liq_df = pd.concat(liquidity_signals, axis=1)
            features["global_liquidity_proxy"] = liq_df.mean(axis=1)

        inflation_signals = []
        if "wti_ret_20d" in features:
            inflation_signals.append(features["wti_ret_20d"])
        if "copper_ret_20d" in features:
            inflation_signals.append(features["copper_ret_20d"])
        if "inflation_breakeven_proxy" in features:
            inflation_signals.append(features["inflation_breakeven_proxy"])

        if inflation_signals:
            inf_df = pd.concat(inflation_signals, axis=1)
            features["inflation_pressure_index"] = inf_df.mean(axis=1)

        # ─────────────────────────────────────
        # 결과 결합 및 저장
        # ─────────────────────────────────────
        if not features:
            result = pd.DataFrame()
        else:
            result = pd.DataFrame(features).sort_index()
            result = result.replace([np.inf, -np.inf], np.nan)
            result = result.sort_index()
            result["data_timestamp"] = pd.Timestamp.now()

        save_path = os.path.join(self.save_dir, "macro_features.parquet")
        result.to_parquet(save_path)

        logger.info(f"Macro features saved: {save_path}, shape={result.shape}")
        print(f"✅ 매크로 데이터 업데이트 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return result

    # ========================================================
    # 3. 증분 업데이트
    # ========================================================

    def sync_macro_data(self, default_start_date: str = "2024-01-01") -> pd.DataFrame:
        """기존 파일 기준으로 누락 구간만 추가 수집"""
        file_path = os.path.join(self.save_dir, "macro_features.parquet")

        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            existing_df.index = pd.to_datetime(existing_df.index)
            last_date = existing_df.index.max()
            start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"Existing macro file found. Incremental update from {start_date}")
        else:
            existing_df = None
            start_date = default_start_date
            logger.info(f"No existing macro file. Full build from {start_date}")

        new_raw = self.collect_all(start_date=start_date)

        if not new_raw:
            logger.warning("No new raw macro data collected")
            return existing_df if existing_df is not None else pd.DataFrame()

        new_features = self.build_macro_features(new_raw)

        if existing_df is None or existing_df.empty:
            combined = new_features
        else:
            combined = pd.concat([existing_df, new_features], axis=0)
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        combined.to_parquet(file_path)
        logger.info(f"Synced macro features saved to {file_path}, shape={combined.shape}")
        return combined

    # ========================================================
    # 4. 보조 함수
    # ========================================================

    def get_feature_names(self) -> List[str]:
        """매크로 피처 전체 목록 반환"""
        return list(MACRO_FEATURE_LIST)

    def get_summary(self, macro_features: pd.DataFrame) -> Dict:
        """최신 매크로 상황 요약"""
        if macro_features is None or macro_features.empty:
            return {}

        latest = macro_features.iloc[-1]
        summary = {}

        for col in [
            "wti_ret_20d",
            "brent_ret_20d",
            "us10y_level",
            "yield_curve_10y_3m",
            "us10y_change_20d",
            "usdkrw_level",
            "dxy_level",
            "krw_regime",
            "risk_appetite_index",
            "global_liquidity_proxy",
            "inflation_pressure_index",
            "credit_spread_z",
            "gold_copper_ratio_z",
            "move_level",
            "vix_level",
        ]:
            if col in latest.index and pd.notna(latest[col]):
                summary[col] = float(latest[col])

        return summary