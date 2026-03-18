import logging
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============================================================
# Major Market Indices & Sector ETFs
# ============================================================
MARKET_INDICES = {
    "SPY":   {"name": "S&P 500",      "market": "US", "type": "broad_index"},
    "QQQ":   {"name": "NASDAQ 100",   "market": "US", "type": "broad_index"},
    "DIA":   {"name": "Dow Jones",    "market": "US", "type": "broad_index"},
    "IWM":   {"name": "Russell 2000", "market": "US", "type": "broad_index"},
    "^VIX":  {"name": "Volatility",   "market": "US", "type": "volatility"},
    "^KS11": {"name": "KOSPI",         "market": "KR", "type": "broad_index"},
    "^KQ11": {"name": "KOSDAQ",        "market": "KR", "type": "broad_index"},
    "EWY":   {"name": "Korea ETF",     "market": "KR", "type": "country_etf"},
}

SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
    "XLE": "Energy", "XLI": "Industrials", "XLC": "Communication Services",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLB": "Materials", "XLRE": "Real Estate", "XLU": "Utilities",
}

class EarningsCollector:
    def __init__(self, config):
        self.config = config
        self.save_dir = config.data.parquet_dir
        # AttributeError 방지: 세션을 사용하지 않거나, 사용할 거면 생성 후 헤더 설정
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def get_sp500_tickers(self) -> List[str]:
        """Wikipedia에서 S&P 500 리스트 수집"""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                # StringIO 에러 방지를 위해 직접 데이터프레임 변환
                df = pd.read_html(response.text)[0]
                return df['Symbol'].replace(r'\.', '-', regex=True).tolist()
        except Exception as e:
            logger.error(f"Wikipedia 로드 실패: {e}")
            
        # 실패 시 백업 100개 리스트
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
            "UNH", "MA", "JNJ", "WMT", "PG", "XOM", "COST", "HD", "ABBV", "MRK",
            "ORCL", "CVX", "BAC", "KO", "AVGO", "PEP", "ADBE", "TMO", "CSCO", "CRM",
            "PFE", "ABT", "DHR", "NFLX", "AMD", "WFC", "ACN", "DIS", "PM", "INTC",
            "CAT", "VZ", "INTU", "TXN", "AMGN", "LOW", "NEE", "HON", "COP", "RTX",
            "IBM", "UNP", "GE", "AMAT", "GS", "SPGI", "QCOM", "PLTR", "SBUX", "BLK",
            "NOW", "AXP", "MS", "ELV", "SYK", "MDLZ", "ISRG", "TJX", "DE", "LMT",
            "GILD", "BA", "ADI", "C", "T", "MU", "VRTX", "LRCX", "REGN", "ETN",
            "BSX", "CB", "PANW", "MMC", "WM", "CI", "ICE", "PGR", "FI", "MDT",
            "SNPS", "CDNS", "SLB", "EOG", "MO", "CVS", "ZTS", "BDX", "ITW", "KLAC"
        ]

    def collect_earnings(self, tickers: List[str]) -> pd.DataFrame:
        logger.info(f"Collecting earnings data for {len(tickers)} tickers...")
        all_earnings = []
        for ticker in tqdm(tickers, desc="Earnings"):
            try:
                # 세션 문제 해결: 최신 yfinance 버전 이슈 대응을 위해 세션 주입 제외 시도
                t = yf.Ticker(ticker) 
                hist = t.earnings_history
                if hist is not None and not hist.empty:
                    for idx, row in hist.iterrows():
                        all_earnings.append({
                            "ticker": ticker,
                            "date": pd.to_datetime(row.get("date", idx)),
                            "eps_surprise_pct": row.get("surprisePercent", 0.0)
                        })
                time.sleep(0.1)
            except:
                continue
        df = pd.DataFrame(all_earnings)
        if not df.empty:
            df.to_parquet(f"{self.save_dir}/earnings_data.parquet", index=False)
        return df

    def build_earnings_features(self, earnings_df, price_data):
        result = {}
        MAX_GAP = 180 
        if earnings_df is None or earnings_df.empty or "ticker" not in earnings_df.columns:
            for ticker in price_data.keys():
                result[ticker] = self._empty_earnings_features(price_data[ticker], MAX_GAP)
            return result

        for ticker, df in price_data.items():
            ticker_earn = earnings_df[earnings_df["ticker"] == ticker].sort_values("date")
            if ticker_earn.empty:
                result[ticker] = self._empty_earnings_features(df, MAX_GAP)
                continue
            
            earn_dates = pd.to_datetime(ticker_earn["date"]).values
            surprises = ticker_earn["eps_surprise_pct"].fillna(0.0).values
            feats = []
            for date in df.index:
                dt = pd.Timestamp(date)
                past = earn_dates[earn_dates <= dt.asm8]
                future = earn_dates[earn_dates > dt.asm8]
                days_since = min((dt - pd.Timestamp(past[-1])).days, MAX_GAP) if len(past) > 0 else MAX_GAP
                days_to = min((pd.Timestamp(future[0]) - dt).days, MAX_GAP) if len(future) > 0 else MAX_GAP
                feats.append({
                    "days_since_earnings": days_since,
                    "days_to_earnings": days_to,
                    "last_eps_surprise_pct": float(surprises[len(past)-1]) if len(past) > 0 else 0.0
                })
            result[ticker] = pd.DataFrame(feats, index=df.index)
        return result

    def _empty_earnings_features(self, df, max_gap):
        return pd.DataFrame({
            "days_since_earnings": max_gap, "days_to_earnings": max_gap, "last_eps_surprise_pct": 0.0
        }, index=df.index)

class MarketContextCollector:
    def __init__(self, config):
        self.config = config

    def collect_indices(self, start_date: str, end_date: str = None) -> Dict[str, pd.DataFrame]:
        all_tickers = list(MARKET_INDICES.keys()) + list(SECTOR_ETFS.keys())
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Collecting {len(all_tickers)} market context tickers...")
        data = {}
        try:
            # yfinance 최신 이슈 대응: session 인자 제거하여 기본값으로 시도
            raw = yf.download(all_tickers, start=start_date, end=end_date, 
                              auto_adjust=True, group_by="ticker")
            for ticker in all_tickers:
                try:
                    df = raw[ticker].dropna(how="all")
                    if len(df) > 20:
                        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                        data[ticker] = df
                except:
                    continue
        except Exception as e:
            logger.error(f"Batch download failed: {e}")
        return data

    def build_market_features(self, index_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        features = {}
        if "SPY" in index_data:
            spy_close = index_data["SPY"]["close"]
            features["spy_ret_20d"] = spy_close.pct_change(20)
            features["market_regime"] = (spy_close > spy_close.rolling(200).mean()).astype(int)
        
        if "^VIX" in index_data:
            features["vix_level"] = index_data["^VIX"]["close"]
            
        result = pd.DataFrame(features).ffill().dropna()
        logger.info(f"Market features built: {result.shape}")
        return result

    def align_to_ticker(self, market_features, ticker_df):
        common = ticker_df.index.intersection(market_features.index)
        return market_features.reindex(common)