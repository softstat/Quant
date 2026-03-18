"""
Quant Survival × GNN × LLaMA
Module 1: Data Collector
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import PipelineConfig, DataConfig

logger = logging.getLogger(__name__)


# ============================================================
# 1. Universe Definition
# ============================================================

class UniverseManager:
    def __init__(self, config: DataConfig):
        self.config = config

    def get_tickers(self) -> List[str]:
        if self.config.universe == "SP500":
            return self._get_sp500_tickers()
        elif self.config.universe == "KOSPI200":
            return self._get_kospi200_tickers()
        elif self.config.universe == "custom":
            return self.config.custom_tickers
        else:
            raise ValueError(f"Unknown universe: {self.config.universe}")

    def _get_sp500_tickers(self) -> List[str]:
        """
        Wikipedia 403 대응:
          1) requests + User-Agent 헤더로 직접 fetch → pd.read_html(StringIO(...))
          2) 실패 시 fallback 리스트 반환
        """
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            df = tables[0]
            tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers
        except Exception as e:
            logger.warning(f"Wikipedia fetch failed ({e}) → using fallback list")
            return self._get_fallback_sp500()

    def _get_kospi200_tickers(self) -> List[str]:
        major_kospi = [
            "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS",
            "006400.KS", "051910.KS", "035420.KS", "000270.KS", "068270.KS",
            "035720.KS", "105560.KS", "055550.KS", "003670.KS", "012330.KS",
            "066570.KS", "096770.KS", "034730.KS", "028260.KS", "003550.KS",
        ]
        logger.info(f"Loaded {len(major_kospi)} KOSPI tickers")
        return major_kospi

    def _get_fallback_sp500(self) -> List[str]:
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH",
            "MA", "HD", "COST", "ABBV", "MRK", "CRM", "PEP", "AVGO",
            "KO", "LLY", "TMO", "ACN", "CSCO", "MCD", "ABT", "ADBE",
            "DHR", "CMCSA", "NKE", "TXN", "NEE", "PM", "VZ", "INTC",
            "RTX", "HON", "LOW", "UPS", "QCOM", "BA", "CAT", "GS",
            "AMGN", "IBM", "SBUX",
        ]


# ============================================================
# 2. Price Data Collector
# ============================================================

class PriceCollector:
    def __init__(self, config: DataConfig):
        self.config = config
        self.save_dir = os.path.join(config.parquet_dir, "prices")
        os.makedirs(self.save_dir, exist_ok=True)

    def collect_all(
        self,
        tickers: List[str],
        batch_size: int = 100,
        sleep_sec: float = 0.5,
    ) -> Dict[str, pd.DataFrame]:
        """
        yfinance multi.py crash 대응:
          - threads=False  → 내부 스레드 충돌 방지
          - 배치 분할      → rate limit / 메모리 압박 완화
          - MultiIndex     → 버전별 안전 처리
        """
        logger.info(f"Collecting price data for {len(tickers)} tickers...")
        end_date = self.config.end_date or datetime.now().strftime("%Y-%m-%d")

        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        all_raw: List[pd.DataFrame] = []

        for idx, batch in enumerate(batches):
            try:
                raw = yf.download(
                    batch,
                    start=self.config.start_date,
                    end=end_date,
                    interval=self.config.price_interval,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=False,      # ✅ 스레드 충돌 방지
                    progress=False,
                )
                all_raw.append((batch, raw))
                logger.info(f"  Batch {idx+1}/{len(batches)} OK ({len(batch)} tickers)")
            except Exception as e:
                logger.warning(f"  Batch {idx+1} failed: {e} → individual fallback")
                fallback = self._download_individual(batch, end_date)
                # individual 결과를 dict로 바로 result에 추가
                all_raw.append(("__individual__", fallback))

            time.sleep(sleep_sec)

        # ── 결과 파싱 ──────────────────────────────────────
        result: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        for batch_item in all_raw:
            key, data = batch_item

            # individual fallback은 이미 dict
            if key == "__individual__":
                for ticker, df in data.items():
                    processed = self._process_df(ticker, df)
                    if processed is not None:
                        result[ticker] = processed
                    else:
                        failed.append(ticker)
                continue

            batch = key  # list of tickers
            for ticker in batch:
                try:
                    # ✅ yfinance 버전별 MultiIndex 안전 처리
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker not in data.columns.get_level_values(0):
                            failed.append(ticker)
                            continue
                        df = data[ticker].copy()
                    else:
                        # 단일 티커 배치
                        df = data.copy()

                    processed = self._process_df(ticker, df)
                    if processed is not None:
                        result[ticker] = processed
                    else:
                        failed.append(ticker)

                except Exception as e:
                    logger.warning(f"{ticker}: {e}")
                    failed.append(ticker)

        logger.info(f"Collected {len(result)} tickers, {len(failed)} failed")
        if failed:
            logger.info(f"Failed tickers (first 10): {failed[:10]}")
        return result

    def _process_df(self, ticker: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """공통 후처리 — 컬럼 정규화, 수익률 추가, 저장"""
        try:
            df = df.dropna(how="all")
            if len(df) < 30:
                logger.warning(f"{ticker}: insufficient data ({len(df)} rows)")
                return None

            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df.index.name = "date"

            df["return_1d"]  = df["close"].pct_change(1)
            df["return_5d"]  = df["close"].pct_change(5)
            df["return_20d"] = df["close"].pct_change(20)
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))

            df["volume_ma20"]  = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma20"]

            filepath = os.path.join(
                self.save_dir, f"{ticker.replace('.', '_')}.parquet"
            )
            df.to_parquet(filepath)
            return df

        except Exception as e:
            logger.warning(f"{ticker} processing failed: {e}")
            return None

    def _download_individual(self, tickers: List[str], end_date: str) -> Dict[str, pd.DataFrame]:
        """개별 다운로드 fallback"""
        result = {}
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=self.config.start_date,
                    end=end_date,
                    interval=self.config.price_interval,
                    auto_adjust=True,
                    threads=False,
                    progress=False,
                )
                if len(df) >= 30:
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    df.index.name = "date"
                    result[ticker] = df
                time.sleep(0.3)
            except Exception:
                continue
        return result


# ============================================================
# 3. Fundamental Data Collector
# ============================================================

class FundamentalCollector:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.save_dir = os.path.join(config.data.parquet_dir, "fundamentals")
        os.makedirs(self.save_dir, exist_ok=True)

    def collect_all(self, tickers: List[str]) -> pd.DataFrame:
        logger.info(f"Collecting fundamentals for {len(tickers)} tickers...")
        records = []
        for ticker in tqdm(tickers, desc="Fundamentals"):
            record = self._collect_single(ticker)
            if record:
                records.append(record)
            time.sleep(0.3)

        df = pd.DataFrame(records)
        filepath = os.path.join(self.save_dir, "fundamentals.parquet")
        df.to_parquet(filepath, index=False)
        logger.info(f"Collected fundamentals for {len(df)} tickers")
        return df

    def _collect_single(self, ticker: str) -> Optional[Dict]:
        try:
            t = yf.Ticker(ticker)
            info = t.info

            record = {
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", np.nan),
                "country": info.get("country", "Unknown"),
                "exchange": info.get("exchange", "Unknown"),
                "currency": info.get("currency", "USD"),
            }

            for field in self.config.features.fundamental_fields:
                record[field] = info.get(field, np.nan)

            try:
                bs = t.balance_sheet
                if bs is not None and len(bs) > 0:
                    latest = bs.iloc[:, 0]
                    record["total_assets"] = latest.get("Total Assets", np.nan)
                    record["total_debt"] = latest.get("Total Debt", np.nan)
                    record["total_equity"] = latest.get("Stockholders Equity", np.nan)
            except Exception:
                pass

            try:
                rec = t.recommendations
                if rec is not None and len(rec) > 0:
                    latest_rec = rec.iloc[-1]
                    record["analyst_rating"] = latest_rec.get("To Grade", "")
                    record["analyst_firm"] = latest_rec.get("Firm", "")
            except Exception:
                pass

            return record

        except Exception as e:
            logger.warning(f"{ticker} fundamentals failed: {e}")
            return None

    def get_sector_industry_map(self, fundamentals_df: pd.DataFrame) -> Dict:
        sector_map = {}
        industry_map = {}

        for _, row in fundamentals_df.iterrows():
            ticker = row["ticker"]
            sector_map[ticker] = row.get("sector", "Unknown")
            industry_map[ticker] = row.get("industry", "Unknown")

        return {
            "sector_map": sector_map,
            "industry_map": industry_map,
            "sectors": fundamentals_df.groupby("sector")["ticker"].apply(list).to_dict(),
            "industries": fundamentals_df.groupby("industry")["ticker"].apply(list).to_dict(),
        }


# ============================================================
# 4. Investing.com Supplementary Scraper
# ============================================================

class InvestingComScraper:
    def __init__(self, config: DataConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        self.save_dir = os.path.join(config.parquet_dir, "investing_com")
        os.makedirs(self.save_dir, exist_ok=True)

    def scrape_economic_calendar(self, start_date=None, end_date=None, importance=3) -> pd.DataFrame:
        logger.info("Scraping economic calendar...")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        events = []
        try:
            url = f"{self.config.investing_com_base}/economic-calendar/"
            response = self._safe_request(url)
            if response is None:
                return pd.DataFrame()

            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.find_all("tr", class_="js-event-item")

            for row in rows:
                try:
                    event = self._parse_calendar_row(row)
                    if event and event.get("importance", 0) >= importance:
                        events.append(event)
                except Exception:
                    continue

            df = pd.DataFrame(events)
            if len(df) > 0:
                filepath = os.path.join(self.save_dir, "economic_calendar.parquet")
                df.to_parquet(filepath, index=False)

            logger.info(f"Scraped {len(df)} economic events")
            return df

        except Exception as e:
            logger.error(f"Economic calendar scraping failed: {e}")
            return pd.DataFrame()

    def _parse_calendar_row(self, row) -> Optional[Dict]:
        try:
            event = {}
            event["datetime"] = row.get("data-event-datetime", "")

            flag = row.find("td", class_="flagCur")
            event["country"] = flag.text.strip() if flag else ""

            name_cell = row.find("td", class_="event")
            event["event_name"] = name_cell.text.strip() if name_cell else ""

            importance = row.find("td", class_="sentiment")
            if importance:
                bulls = importance.find_all("i", class_="grayFullBullishIcon")
                event["importance"] = len(bulls)
            else:
                event["importance"] = 0

            for field in ["actual", "forecast", "previous"]:
                cell = row.find("td", class_=field)
                if cell:
                    val = cell.text.strip().replace("%", "").replace(",", "")
                    try:
                        event[field] = float(val)
                    except (ValueError, TypeError):
                        event[field] = np.nan
                else:
                    event[field] = np.nan

            if not np.isnan(event.get("actual", np.nan)) and not np.isnan(event.get("forecast", np.nan)):
                event["surprise"] = event["actual"] - event["forecast"]
            else:
                event["surprise"] = np.nan

            return event
        except Exception:
            return None

    def _safe_request(self, url: str) -> Optional[requests.Response]:
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(self.config.request_delay)
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    return response
                elif response.status_code == 403:
                    logger.warning("403 Forbidden")
                    time.sleep(5 * (attempt + 1))
                elif response.status_code == 429:
                    logger.warning("Rate limited")
                    time.sleep(30 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error (attempt {attempt+1}): {e}")
                time.sleep(2 * (attempt + 1))
        return None


# ============================================================
# 5. News Collector
# ============================================================

class NewsCollector:
    def __init__(self, config: DataConfig):
        self.config = config
        self.save_dir = os.path.join(config.parquet_dir, "news")
        os.makedirs(self.save_dir, exist_ok=True)

    def collect_yfinance_news(self, tickers: List[str]) -> pd.DataFrame:
        logger.info(f"Collecting news for {len(tickers)} tickers...")
        all_news = []

        for ticker in tqdm(tickers, desc="News collection"):
            try:
                t = yf.Ticker(ticker)
                news = t.news
                if news:
                    for article in news:
                        all_news.append({
                            "ticker": ticker,
                            "title": article.get("title", ""),
                            "publisher": article.get("publisher", ""),
                            "link": article.get("link", ""),
                            "published_at": datetime.fromtimestamp(
                                article.get("providerPublishTime", 0)
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                            "type": article.get("type", ""),
                        })
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"News for {ticker} failed: {e}")

        df = pd.DataFrame(all_news)
        if len(df) > 0:
            df["published_at"] = pd.to_datetime(df["published_at"])
            df = df.sort_values("published_at", ascending=False)
            df.to_parquet(os.path.join(self.save_dir, "yfinance_news.parquet"), index=False)

        logger.info(f"Collected {len(df)} news articles")
        return df

    def collect_rss_feeds(self, feeds: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        import feedparser
        if feeds is None:
            feeds = {
                "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
                "reuters_markets": "https://feeds.reuters.com/reuters/marketsNews",
                "yahoo_finance": "https://finance.yahoo.com/rss/",
                "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            }
        all_articles = []
        for source_name, feed_url in feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    all_articles.append({
                        "source": source_name,
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "link": entry.get("link", ""),
                        "published_at": entry.get("published", ""),
                    })
            except Exception as e:
                logger.warning(f"RSS feed {source_name} failed: {e}")

        df = pd.DataFrame(all_articles)
        if len(df) > 0:
            df.to_parquet(os.path.join(self.save_dir, "rss_news.parquet"), index=False)
        logger.info(f"Collected {len(df)} RSS articles")
        return df


# ============================================================
# 6. Technical Indicators Calculator
# ============================================================

class TechnicalIndicators:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def calculate_all(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        logger.info("Calculating technical indicators...")
        result = {}
        for ticker, df in tqdm(price_data.items(), desc="Technical indicators"):
            try:
                result[ticker] = self._add_indicators(df.copy())
            except Exception as e:
                logger.warning(f"{ticker} TA calculation failed: {e}")
                result[ticker] = df
        return result

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        for w in self.config.features.ta_windows:
            df[f"sma_{w}"] = close.rolling(w).mean()
            df[f"ema_{w}"] = close.ewm(span=w).mean()
            df[f"sma_ratio_{w}"] = close / df[f"sma_{w}"]

        period = self.config.features.rsi_period
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        fast, slow, signal = (
            self.config.features.macd_fast,
            self.config.features.macd_slow,
            self.config.features.macd_signal,
        )
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        bb_period = self.config.features.bb_period
        bb_std = self.config.features.bb_std
        bb_mid = close.rolling(bb_period).mean()
        bb_std_val = close.rolling(bb_period).std()
        df["bb_upper"] = bb_mid + bb_std * bb_std_val
        df["bb_lower"] = bb_mid - bb_std * bb_std_val
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr_14"] / close

        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_ma20"] = obv.rolling(20).mean()

        df["momentum_10"] = close / close.shift(10) - 1
        df["momentum_20"] = close / close.shift(20) - 1

        df["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
        df["volatility_60"] = close.pct_change().rolling(60).std() * np.sqrt(252)

        df["high_52w"] = high.rolling(252).max()
        df["low_52w"] = low.rolling(252).min()
        df["position_52w"] = (close - df["low_52w"]) / (df["high_52w"] - df["low_52w"])

        return df


# ============================================================
# 7. Survival Event Labeler
# ============================================================

class SurvivalEventLabeler:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.target = config.survival.target_return
        self.stop = config.survival.stop_loss
        self.max_days = config.survival.max_holding_days

    def label_all(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("Creating survival event labels...")
        all_records = []
        for ticker, df in tqdm(price_data.items(), desc="Survival labeling"):
            records = self._label_single(ticker, df)
            all_records.extend(records)

        result = pd.DataFrame(all_records)

        if len(result) > 0:
            save_path = os.path.join(self.config.data.parquet_dir, "survival_labels.parquet")
            result.to_parquet(save_path, index=False)

            n_total    = len(result)
            n_profit   = (result["event_type"] == 1).sum()
            n_loss     = (result["event_type"] == 2).sum()
            n_censored = (result["event_type"] == 0).sum()
            logger.info(
                f"Survival labels: {n_total} total | "
                f"Profit: {n_profit} ({n_profit/n_total*100:.1f}%) | "
                f"Loss: {n_loss} ({n_loss/n_total*100:.1f}%) | "
                f"Censored: {n_censored} ({n_censored/n_total*100:.1f}%)"
            )

        return result

    def _label_single(self, ticker: str, df: pd.DataFrame) -> List[Dict]:
        records = []
        close = df["close"].values
        dates = df.index

        for i in range(len(close) - self.max_days):
            entry_price = close[i]
            entry_date  = dates[i]
            duration    = self.max_days
            event_type  = 0
            exit_price  = close[min(i + self.max_days, len(close) - 1)]

            for j in range(1, self.max_days + 1):
                if i + j >= len(close):
                    break
                current_return = (close[i + j] - entry_price) / entry_price
                if current_return >= self.target:
                    duration, event_type, exit_price = j, 1, close[i + j]
                    break
                if current_return <= self.stop:
                    duration, event_type, exit_price = j, 2, close[i + j]
                    break

            records.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "duration": duration,
                "event_type": event_type,
                "realized_return": (exit_price - entry_price) / entry_price,
            })

        return records


# ============================================================
# 8. Main Pipeline Orchestrator
# ============================================================

class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.universe_mgr         = UniverseManager(config.data)
        self.price_collector      = PriceCollector(config.data)
        self.fundamental_collector = FundamentalCollector(config)
        self.investing_scraper    = InvestingComScraper(config.data)
        self.news_collector       = NewsCollector(config.data)
        self.ta_calculator        = TechnicalIndicators(config)
        self.survival_labeler     = SurvivalEventLabeler(config)

    def run(self, steps: Optional[List[str]] = None) -> Dict:
        if steps is None:
            steps = [
                "universe", "prices", "fundamentals",
                "news", "technical", "calendar", "survival_labels",
            ]

        result = {}

        if "universe" in steps:
            logger.info("=" * 60)
            logger.info("STEP 1: Loading universe")
            logger.info("=" * 60)
            tickers = self.universe_mgr.get_tickers()
            result["tickers"] = tickers
            save_path = os.path.join(self.config.data.data_dir, "universe.json")
            with open(save_path, "w") as f:
                json.dump({"tickers": tickers, "universe": self.config.data.universe}, f)
        else:
            load_path = os.path.join(self.config.data.data_dir, "universe.json")
            with open(load_path) as f:
                tickers = json.load(f)["tickers"]
            result["tickers"] = tickers

        if "prices" in steps:
            logger.info("=" * 60)
            logger.info("STEP 2: Collecting price data")
            logger.info("=" * 60)
            price_data = self.price_collector.collect_all(tickers)
            result["prices"] = price_data

        if "fundamentals" in steps:
            logger.info("=" * 60)
            logger.info("STEP 3: Collecting fundamental data")
            logger.info("=" * 60)
            fundamentals = self.fundamental_collector.collect_all(tickers)
            sector_map   = self.fundamental_collector.get_sector_industry_map(fundamentals)
            result["fundamentals"] = fundamentals
            result["sector_map"]   = sector_map
            save_path = os.path.join(self.config.data.data_dir, "sector_map.json")
            with open(save_path, "w") as f:
                json.dump(sector_map, f, indent=2)

        if "news" in steps:
            logger.info("=" * 60)
            logger.info("STEP 4: Collecting news")
            logger.info("=" * 60)
            result["news"]     = self.news_collector.collect_yfinance_news(tickers)
            result["rss_news"] = self.news_collector.collect_rss_feeds()

        if "technical" in steps and "prices" in result:
            logger.info("=" * 60)
            logger.info("STEP 5: Calculating technical indicators")
            logger.info("=" * 60)
            result["prices_with_ta"] = self.ta_calculator.calculate_all(result["prices"])

        if "calendar" in steps:
            logger.info("=" * 60)
            logger.info("STEP 6: Scraping economic calendar")
            logger.info("=" * 60)
            result["economic_calendar"] = self.investing_scraper.scrape_economic_calendar()

        if "survival_labels" in steps and "prices" in result:
            logger.info("=" * 60)
            logger.info("STEP 7: Creating survival labels")
            logger.info("=" * 60)
            result["survival_labels"] = self.survival_labeler.label_all(result["prices"])

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        return result


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Quant Survival GNN Data Pipeline")
    parser.add_argument("--universe", choices=["SP500", "KOSPI200", "test"], default="test")
    parser.add_argument("--steps", nargs="+", default=None)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--target-return", type=float, default=0.10)
    parser.add_argument("--stop-loss", type=float, default=-0.05)
    args = parser.parse_args()

    if args.universe == "test":
        from config import get_test_config
        cfg = get_test_config()
    elif args.universe == "SP500":
        from config import get_sp500_config
        cfg = get_sp500_config()
    elif args.universe == "KOSPI200":
        from config import get_kospi_config
        cfg = get_kospi_config()

    cfg.data.start_date = args.start_date
    cfg.survival.target_return = args.target_return
    cfg.survival.stop_loss = args.stop_loss

    pipeline = DataPipeline(cfg)
    results  = pipeline.run(steps=args.steps)

    print("\n" + "=" * 60)
    print("DATA PIPELINE SUMMARY")
    print("=" * 60)
    for key, val in results.items():
        if isinstance(val, pd.DataFrame):
            print(f"  {key}: {val.shape}")
        elif isinstance(val, (dict, list)):
            print(f"  {key}: {len(val)} items")
    print("=" * 60)