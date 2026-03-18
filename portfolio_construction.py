"""
portfolio_construction.py
-------------------------
Construct a diversified long-only portfolio from GAT survival model rankings.

construct_portfolio() 인자:
  rankings        : MultiMarketSignalRanker.rank() 결과 DataFrame
  sector_map      : DataPipeline이 생성한 sector_map dict
                    {"sector_map": {ticker: sector}, "industry_map": {ticker: industry}, ...}
  total_n         : 최종 편입 종목 수 (default 15)
  max_per_sector  : 섹터별 최대 편입 수 (default 3)
  max_per_industry: 산업별 최대 편입 수 (default 2)
  min_score_col   : 정렬 기준 컬럼 (default "score")
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. sector_map 주입
# ─────────────────────────────────────────────

def _inject_sector_info(
    df: pd.DataFrame,
    sector_map: Optional[Dict],
) -> pd.DataFrame:
    """
    rankings DataFrame에 sector / industry 컬럼이 없거나 비어있을 때
    sector_map에서 채워넣는다.

    sector_map 구조 (data_pipeline.py의 get_sector_industry_map 출력):
      {
        "sector_map":   {ticker: sector_str},
        "industry_map": {ticker: industry_str},
        "sectors":      {sector_str: [ticker, ...]},
        "industries":   {industry_str: [ticker, ...]},
      }
    """
    out = df.copy()

    if sector_map is None:
        return out

    s_map = sector_map.get("sector_map", {})
    i_map = sector_map.get("industry_map", {})

    if "ticker" not in out.columns:
        logger.warning("rankings에 ticker 컬럼 없음 → sector 주입 불가")
        return out

    # sector
    if "sector" not in out.columns or out["sector"].isna().all():
        out["sector"] = out["ticker"].map(s_map).fillna("Unknown")
    else:
        # 기존 컬럼에서 비어있는 행만 보완
        mask = out["sector"].isna() | (out["sector"] == "") | (out["sector"] == "Unknown")
        out.loc[mask, "sector"] = out.loc[mask, "ticker"].map(s_map).fillna("Unknown")

    # industry
    if "industry" not in out.columns or out["industry"].isna().all():
        out["industry"] = out["ticker"].map(i_map).fillna("Unknown")
    else:
        mask = out["industry"].isna() | (out["industry"] == "") | (out["industry"] == "Unknown")
        out.loc[mask, "industry"] = out.loc[mask, "ticker"].map(i_map).fillna("Unknown")

    return out


# ─────────────────────────────────────────────
# 2. 필수 컬럼 보완
# ─────────────────────────────────────────────

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "sector"   not in out.columns: out["sector"]   = "Unknown"
    if "industry" not in out.columns: out["industry"] = "Unknown"
    if "market"   not in out.columns: out["market"]   = "US"
    if "signal"   not in out.columns: out["signal"]   = "HOLD"

    if "score" not in out.columns:
        if "entry_score" in out.columns:
            out["score"] = out["entry_score"]
        elif "expected_return" in out.columns:
            out["score"] = out["expected_return"]
        else:
            out["score"] = 0.0

    if "expected_return" not in out.columns: out["expected_return"] = 0.0
    if "profit_prob"     not in out.columns: out["profit_prob"]     = np.nan
    if "loss_prob"       not in out.columns: out["loss_prob"]       = np.nan

    return out


# ─────────────────────────────────────────────
# 3. 그리디 섹터/산업 다각화 선택
# ─────────────────────────────────────────────

def _greedy_select(
    df: pd.DataFrame,
    total_n: int,
    max_per_sector: Optional[int],
    max_per_industry: Optional[int],
) -> pd.DataFrame:
    selected_rows   = []
    sector_counts   = {}
    industry_counts = {}

    for _, row in df.iterrows():
        sector   = row.get("sector",   "Unknown")
        industry = row.get("industry", "Unknown")

        if max_per_sector is not None and sector_counts.get(sector, 0) >= max_per_sector:
            continue
        if max_per_industry is not None and industry_counts.get(industry, 0) >= max_per_industry:
            continue

        selected_rows.append(row)
        sector_counts[sector]     = sector_counts.get(sector, 0) + 1
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

        if len(selected_rows) >= total_n:
            break

    if not selected_rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(selected_rows).reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. 가중치 계산 (Equal / Score-proportional)
# ─────────────────────────────────────────────

def _assign_weights(
    portfolio: pd.DataFrame,
    score_col: str = "score",
    weighting: str = "equal",  # "equal" | "score"
) -> pd.DataFrame:
    n = len(portfolio)
    if n == 0:
        portfolio["weight"] = pd.Series(dtype=float)
        return portfolio

    if weighting == "score" and score_col in portfolio.columns:
        scores = portfolio[score_col].clip(lower=0)
        total  = scores.sum()
        if total > 1e-9:
            portfolio["weight"] = scores / total
        else:
            portfolio["weight"] = 1.0 / n
    else:
        portfolio["weight"] = 1.0 / n

    return portfolio


# ─────────────────────────────────────────────
# 5. 메인 함수
# ─────────────────────────────────────────────

def construct_portfolio(
    rankings: pd.DataFrame,
    sector_map: Optional[Dict] = None,   # ✅ sector_map 정식 지원
    total_n: int = 15,
    max_per_sector: int = 3,
    max_per_industry: int = 2,
    min_score_col: str = "score",
    weighting: str = "equal",            # "equal" | "score"
    relax_on_shortage: bool = True,      # 후보 부족 시 섹터 제한 완화 여부
) -> pd.DataFrame:
    """
    Parameters
    ----------
    rankings         : MultiMarketSignalRanker.rank() 반환 DataFrame
    sector_map       : data_pipeline.get_sector_industry_map() 반환 dict
    total_n          : 최종 편입 종목 수
    max_per_sector   : 섹터별 최대 편입 (None = 제한 없음)
    max_per_industry : 산업별 최대 편입 (None = 제한 없음)
    min_score_col    : 1차 정렬 기준 컬럼
    weighting        : 가중치 방식 ("equal" | "score")
    relax_on_shortage: 종목 부족 시 섹터 제한 완화하여 total_n 채우기

    Returns
    -------
    pd.DataFrame : ticker / weight / sector / industry / score / expected_return 포함
    """

    if rankings is None or rankings.empty:
        logger.warning("construct_portfolio: 빈 rankings 수신")
        return pd.DataFrame()

    # ── Step 1: sector 정보 주입 ──────────────
    df = _inject_sector_info(rankings, sector_map)
    df = _ensure_columns(df)

    if min_score_col not in df.columns:
        raise ValueError(f"min_score_col='{min_score_col}' 컬럼이 rankings에 없습니다.")

    # ── Step 2: 필터링 (BUY 신호 + 양의 기대수익) ──
    df = df[df["signal"].isin(["BUY", "STRONG_BUY"])].copy()
    df = df[df["expected_return"] > 0].copy()

    # profit_prob - loss_prob > 2% 필터 (컬럼이 있을 때만)
    if {"profit_prob", "loss_prob"}.issubset(df.columns):
        valid_probs = (
            df["profit_prob"].notna()
            & df["loss_prob"].notna()
            & ((df["profit_prob"] - df["loss_prob"]) > 0.02)
        )
        df = df[valid_probs].copy()

    if df.empty:
        logger.warning("BUY 조건을 통과한 후보가 없습니다")
        return pd.DataFrame()

    # ── Step 3: 정렬 ──────────────────────────
    sort_cols  = [min_score_col]
    ascending  = [False]
    if "expected_return" in df.columns and "expected_return" != min_score_col:
        sort_cols.append("expected_return")
        ascending.append(False)
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    # ── Step 4: 섹터 다각화 선택 ──────────────
    portfolio = _greedy_select(
        df,
        total_n=total_n,
        max_per_sector=max_per_sector,
        max_per_industry=max_per_industry,
    )

    # ── Step 5: 부족 시 제한 완화 보충 ──────────
    if relax_on_shortage and len(portfolio) < total_n:
        shortage = total_n - len(portfolio)
        logger.info(
            f"섹터 제한 적용 후 {len(portfolio)}개 → "
            f"제한 완화하여 {shortage}개 추가 시도"
        )
        already    = set(portfolio["ticker"]) if "ticker" in portfolio.columns else set()
        remaining  = df[~df["ticker"].isin(already)].copy()
        relaxed    = _greedy_select(
            remaining,
            total_n=shortage,
            max_per_sector=None,
            max_per_industry=None,
        )
        if not relaxed.empty:
            portfolio = pd.concat([portfolio, relaxed], axis=0).reset_index(drop=True)

    if portfolio.empty:
        logger.warning("최종 포트폴리오가 비어있습니다")
        return pd.DataFrame()

    # ── Step 6: 중복 제거 + 정렬 ──────────────
    if "ticker" in portfolio.columns:
        portfolio = portfolio.drop_duplicates(subset=["ticker"], keep="first")
    portfolio = portfolio.sort_values(min_score_col, ascending=False).reset_index(drop=True)

    # ── Step 7: 가중치 부여 ───────────────────
    portfolio = _assign_weights(portfolio, score_col=min_score_col, weighting=weighting)

    # ── 로그 요약 ─────────────────────────────
    logger.info(f"✅ Portfolio: {len(portfolio)}개 종목 선택")
    if "sector" in portfolio.columns:
        logger.info(f"   섹터 분포: {portfolio['sector'].value_counts(dropna=False).to_dict()}")
    if "industry" in portfolio.columns:
        logger.info(f"   산업 분포: {portfolio['industry'].value_counts(dropna=False).to_dict()}")
    if "expected_return" in portfolio.columns:
        logger.info(
            f"   기대수익률: avg={portfolio['expected_return'].mean():.3f} "
            f"max={portfolio['expected_return'].max():.3f}"
        )

    return portfolio