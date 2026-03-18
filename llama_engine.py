"""
Quant Survival x GNN x LLaMA
Module 3: LLaMA Multimodal Engine (Groq API Optimized)
"""

import json
import logging
import re
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# Prompt Templates
# ============================================================

SENTIMENT_PROMPT = """You are a financial sentiment analyst. Analyze the following financial news headline and provide:
1. sentiment_score: float from -1.0 (very bearish) to +1.0 (very bullish)
2. confidence: float from 0.0 to 1.0
3. event_type: one of [earnings, macro, policy, m_and_a, product, management, legal, market, other]
4. impact_duration: one of [short_term, medium_term, long_term]

Respond ONLY with valid JSON, no other text.

Headline: {headline}
Ticker: {ticker}

JSON:"""

RELATIONSHIP_PROMPT = """You are a financial relationship extractor. From the following news article, extract company relationships.

For each relationship found, provide:
- source: company ticker or name
- target: company ticker or name  
- relation_type: one of [supplier, customer, competitor, partner, acquirer, target]
- confidence: float 0.0-1.0

Respond ONLY with a JSON array. If no relationships found, return [].

Article: {article}

JSON:"""

REPORT_PROMPT = """You are a quantitative investment analyst. Generate a concise investment report based on the following model outputs.

Stock: {ticker} ({company_name})
Sector: {sector} | Industry: {industry}

=== Model Signals ===
Survival Probability (target +{target_pct}% in {horizon}d): {survival_prob:.1%}
Median Time to Target: {median_time} trading days
Competing Risk - Loss Probability: {loss_prob:.1%}
Risk-Adjusted Expected Return: {expected_return:.2%}

=== Key Features ===
{feature_summary}

=== Related Stocks (GNN Attention) ===
{attention_summary}

Generate a structured investment report with:
1. Signal Summary (BUY/HOLD/AVOID with conviction level)
2. Key Drivers (top 3 factors)
3. Risk Factors (top 3 risks)
4. Suggested Position Size (as % of portfolio, conservative)
5. Entry/Exit Strategy

Report:"""


# ============================================================
# LLaMA Client (Groq Integration)
# ============================================================

class LLaMAClient:
    """Unified LLaMA inference client using Groq API for generation 
    and sentence-transformers for local embeddings"""
    
    def __init__(self, config, backend: str = "groq"):
        self.config = config
        self.backend = backend
        self._client = None
        
        if self.backend == "groq":
            self._init_groq()
    
    def _init_groq(self):
        """Connect to Groq API"""
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY is not set in environment variables!")
            
        self._client = Groq(api_key=api_key)
        logger.info("Connected to Groq API for LLaMA 3.1")

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Generate text from LLaMA via Groq"""
        max_tokens = max_tokens or self.config.llama.max_tokens
        temperature = temperature or self.config.llama.temperature
        
        if self.backend == "groq":
            return self._generate_groq(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Backend {self.backend} is not supported in this Colab setup.")
    
    def _generate_groq(self, prompt, max_tokens, temperature):
        # Groq에서 지원하는 LLaMA 3.1 8B 모델 ID
        if self._client is None:
            return "LLM generation skipped because GROQ_API_KEY is not configured."
        model_id = "llama-3.1-8b-instant"
        try:
            response = self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            return ""

    def get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using local lightweight sentence-transformer"""
        return self._get_embedding_sentence_transformer(text)
    
    def _get_embedding_sentence_transformer(self, text):
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_st_model'):
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            embedding = self._st_model.encode(text)
            
            # config에 embedding_dim이 명시되어 있다면 맞추기
            target_dim = getattr(self.config.features, 'embedding_dim', 384)
            if len(embedding) > target_dim:
                return embedding[:target_dim]
            elif len(embedding) < target_dim:
                return np.pad(embedding, (0, target_dim - len(embedding)))
            return embedding
        except ImportError:
            logger.warning("sentence-transformers not available, using zeros")
            target_dim = getattr(self.config.features, 'embedding_dim', 384)
            return np.zeros(target_dim, dtype=np.float32)


# ============================================================
# Sentiment & Relationship Extractors
# ============================================================
class SentimentExtractor:
    def __init__(self, client: LLaMAClient):
        self.client = client
    
    def extract_single(self, headline: str, ticker: str = "") -> Dict:
        prompt = SENTIMENT_PROMPT.format(headline=headline, ticker=ticker)
        response = self.client.generate(prompt, max_tokens=200, temperature=0.05)
        try:
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"sentiment_score": 0.0, "event_type": "other"}

class RelationshipExtractor:
    def __init__(self, client: LLaMAClient):
        self.client = client
    
    def extract_from_article(self, article_text: str) -> List[Dict]:
        prompt = RELATIONSHIP_PROMPT.format(article=article_text[:2000])
        response = self.client.generate(prompt, max_tokens=500, temperature=0.05)
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return []

class FeatureEmbedder:
    """Convert text data to dense vectors for GNN node features"""
    
    def __init__(self, client: LLaMAClient, config):
        self.client = client
        self.config = config
    
    def embed_ticker_context(self, ticker: str, fundamentals: Dict, recent_news: List[str] = None) -> np.ndarray:
        """Create a context-aware embedding for a ticker"""
        context = f"Stock: {ticker}. "
        context += f"Sector: {fundamentals.get('sector', 'Unknown')}. "
        context += f"Industry: {fundamentals.get('industry', 'Unknown')}. "
        
        if recent_news:
            context += "Recent news: " + " | ".join(recent_news[:5])
        
        return self.client.get_embedding(context)
    
    def embed_all_tickers(self, tickers: List[str], fundamentals_df: pd.DataFrame, news_df: pd.DataFrame = None) -> np.ndarray:
        """Create embeddings for all tickers in the graph"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Generating LLaMA embeddings for {len(tickers)} tickers...")
        
        embeddings = []
        
        # 기본 정보 딕셔너리화
        fund_dict = {}
        if fundamentals_df is not None and len(fundamentals_df) > 0:
            for _, row in fundamentals_df.iterrows():
                # 'ticker' 컬럼이 있으면 사용, 없으면 인덱스 사용
                t_key = row.get("ticker", row.name)
                fund_dict[t_key] = row.to_dict()
        
        # 뉴스 정보 딕셔너리화
        news_dict = {}
        if news_df is not None and len(news_df) > 0:
            for ticker in tickers:
                ticker_news = news_df[news_df["ticker"] == ticker]["title"].tolist()
                news_dict[ticker] = ticker_news[:5]
        
        # 각 티커별로 임베딩 생성
        from tqdm import tqdm
        for ticker in tqdm(tickers, desc="Embedding Tickers"):
            fund = fund_dict.get(ticker, {"sector": "Unknown", "industry": "Unknown"})
            news = news_dict.get(ticker, [])
            emb = self.embed_ticker_context(ticker, fund, news)
            embeddings.append(emb)
        
        return np.array(embeddings)
# ReportGenerator 클래스 추가 업데이트
class ReportGenerator:
    """Generate investment reports from model outputs using LLaMA"""
    
    def __init__(self, client: LLaMAClient):
        self.client = client
    
    def generate_report(self, ticker: str, model_output: dict, context: dict) -> str:
        """
        GAT-Survival 모델의 출력값과 시장 컨텍스트를 결합하여 
        전문적인 투자 분석 리포트를 생성합니다.
        """
        # llama_engine.py 상단에 정의된 REPORT_PROMPT 템플릿 사용
        prompt = REPORT_PROMPT.format(
            ticker=ticker,
            company_name=context.get("name", ticker),
            sector=context.get("sector", "Unknown"),
            industry=context.get("industry", "Unknown"),
            target_pct=int(model_output.get("target_return", 0.1) * 100),
            horizon=model_output.get("max_holding_days", 60),
            survival_prob=model_output.get("survival_probability", 0.5),
            median_time=model_output.get("median_survival_time", 30),
            loss_prob=model_output.get("loss_probability", 0.2),
            expected_return=model_output.get("expected_return", 0.05),
            feature_summary=context.get("feature_summary", "N/A"),
            attention_summary=context.get("attention_summary", "N/A"),
        )
        
        # LLaMA를 통해 리포트 생성 (약간의 창의성을 위해 temperature 0.3 설정)
        return self.client.generate(prompt, max_tokens=800, temperature=0.3)