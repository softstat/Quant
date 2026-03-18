"""
Quant Survival × GNN × LLaMA - Configuration
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data collection configuration"""
    # Target universe
    universe: str = "SP500"  # SP500, KOSPI200, custom
    custom_tickers: List[str] = field(default_factory=list)
    
    # Date range
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None  # None = today
    
    # Data frequency
    price_interval: str = "1d"  # 1d, 1wk
    
    # Storage
    data_dir: str = "./data"
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    parquet_dir: str = "./data/parquet"
    
    # Scraping
    investing_com_base: str = "https://www.investing.com"
    request_delay: float = 1.5  # seconds between requests
    max_retries: int = 3
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Technical indicators
    ta_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Fundamental ratios to collect
    fundamental_fields: List[str] = field(default_factory=lambda: [
        "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
        "enterpriseToRevenue", "enterpriseToEbitda", "returnOnEquity",
        "returnOnAssets", "debtToEquity", "currentRatio", "quickRatio",
        "operatingMargins", "profitMargins", "grossMargins",
        "revenueGrowth", "earningsGrowth", "freeCashflow",
        "marketCap", "beta", "dividendYield",
    ])
    
    # LLaMA embedding dimension
    embedding_dim: int = 128
    
    # Feature fusion
    quant_feature_dim: int = 64
    fused_feature_dim: int = 192  # quant + text embedding


@dataclass
class GraphConfig:
    """Graph construction configuration"""
    # Sector graph
    sector_edge_weight: float = 1.0
    industry_edge_weight: float = 0.7
    cross_sector_weight: float = 0.3
    
    # Supply chain graph
    supply_chain_weight: float = 0.8
    competitor_weight: float = 0.5
    
    # Dynamic edges
    correlation_window: int = 60  # trading days
    correlation_threshold: float = 0.6
    edge_update_frequency: str = "weekly"
    
    # Graph parameters
    max_neighbors: int = 20
    use_self_loops: bool = True


@dataclass
class SurvivalConfig:
    """Survival analysis configuration"""
    # Event definition
    target_return: float = 0.10      # +10% = profit event
    stop_loss: float = -0.05         # -5% = loss event
    max_holding_days: int = 60       # max observation window
    
    # Competing risks
    use_competing_risks: bool = True
    events: List[str] = field(default_factory=lambda: ["profit", "loss"])
    
    # Model
    model_type: str = "deepsurv"  # deepsurv, cox, rsf, drsa
    
    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 15
    dropout: float = 0.3
    weight_decay: float = 1e-4


@dataclass
class GATConfig:
    """Graph Attention Network configuration"""
    num_heads: int = 8
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    residual: bool = True
    layer_norm: bool = True
    
    # Training
    gat_lr: float = 0.0005
    gat_epochs: int = 200
    gat_patience: int = 20


@dataclass
class LLaMAConfig:
    """LLaMA model configuration"""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    embedding_model: str = "meta-llama/Llama-3.2-1B"
    
    # Inference
    max_tokens: int = 512
    temperature: float = 0.1
    batch_size: int = 8
    
    # LoRA fine-tuning
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # vLLM serving
    vllm_port: int = 8000
    tensor_parallel_size: int = 1
    
    # RAG
    vector_db_path: str = "./data/vector_db"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class PipelineConfig:
    """Master pipeline configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    survival: SurvivalConfig = field(default_factory=SurvivalConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    llama: LLaMAConfig = field(default_factory=LLaMAConfig)
    
    # Global
    seed: int = 42
    device: str = "cpu"  # auto-upgraded to cuda in train.py if available
    num_workers: int = 4
    log_level: str = "INFO"
    
    def __post_init__(self):
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.data.raw_dir, exist_ok=True)
        os.makedirs(self.data.processed_dir, exist_ok=True)
        os.makedirs(self.data.parquet_dir, exist_ok=True)


# ============================================================
# Quick config presets
# ============================================================

def get_sp500_config() -> PipelineConfig:
    """S&P 500 universe preset"""
    cfg = PipelineConfig()
    cfg.data.universe = "SP500"
    cfg.data.start_date = "2010-01-01"
    return cfg


def get_kospi_config() -> PipelineConfig:
    """KOSPI 200 universe preset"""
    cfg = PipelineConfig()
    cfg.data.universe = "KOSPI200"
    cfg.data.start_date = "2010-01-01"
    return cfg


def get_test_config() -> PipelineConfig:
    """Small test universe for development"""
    cfg = PipelineConfig()
    cfg.data.universe = "custom"
    cfg.data.custom_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "V", "JNJ",
        "WMT", "PG", "XOM", "UNH", "MA",
    ]
    cfg.data.start_date = "2022-01-01"
    cfg.device = "cpu"
    return cfg
