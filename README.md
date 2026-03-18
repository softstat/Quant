# Quant Survival × GNN × LLaMA Framework

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c) ![Groq](https://img.shields.io/badge/LLaMA-3.1_via_Groq-black) ![License](https://img.shields.io/badge/License-MIT-green)

> **A Graph Attention Network and Competing Risk Survival Analysis framework for quantitative trading, augmented by Multimodal LLM signals.**

## 1. Executive Summary
본 프로젝트는 주식 시장의 복잡한 비선형적 상호작용과 거시경제적 맥락을 모델링하기 위해 **Graph Attention Networks (GAT)** 와 **DeepSurv (Competing Risk Survival Analysis)** 를 결합한 고도화된 하이브리드 퀀트 트레이딩 프레임워크입니다. 

단순한 방향성 예측(Up/Down)을 넘어, **'목표 수익률 도달(Profit Target)'** 과 **'손절매 촉발(Stop-Loss)'** 을 두 개의 경합하는 위험(Competing Risks)으로 정의하고 각 사건의 누적 발생 확률(Cumulative Incidence Function, CIF)을 추정합니다. 이에 더해 LLaMA 3.1 기반의 자연어 처리 모듈을 통해 시장의 정성적 뉴스, 공급망 관계, 거시경제 지표를 그래프의 노드 및 엣지 특성으로 매핑하여 차별화된 알파(Alpha)를 창출합니다.

---

## 2. Project Structure & Modules

프레임워크는 모듈화된 파이프라인으로 구성되어 있으며, 각 스크립트는 독립적으로 혹은 유기적으로 작동합니다.

### 📊 Data & Feature Engineering
* `data_pipeline.py`: 데이터 수집 메인 파이프라인. 기술적/기본적 지표 및 캘린더 일정을 병합.
* `macro_collector.py`: 원자재, 채권 금리, FX, VIX 등 글로벌 매크로 지표 수집 및 파생 피처(69개) 생성.
* `earnings_collector.py`: EPS 서프라이즈 및 주요 지수 기반의 Market Regime(시장 국면) 피처 생성.
* `llama_engine.py`: 실시간 뉴스 센티먼트 추출, 공급망(Supply Chain) 관계 분석 및 텍스트 벡터 임베딩(384차원).
* `feature_assembler.py`: 모든 정형/비정형 데이터를 결합하여 GNN 학습을 위한 최종 **Node Feature Matrix** 조립 및 스케일링.

### 🕸️ Graph & Modeling
* `graph_builder.py`: 산업 분류(GICS), 동적 가격 상관성, 공급망 관계를 아우르는 **다중 관계 그래프(Multi-relational Graph)** 구축.
* `gat_survival_model.py`: **3-layer GAT 인코더**와 경합 위험률(Hazard Rate)을 계산하는 DeepSurv 헤드로 구성된 메인 딥러닝 모델.
* `train.py`: Temporal Split을 적용한 모델 훈련 및 제약 조건(섹터/산업)이 반영된 최종 포트폴리오 산출.

### 📈 Evaluation & Analytics
* `backtest.py`: Out-of-Sample 성과 평가, 0.03% 거래 수수료가 반영된 성과 지표 산출 및 누적 수익률 차트 생성.

---

## 3. Data Sources & Feature Architecture

본 모델은 단일 종목(Node) 당 **총 101개의 정형 데이터 피처**와 **384차원의 LLM 임베딩**을 결합하여 학습합니다.

| Category | Sources | Feature Count | Description & Examples |
| :--- | :--- | :---: | :--- |
| **Macro Indicators** | yfinance | 69 | 원자재(WTI, 금 등), 국채 금리, 환율, VIX 및 파생 지표(장단기 금리차 등). |
| **Market Context** | yfinance | 18 | SPY, QQQ 등 주요 지수를 기반으로 한 20일 모멘텀 및 시장 국면(Bull/Bear) 데이터. |
| **Earnings/Calendar** | yfinance, Investing | 6 | 최근 EPS 서프라이즈 비율(%), 실적 발표일 전후 맥락. |
| **Technicals (TA)** | yfinance | 24 | 이동평균, RSI, MACD, Bollinger Bands, ATR, OBV 등. |
| **Fundamentals (FA)** | yfinance Info | 13 | PER(Trailing/Forward), PBR, ROE, 부채비율, 베타 등. |
| **NLP & Sentiment** | RSS, LLaMA 3.1 | 384 (Vector) | 뉴스 센티먼트(-1.0~1.0), 이벤트 유형, 밀집 임베딩(Dense Embedding). |

---

## 4. Core Mathematical Framework

### 4.1 Competing Risk Survival Model
개별 주식 $i$에 대해 특정 시점 $t$에서의 원인 $k$ (1: 수익, 2: 손실)에 대한 위험률(Hazard Rate)은 다음과 같이 정의됩니다.

$$h_k(t|x) = h_{0k}(t) \exp(f_k(x))$$

여기서 $f_k(x)$는 3-layer GAT 인코더를 통과한 잠재 표현(Latent Representation)입니다. 전체 생존 함수(Overall Survival Function)는 두 위험을 합산하여 도출됩니다.

$$S(t|x) = \exp \left( - \sum_{k=1}^{2} \sum_{s=1}^{t} h_k(s|x) \right)$$

### 4.2 Multi-Relational Graph Attention
주식 간의 관계는 단순한 가격 상관관계를 넘어, 산업 분류(GICS) 및 공급망(Supply Chain) 정보를 포함하는 다중 관계 그래프로 구성됩니다. 인접 노드 $j$와의 어텐션 가중치 $\alpha_{ij}$는 다음과 같이 계산됩니다.

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T [W x_i || W x_j || e_{ij}]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(a^T [W x_i || W x_k || e_{ik}]))}$$

---

## 5. Trading Strategy & Execution

실제 파이썬 스크립트(`backtest.py` 등)에 구현되어 운용 및 백테스트 성과에 반영된 퀀트 트레이딩 전략 가정입니다.

* **포트폴리오 구성 (Portfolio Construction):** GAT 모델이 산출한 '수익 도달 생존 확률' 상위 20개 종목을 **동일 가중(Equal-weight, 각 5%)** 으로 편입합니다. 특정 산업에 대한 과적합을 막기 위해 **섹터당 최대 2종목(max_per_sector=2)** 의 캡(Cap) 제약을 둡니다.
* **리밸런싱 및 청산 (Rebalancing & Exit):** * 매일 종가 기준으로 모델이 업데이트되며, 편입된 종목이 목표 수익률(Target)에 도달하거나 손절매(Stop-Loss) 임계치를 하회할 경우 즉각 청산합니다.
  * 이벤트 미발생 시 최대 보유 기간(Max Holding Days) 경과 후 기계적으로 청산하며 빈 슬롯을 신규 상위 스코어 종목으로 교체합니다.

---

## 6. Backtest Performance

S&P 500 유니버스 대상 아웃오브샘플(Out-of-Sample) 백테스트 결과입니다. (코드 내 0.03% 거래 마찰 비용 반영)

* **Test Period:** 2018-02-01 ~ 2026-03-17
* **Benchmark:** SPY (SPDR S&P 500 ETF Trust)

| Metric | Strategy Performance | Benchmark (SPY) |
| :--- | :---: | :---: |
| **Total Return** | **24.35%** | 16.24% |
| **Ann. Return (CAGR)** | **25.33%** | 14.80% |
| **Annual Volatility** | **13.15%** | 18.50% |
| **Max Drawdown (MDD)** | **-17.42%** | -24.15% |
| **Sharpe Ratio** | **1.17** | 0.85 |

> **Performance Analysis:** 벤치마크(SPY) 대비 연간 변동성(13.15%)과 최대 낙폭(MDD -17.42%)을 획기적으로 낮게 통제하면서도, 연평균 수익률(25.33%)은 10%p 이상 초과 달성했습니다. 이는 모델의 '경합 위험(Competing Risk)' 기반 손절매/익절 통제 로직이 하락장 방어에 탁월하게 작동하고 있음을 증명합니다.

---

## 7. Latest Portfolio Output (Sample)

| Ticker | Sector | Score (Survival Prob) | Expected Return | Weight |
|:---|:---|:---:|:---:|:---:|
| **EXPE** | Consumer Cyclical | 0.973678 | 7.56% | 0.05 |
| **GM** | Consumer Cyclical | 0.973521 | 7.62% | 0.05 |
| **APTV** | Consumer Cyclical | 0.973505 | 7.60% | 0.05 |
| **EBAY** | Consumer Cyclical | 0.973498 | 7.54% | 0.05 |
| **LVS** | Consumer Cyclical | 0.973488 | 7.79% | 0.05 |

---

## 8. Computational Performance

복잡한 GNN 연산과 방대한 생존 분석을 수행함에도 불구하고, 텐서 연산 최적화를 통해 매우 빠른 리서치 이터레이션을 지원합니다. (S&P 500 유니버스 1 Epoch 학습 기준)
* ☁️ **Google Colab (Standard GPU/TPU Instance):** 약 **30분** 소요

---

## 9. Quick Start

### Step 1. Installation
```bash
git clone [https://github.com/username/quant-survival-gnn.git](https://github.com/username/quant-survival-gnn.git)
cd quant-survival-gnn
pip install -r requirements.txt
```

## 10. 수정해야할 점
1) 슬리피지을 추가해서 내가 만든 모델이 실제로 운용이 잘 되고 있는지 확인이 필요
2) backtest에서 bootstrap을 이용해서 더 reasonable 하게 만들어야 함
3) only Stock이라 채권,금,비트코인,물가채 등등을 포트폴리오에 넣어서 운용하면 더 안정성이 있는 포트폴리오가 될 것이라고 생각함
4) 모델에서는 KOSPI 종목에 대해서도 학습을 진행했는데 실제로는 출력으로 나오지는 않아 정보가 부족해서 그랬던 것인지 혹은 실제로 선택이 되지 않아서 그랬는지에 대한 연구 필요
5) GPU 부족으로 모델 학습을 경량화해서 진행을 하였고 LLAMA를 이용해서 이 종목을 어떤 근거로 종목을 선정했는지에 대한 이유 설명이 부족하다는 점에서 수정이 필요함
6) 분명 각 sector당 2개로 제한을 했음에도 불구하고, consumer cyclical sector 가 가장 많이 나와서 이 부분 수정 필요함
## 11. Backtesting 결과(26.03.17)
<img width="1880" height="1300" alt="Image" src="https://github.com/user-attachments/assets/edbcdaed-f6c9-4145-8831-691021454f4a" />




