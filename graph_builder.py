"""
Quant Survival x GNN x LLaMA
Module 2: Graph Builder

Constructs multi-relational graphs for GAT:
  1. Sector/Industry graph (GICS-based hierarchy)
  2. Supply Chain graph (supplier-customer-competitor)
  3. Correlation graph (dynamic, rolling window)
  
Output: PyTorch Geometric Data objects
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class SectorIndustryGraph:
    """Build graph from GICS sector/industry classification"""
    
    CROSS_SECTOR_MAP = {
        "Technology": ["Communication Services", "Consumer Discretionary"],
        "Communication Services": ["Technology", "Consumer Discretionary"],
        "Consumer Discretionary": ["Technology", "Consumer Staples"],
        "Consumer Staples": ["Consumer Discretionary", "Healthcare"],
        "Healthcare": ["Technology", "Consumer Staples"],
        "Financials": ["Real Estate", "Industrials"],
        "Real Estate": ["Financials", "Utilities"],
        "Energy": ["Industrials", "Materials"],
        "Materials": ["Energy", "Industrials"],
        "Industrials": ["Materials", "Energy", "Technology"],
        "Utilities": ["Real Estate", "Energy"],
    }
    
    def __init__(self, sector_map: Dict, config=None):
        self.sector_map = sector_map
        self.config = config
    
    def build(self) -> Tuple[nx.Graph, Dict]:
        G = nx.Graph()
        tickers = list(self.sector_map["sector_map"].keys())
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        
        for ticker in tickers:
            G.add_node(ticker, idx=ticker_to_idx[ticker],
                       sector=self.sector_map["sector_map"].get(ticker, "Unknown"),
                       industry=self.sector_map["industry_map"].get(ticker, "Unknown"))
        
        edge_count = {"same_industry": 0, "same_sector": 0, "cross_sector": 0}
        
        # Same industry edges (weight=1.0)
        for industry, ind_tickers in self.sector_map["industries"].items():
            for i, t1 in enumerate(ind_tickers):
                for t2 in ind_tickers[i+1:]:
                    if t1 in ticker_to_idx and t2 in ticker_to_idx:
                        G.add_edge(t1, t2, weight=1.0, edge_type="same_industry")
                        edge_count["same_industry"] += 1
        
        # Same sector, different industry (weight=0.7)
        for sector, sec_tickers in self.sector_map["sectors"].items():
            industry_groups = {}
            for t in sec_tickers:
                ind = self.sector_map["industry_map"].get(t, "Unknown")
                industry_groups.setdefault(ind, []).append(t)
            industries = list(industry_groups.keys())
            for i, ind1 in enumerate(industries):
                for ind2 in industries[i+1:]:
                    for t1 in industry_groups[ind1]:
                        for t2 in industry_groups[ind2]:
                            if t1 in ticker_to_idx and t2 in ticker_to_idx and not G.has_edge(t1, t2):
                                G.add_edge(t1, t2, weight=0.7, edge_type="same_sector")
                                edge_count["same_sector"] += 1
        
        # Cross-sector edges (weight=0.3)
        for sector, related_sectors in self.CROSS_SECTOR_MAP.items():
            sec_tickers = self.sector_map["sectors"].get(sector, [])
            for related in related_sectors:
                rel_tickers = self.sector_map["sectors"].get(related, [])
                n_connect = min(5, len(sec_tickers), len(rel_tickers))
                for t1 in sec_tickers[:n_connect]:
                    for t2 in rel_tickers[:n_connect]:
                        if t1 in ticker_to_idx and t2 in ticker_to_idx and not G.has_edge(t1, t2):
                            G.add_edge(t1, t2, weight=0.3, edge_type="cross_sector")
                            edge_count["cross_sector"] += 1
        
        logger.info(f"Sector graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, ticker_to_idx


class CorrelationGraph:
    """Build dynamic correlation graph from return series"""
    
    def __init__(self, config=None):
        self.window = config.graph.correlation_window if config else 60
        self.threshold = config.graph.correlation_threshold if config else 0.6
    
    def build(self, price_data: Dict[str, pd.DataFrame], as_of_date: Optional[str] = None) -> nx.Graph:
        returns = {}
        for ticker, df in price_data.items():
            if "return_1d" in df.columns:
                returns[ticker] = df["return_1d"]
            elif "close" in df.columns:
                returns[ticker] = df["close"].pct_change()
        
        returns_df = pd.DataFrame(returns).dropna(how="all")
        if as_of_date:
            returns_df = returns_df.loc[:as_of_date]
        returns_df = returns_df.tail(self.window)
        corr_matrix = returns_df.corr()
        
        G = nx.Graph()
        tickers = list(corr_matrix.columns)
        for t in tickers:
            G.add_node(t)
        
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                corr = corr_matrix.loc[t1, t2]
                if not np.isnan(corr) and abs(corr) >= self.threshold:
                    G.add_edge(t1, t2, weight=abs(corr), correlation=corr,
                               edge_type="positive_corr" if corr > 0 else "negative_corr")
        
        logger.info(f"Correlation graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G


class SupplyChainGraph:
    """Build supply chain graph from known relationships + LLaMA NER"""
    
    KNOWN_SUPPLY_CHAINS = {
        "AAPL": {"suppliers": ["TSM", "QCOM", "AVGO", "TXN", "MU"], "competitors": ["MSFT", "GOOGL"]},
        "TSLA": {"suppliers": ["ALB"], "competitors": ["F", "GM", "RIVN"]},
        "AMZN": {"suppliers": ["UPS", "FDX"], "competitors": ["WMT", "SHOP"]},
        "NVDA": {"suppliers": ["TSM"], "competitors": ["AMD", "INTC"]},
        "MSFT": {"suppliers": ["INTC", "AMD", "NVDA"], "competitors": ["AAPL", "GOOGL", "CRM"]},
        "GOOGL": {"suppliers": [], "competitors": ["META", "MSFT", "AMZN"]},
        "META": {"suppliers": ["NVDA", "AMD"], "competitors": ["GOOGL", "SNAP"]},
        "JPM": {"suppliers": [], "competitors": ["BAC", "GS", "MS", "WFC"]},
    }
    
    def __init__(self, config=None):
        self.config = config
    
    def build_from_known(self, tickers: List[str]) -> nx.DiGraph:
        G = nx.DiGraph()
        for t in tickers:
            G.add_node(t)
        
        ticker_set = set(tickers)
        for company, relations in self.KNOWN_SUPPLY_CHAINS.items():
            if company not in ticker_set:
                continue
            for supplier in relations.get("suppliers", []):
                if supplier in ticker_set:
                    G.add_edge(supplier, company, weight=0.8, edge_type="supply_chain")
            for competitor in relations.get("competitors", []):
                if competitor in ticker_set:
                    G.add_edge(company, competitor, weight=0.5, edge_type="competitor")
                    G.add_edge(competitor, company, weight=0.5, edge_type="competitor")
        
        logger.info(f"Supply chain graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def add_llama_extracted_edges(self, G: nx.DiGraph, extracted_relations: List[Dict]) -> nx.DiGraph:
        n_added = 0
        for rel in extracted_relations:
            src, tgt = rel.get("source"), rel.get("target")
            conf = rel.get("confidence", 0.5)
            if src in G.nodes and tgt in G.nodes and conf > 0.7:
                G.add_edge(src, tgt, weight=conf * 0.6, edge_type=f"llama_{rel.get('relation_type', 'related')}")
                n_added += 1
        logger.info(f"Added {n_added} LLaMA-extracted edges")
        return G


class MultiRelationalGraphBuilder:
    """Merge all graph types into unified multi-relational graph"""
    
    EDGE_TYPE_MAP = {
        "same_industry": 0, "same_sector": 1, "cross_sector": 2,
        "positive_corr": 3, "negative_corr": 4,
        "supply_chain": 5, "competitor": 6, "self_loop": 7,
    }
    
    def __init__(self, config=None):
        self.config = config
    
    def merge_graphs(self, sector_graph, corr_graph, supply_graph):
        G = sector_graph.copy()
        
        for u, v, data in corr_graph.edges(data=True):
            if G.has_node(u) and G.has_node(v):
                if G.has_edge(u, v):
                    G[u][v]["weight"] = max(G[u][v]["weight"], data["weight"])
                else:
                    G.add_edge(u, v, **data)
        
        for u, v, data in supply_graph.edges(data=True):
            if G.has_node(u) and G.has_node(v):
                if G.has_edge(u, v):
                    G[u][v]["weight"] = max(G[u][v]["weight"], data["weight"])
                else:
                    G.add_edge(u, v, **data)
        
        tickers = sorted(G.nodes())
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        
        edge_index, edge_weight, edge_type_list = [], [], []
        
        for u, v, data in G.edges(data=True):
            i, j = ticker_to_idx[u], ticker_to_idx[v]
            w = data.get("weight", 0.5)
            et = self.EDGE_TYPE_MAP.get(data.get("edge_type", "same_sector"), 1)
            edge_index.extend([[i, j], [j, i]])
            edge_weight.extend([w, w])
            edge_type_list.extend([et, et])
        
        # Self-loops
        if self.config and self.config.graph.use_self_loops:
            for i in range(len(tickers)):
                edge_index.append([i, i])
                edge_weight.append(1.0)
                edge_type_list.append(7)
        
        graph_data = {
            "tickers": tickers, "ticker_to_idx": ticker_to_idx,
            "edge_index": np.array(edge_index).T,
            "edge_weight": np.array(edge_weight),
            "edge_type": np.array(edge_type_list),
            "num_nodes": len(tickers), "num_edges": len(edge_index),
            "num_edge_types": len(self.EDGE_TYPE_MAP),
            "networkx_graph": G,
        }
        
        logger.info(f"Merged graph: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
        return G, graph_data
    
    def to_pyg_data(self, graph_data, node_features):
        try:
            import torch
            from torch_geometric.data import Data
            data = Data(
                x=torch.FloatTensor(node_features),
                edge_index=torch.LongTensor(graph_data["edge_index"]),
                edge_attr=torch.FloatTensor(np.stack([graph_data["edge_weight"], graph_data["edge_type"]], axis=1)),
            )
            data.tickers = graph_data["tickers"]
            data.ticker_to_idx = graph_data["ticker_to_idx"]
            return data
        except ImportError:
            logger.warning("PyTorch Geometric not installed")
            graph_data["node_features"] = node_features
            return graph_data


def build_full_graph(sector_map, price_data, tickers, config=None):
    sector_builder = SectorIndustryGraph(sector_map, config)
    sector_graph, _ = sector_builder.build()
    
    corr_builder = CorrelationGraph(config)
    corr_graph = corr_builder.build(price_data)
    
    supply_builder = SupplyChainGraph(config)
    supply_graph = supply_builder.build_from_known(tickers)
    
    merger = MultiRelationalGraphBuilder(config)
    return merger.merge_graphs(sector_graph, corr_graph, supply_graph)
