"""
Quant Survival x GNN x LLaMA
MODEL CORE: GAT + DeepSurv Competing Risk

Architecture:
  Input Features [N, F] --> GAT Encoder [N, H] --> Survival Head --> {
      Cause 1: profit target reached  (hazard_profit)
      Cause 2: stop-loss triggered    (hazard_loss)
  }

  Entry Signal = P(profit before loss within T days)
  Expected Return = integral of survival function * target_return

Multi-market support: US (S&P500) + Korea (KOSPI)
Earnings-aware: uses earnings surprise + calendar as time-varying covariates
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# 1. GAT Encoder
# ============================================================

class GATLayer(nn.Module):
    """Single Graph Attention Layer with multi-head attention"""

    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.3,
                 residual=True, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        self.residual = residual

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.FloatTensor(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.FloatTensor(num_heads, out_dim))

        # Edge type embedding (8 types: sector, industry, corr, supply chain, etc.)
        self.edge_type_emb = nn.Embedding(8, num_heads)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if residual:
            if in_dim != (out_dim * num_heads if concat else out_dim):
                self.res_proj = nn.Linear(in_dim, out_dim * num_heads if concat else out_dim)
            else:
                self.res_proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(out_dim * num_heads if concat else out_dim)

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] edge connectivity
            edge_type: [E] edge type indices (optional)
            edge_weight: [E] edge weights (optional)
        Returns:
            [N, out_dim * num_heads] if concat, else [N, out_dim]
        """
        N = x.size(0)
        H = self.num_heads
        D = self.out_dim

        # Linear transform + reshape to [N, H, D]
        h = self.W(x).view(N, H, D)

        # Attention coefficients
        src_idx, dst_idx = edge_index[0], edge_index[1]
        e_src = (h[src_idx] * self.a_src.unsqueeze(0)).sum(dim=-1)  # [E, H]
        e_dst = (h[dst_idx] * self.a_dst.unsqueeze(0)).sum(dim=-1)  # [E, H]
        e = self.leaky_relu(e_src + e_dst)  # [E, H]

        # Edge type modulation
        if edge_type is not None:
            et_emb = self.edge_type_emb(edge_type.long())  # [E, H]
            e = e * (1.0 + et_emb)

        # Edge weight modulation
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(-1)

        # Softmax attention per destination node
        e_max = torch.zeros(N, H, device=x.device)
        e_max.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand(-1, H), e, reduce="amax")
        e = torch.exp(e - e_max[dst_idx])

        e_sum = torch.zeros(N, H, device=x.device)
        e_sum.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, H), e)
        alpha = e / (e_sum[dst_idx] + 1e-10)  # [E, H]

        alpha = self.dropout(alpha)

        # Aggregate
        msg = h[src_idx] * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros(N, H, D, device=x.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, H, D), msg)

        if self.concat:
            out = out.view(N, H * D)
        else:
            out = out.mean(dim=1)

        # Residual + LayerNorm
        if self.residual:
            out = out + self.res_proj(x)
        out = self.layer_norm(out)

        return out


class GATEncoder(nn.Module):
    """Multi-layer GAT encoder

    Produces stock embeddings that capture:
    - Individual stock features (technical + fundamental + earnings)
    - Market context (SPY, QQQ, KOSPI regime)
    - Inter-stock relationships (sector, supply chain, correlation)
    """

    def __init__(self, in_dim, hidden_dim=128, out_dim=128,
                 num_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(GATLayer(
            in_dim, hidden_dim, num_heads=num_heads,
            dropout=dropout, residual=True, concat=True,
        ))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(
                hidden_dim * num_heads, hidden_dim, num_heads=num_heads,
                dropout=dropout, residual=True, concat=True,
            ))

        # Output layer (no concat, average heads)
        self.layers.append(GATLayer(
            hidden_dim * num_heads, out_dim, num_heads=1,
            dropout=dropout, residual=True, concat=False,
        ))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type, edge_weight)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        return x


# ============================================================
# 2. DeepSurv Competing Risk Head
# ============================================================

class CompetingRiskHead(nn.Module):
    """DeepSurv-based competing risk model

    Models two cause-specific hazards:
      h_1(t|x): hazard of profit target being reached
      h_2(t|x): hazard of stop-loss being triggered

    The cause-specific cumulative incidence function (CIF) gives:
      F_k(t|x) = P(T <= t, event=k | x)

    Entry signal = F_1(T_max|x) / [F_1(T_max|x) + F_2(T_max|x)]
    Expected return = sum_t S(t|x) * target_return * dt / T_max
    """

    def __init__(self, in_dim, hidden_dim=128, num_time_bins=60,
                 num_risks=2, dropout=0.3):
        super().__init__()
        self.num_risks = num_risks
        self.num_time_bins = num_time_bins

        # Shared representation
        self.shared_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.AlphaDropout(dropout),
        )

        # Cause-specific sub-networks
        self.risk_nets = nn.ModuleList()
        for _ in range(num_risks):
            self.risk_nets.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SELU(),
                nn.AlphaDropout(dropout),
                nn.Linear(hidden_dim // 2, num_time_bins),
            ))

        # Baseline hazard (learnable, cause-specific)
        self.baseline_hazard = nn.Parameter(
            torch.zeros(num_risks, num_time_bins)
        )

    def forward(self, x):
        """
        Args:
            x: [B, in_dim] stock embeddings from GAT

        Returns:
            hazards: [B, num_risks, T] cause-specific hazard rates
            survival: [B, T] overall survival function
            cif: [B, num_risks, T] cumulative incidence functions
        """
        shared = self.shared_net(x)  # [B, hidden_dim]

        # Cause-specific log-hazards
        log_hazards = []
        for k in range(self.num_risks):
            lh = self.risk_nets[k](shared)  # [B, T]
            lh = lh + self.baseline_hazard[k].unsqueeze(0)
            log_hazards.append(lh)

        log_hazards = torch.stack(log_hazards, dim=1)  # [B, K, T]
        hazards = F.softplus(log_hazards)  # ensure positive

        # Overall hazard: sum of cause-specific hazards
        total_hazard = hazards.sum(dim=1)  # [B, T]

        # Overall survival: S(t) = exp(-cumulative_hazard)
        cum_hazard = total_hazard.cumsum(dim=-1)
        survival = torch.exp(-cum_hazard)  # [B, T]

        # Cumulative incidence function (CIF) for each cause
        # F_k(t) = sum_{s<=t} h_k(s) * S(s-1)
        survival_shifted = torch.cat([
            torch.ones(x.size(0), 1, device=x.device),
            survival[:, :-1]
        ], dim=1)  # [B, T], S(t-1)

        cif = hazards * survival_shifted.unsqueeze(1)  # [B, K, T]
        cif = cif.cumsum(dim=-1)  # cumulative

        return hazards, survival, cif


# ============================================================
# 3. Full Model: GAT-DeepSurv
# ============================================================

class GATSurvivalModel(nn.Module):
    """End-to-end GAT + DeepSurv Competing Risk Model

    Pipeline:
        Raw Features --> GAT Encoder --> Stock Embeddings
                                          |
                                          v
                              CompetingRiskHead
                                    |
                    +---------------+---------------+
                    |                               |
              Profit Hazard                   Loss Hazard
                    |                               |
              P(profit by t)                 P(loss by t)
                    |                               |
                    +---------- Signals -----------+
                    |                               |
              Entry Score                  Expected Return
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        feature_dim = config.get("feature_dim", 128)
        gat_hidden = config.get("gat_hidden", 128)
        gat_out = config.get("gat_out", 128)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("num_layers", 3)
        dropout = config.get("dropout", 0.3)
        num_time_bins = config.get("max_holding_days", 60)
        num_risks = config.get("num_risks", 2)

        self.gat_encoder = GATEncoder(
            in_dim=feature_dim,
            hidden_dim=gat_hidden,
            out_dim=gat_out,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.survival_head = CompetingRiskHead(
            in_dim=gat_out,
            hidden_dim=gat_hidden,
            num_time_bins=num_time_bins,
            num_risks=num_risks,
            dropout=dropout,
        )

        # Target return and stop loss
        self.target_return = config.get("target_return", 0.10)
        self.stop_loss = config.get("stop_loss", -0.05)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        """
        Returns:
            stock_embeddings: [N, gat_out]
            hazards: [N, 2, T] (profit hazard, loss hazard)
            survival: [N, T]
            cif: [N, 2, T]
            signals: dict with entry scores and expected returns
        """
        # GAT encoding
        embeddings = self.gat_encoder(x, edge_index, edge_type, edge_weight)

        # Survival prediction
        hazards, survival, cif = self.survival_head(embeddings)

        # Compute trading signals
        signals = self._compute_signals(survival, cif)

        return {
            "embeddings": embeddings,
            "hazards": hazards,
            "survival": survival,
            "cif": cif,
            "signals": signals,
        }

    def _compute_signals(self, survival, cif):
        """Convert survival model output to trading signals

        Args:
            survival: [N, T] overall survival probability
            cif: [N, 2, T] cumulative incidence (profit, loss)
        """
        T = survival.size(1)
        profit_cif = cif[:, 0, :]  # [N, T] P(profit by t)
        loss_cif = cif[:, 1, :]    # [N, T] P(loss by t)

        # Final cumulative incidence at horizon
        profit_prob = profit_cif[:, -1]  # P(profit within T days)
        loss_prob = loss_cif[:, -1]      # P(loss within T days)
        censor_prob = survival[:, -1]    # P(neither event)

        # Entry score: probability-weighted
        entry_score = profit_prob / (profit_prob + loss_prob + 1e-10)

        # Expected return (risk-adjusted)
        # E[R] = P(profit) * target - P(loss) * |stop_loss| - P(censor) * 0
        expected_return = (
            profit_prob * self.target_return
            + loss_prob * self.stop_loss
        )

        # Median survival time (time when S(t) crosses 0.5)
        below_half = (survival < 0.5).float()
        # First time survival < 0.5
        has_event = below_half.sum(dim=-1) > 0
        median_time = torch.where(
            has_event,
            below_half.argmax(dim=-1).float() + 1,
            torch.tensor(float(T), device=survival.device),
        )

        # Time to profit peak (when profit CIF grows fastest)
        profit_hazard_peak = cif[:, 0, :].diff(dim=-1)
        if profit_hazard_peak.size(-1) > 0:
            peak_time = profit_hazard_peak.argmax(dim=-1).float() + 1
        else:
            peak_time = torch.ones(survival.size(0), device=survival.device) * T

        # Sharpe-like ratio: expected_return / volatility proxy
        volatility_proxy = torch.sqrt(
            profit_prob * (self.target_return - expected_return)**2
            + loss_prob * (self.stop_loss - expected_return)**2
            + 1e-10
        )
        risk_adjusted_score = expected_return / (volatility_proxy + 1e-10)

        return {
            "entry_score": entry_score,
            "expected_return": expected_return,
            "profit_prob": profit_prob,
            "loss_prob": loss_prob,
            "censor_prob": censor_prob,
            "median_survival_time": median_time,
            "peak_profit_time": peak_time,
            "risk_adjusted_score": risk_adjusted_score,
            "survival_curve": survival,
            "profit_cif": profit_cif,
            "loss_cif": loss_cif,
        }


# ============================================================
# 4. Loss Functions
# ============================================================

class CompetingRiskLoss(nn.Module):
    """Loss function for competing risk survival analysis

    Combines:
      1. Cause-specific negative log-likelihood
      2. Concordance regularization
      3. Calibration loss (optional)
    """

    def __init__(self, alpha_concordance=0.1, alpha_calibration=0.05):
        super().__init__()
        self.alpha_c = alpha_concordance
        self.alpha_cal = alpha_calibration

    def forward(self, hazards, survival, cif, durations, event_types):
        """
        Args:
            hazards: [B, K, T]
            survival: [B, T]
            cif: [B, K, T]
            durations: [B] time to event (integer, 1-indexed)
            event_types: [B] 0=censored, 1=profit, 2=loss
        """
        B = durations.size(0)
        device = hazards.device

        # Clamp durations to valid range
        T = hazards.size(2)
        durations = durations.clamp(1, T).long()
        time_idx = durations - 1  # 0-indexed

        # 1. Cause-specific NLL
        nll = torch.tensor(0.0, device=device)

        for k in range(hazards.size(1)):
            event_mask = (event_types == (k + 1)).float()  # events for cause k
            censor_mask = (event_types == 0).float()

            # For observed events: log[h_k(t_i)] + log[S(t_i - 1)]
            h_k_at_t = hazards[:, k, :].gather(1, time_idx.unsqueeze(1)).squeeze(1)
            log_h = torch.log(h_k_at_t + 1e-10)

            # Survival at t-1
            survival_at_tm1 = torch.where(
                time_idx > 0,
                survival.gather(1, (time_idx - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
                torch.ones(B, device=device),
            )
            log_s = torch.log(survival_at_tm1 + 1e-10)

            # Event contribution
            event_loss = -event_mask * (log_h + log_s)

            # Censored contribution: log[S(t_i)]
            survival_at_t = survival.gather(1, time_idx.unsqueeze(1)).squeeze(1)
            censor_loss = -censor_mask * torch.log(survival_at_t + 1e-10) / hazards.size(1)

            nll = nll + (event_loss + censor_loss).mean()

        # 2. Concordance regularization (ranking loss)
        concordance_loss = self._concordance_loss(cif, durations, event_types)

        # Total loss
        total = nll + self.alpha_c * concordance_loss

        return {
            "total": total,
            "nll": nll.detach(),
            "concordance_loss": concordance_loss.detach(),
        }

    def _concordance_loss(self, cif, durations, event_types, n_pairs=256):
        """Differentiable concordance index approximation

        For comparable pairs, the model should assign higher CIF
        to subjects who experienced the event earlier.
        """
        device = cif.device
        B = durations.size(0)

        if B < 4:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)

        for k in range(cif.size(1)):
            event_mask = (event_types == (k + 1))
            if event_mask.sum() < 2:
                continue

            event_idx = event_mask.nonzero(as_tuple=True)[0]
            n = min(n_pairs, len(event_idx) * (B - len(event_idx)))
            if n == 0:
                continue

            # Sample pairs
            i_idx = event_idx[torch.randint(len(event_idx), (n,), device=device)]
            j_idx = torch.randint(B, (n,), device=device)

            # Comparable: i had event, j had later event or was censored
            comparable = durations[j_idx] > durations[i_idx]
            if comparable.sum() == 0:
                continue

            i_idx = i_idx[comparable]
            j_idx = j_idx[comparable]

            # CIF at time of event for i
            t_i = (durations[i_idx] - 1).clamp(0, cif.size(2) - 1)
            cif_i = cif[i_idx, k, :].gather(1, t_i.unsqueeze(1)).squeeze(1)
            cif_j = cif[j_idx, k, :].gather(1, t_i.unsqueeze(1)).squeeze(1)

            # i should have higher CIF (event happened)
            loss = loss + F.softplus(cif_j - cif_i + 0.1).mean()

        return loss


# ============================================================
# 5. Training & Inference Engine
# ============================================================

class GATSurvivalTrainer:
    """Training loop for GAT-Survival model"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2,
        )
        self.loss_fn = CompetingRiskLoss()
        self.best_c_index = 0.0
        self.patience_counter = 0
        self.best_state = None

    def _get_mask(self, data: Dict, primary_key: str, fallback_keys: Optional[List[str]] = None):
        """Safely get boolean mask from data dict."""
        fallback_keys = fallback_keys or []

        if primary_key in data and data[primary_key] is not None:
            return data[primary_key]

        for key in fallback_keys:
            if key in data and data[key] is not None:
                return data[key]

        return None

    def train_epoch(self, train_data: Dict) -> Dict:
        """Train one epoch

        train_data contains:
          - x: [N, F] node features
          - edge_index: [2, E]
          - edge_type: [E] (optional)
          - edge_weight: [E] (optional)
          - durations: [N]
          - event_types: [N]
          - train_mask or mask: [N] boolean mask (optional)
        """
        self.model.train()

        x = train_data["x"].to(self.device)
        edge_index = train_data["edge_index"].to(self.device)

        edge_type = train_data.get("edge_type")
        edge_weight = train_data.get("edge_weight")
        if edge_type is not None:
            edge_type = edge_type.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        durations = train_data["durations"].to(self.device)
        event_types = train_data["event_types"].to(self.device)

        train_mask = self._get_mask(train_data, "train_mask", ["mask"])
        if train_mask is None:
            train_mask = torch.ones(x.size(0), dtype=torch.bool, device=self.device)
        else:
            train_mask = train_mask.to(self.device).bool()

        # Safety checks
        if durations.size(0) != x.size(0):
            raise ValueError(
                f"durations length mismatch: durations={durations.size(0)}, x={x.size(0)}"
            )
        if event_types.size(0) != x.size(0):
            raise ValueError(
                f"event_types length mismatch: event_types={event_types.size(0)}, x={x.size(0)}"
            )
        if train_mask.size(0) != x.size(0):
            raise ValueError(
                f"train_mask length mismatch: train_mask={train_mask.size(0)}, x={x.size(0)}"
            )
        if train_mask.sum().item() == 0:
            raise ValueError("train_mask has no True values.")

        # Forward
        output = self.model(x, edge_index, edge_type, edge_weight)

        # Compute loss only on train nodes
        loss_dict = self.loss_fn(
            output["hazards"][train_mask],
            output["survival"][train_mask],
            output["cif"][train_mask],
            durations[train_mask],
            event_types[train_mask],
        )

        # Backward
        self.optimizer.zero_grad()
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            k: (v.item() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
        }

    @torch.no_grad()
    def evaluate(self, eval_data: Optional[Dict]) -> Optional[Dict]:
        """Evaluate on validation/test set.
        
        Returns None if eval_data is None.
        """
        if eval_data is None:
            return None

        self.model.eval()

        x = eval_data["x"].to(self.device)
        edge_index = eval_data["edge_index"].to(self.device)

        edge_type = eval_data.get("edge_type")
        edge_weight = eval_data.get("edge_weight")
        if edge_type is not None:
            edge_type = edge_type.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        durations = eval_data["durations"].to(self.device)
        event_types = eval_data["event_types"].to(self.device)

        eval_mask = self._get_mask(eval_data, "eval_mask", ["val_mask", "mask"])
        if eval_mask is None:
            eval_mask = torch.ones(x.size(0), dtype=torch.bool, device=self.device)
        else:
            eval_mask = eval_mask.to(self.device).bool()

        # Safety checks
        if durations.size(0) != x.size(0):
            raise ValueError(
                f"durations length mismatch: durations={durations.size(0)}, x={x.size(0)}"
            )
        if event_types.size(0) != x.size(0):
            raise ValueError(
                f"event_types length mismatch: event_types={event_types.size(0)}, x={x.size(0)}"
            )
        if eval_mask.size(0) != x.size(0):
            raise ValueError(
                f"eval_mask length mismatch: eval_mask={eval_mask.size(0)}, x={x.size(0)}"
            )
        if eval_mask.sum().item() == 0:
            raise ValueError("eval_mask has no True values.")

        output = self.model(x, edge_index, edge_type, edge_weight)

        # Loss
        loss_dict = self.loss_fn(
            output["hazards"][eval_mask],
            output["survival"][eval_mask],
            output["cif"][eval_mask],
            durations[eval_mask],
            event_types[eval_mask],
        )

        # C-index (cause-specific)
        c_indices = {}
        num_causes = output["cif"].size(1)
        cause_names = ["profit", "loss"]

        for k in range(num_causes):
            cause_name = cause_names[k] if k < len(cause_names) else f"risk_{k+1}"
            c_idx = self._compute_c_index(
                output["cif"][eval_mask, k, -1],
                durations[eval_mask],
                event_types[eval_mask],
                event_of_interest=k + 1,
            )
            c_indices[f"c_index_{cause_name}"] = c_idx

        signals = output["signals"]
        eval_signals = {
            k: v[eval_mask].detach().cpu().numpy()
            for k, v in signals.items()
            if torch.is_tensor(v) and v.dim() <= 1
        }

        return {
            **{
                k: (v.item() if torch.is_tensor(v) else v)
                for k, v in loss_dict.items()
            },
            **c_indices,
            **{
                f"mean_{k}": float(v.mean())
                for k, v in eval_signals.items()
                if isinstance(v, np.ndarray)
            },
        }

    def _compute_c_index(self, risk_scores, durations, event_types, event_of_interest=1):
        """Compute cause-specific concordance index"""
        risk = risk_scores.detach().cpu().numpy()
        dur = durations.detach().cpu().numpy()
        evt = event_types.detach().cpu().numpy()

        concordant, discordant, tied = 0, 0, 0
        event_mask = evt == event_of_interest

        for i in range(len(risk)):
            if not event_mask[i]:
                continue
            for j in range(len(risk)):
                if i == j:
                    continue
                if dur[j] <= dur[i]:
                    continue

                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
                else:
                    tied += 1

        total_pairs = concordant + discordant + tied
        if total_pairs == 0:
            return 0.5

        return (concordant + 0.5 * tied) / total_pairs

    def fit(self, train_data, val_data=None, epochs=100, patience=15):
        """Full training loop with optional early stopping"""
        logger.info(f"Training GAT-Survival model for up to {epochs} epochs...")

        best_val_loss = float("inf")
        history = []
        self.patience_counter = 0

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_data)
            val_metrics = self.evaluate(val_data) if val_data is not None else None

            epoch_record = {
                "epoch": epoch,
                **train_metrics,
            }

            if val_metrics is not None:
                epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(epoch_record)

            if epoch % 5 == 0:
                if val_metrics is not None:
                    logger.info(
                        f"Epoch {epoch:3d} | "
                        f"Train NLL: {train_metrics.get('nll', float('nan')):.4f} | "
                        f"Val NLL: {val_metrics.get('nll', float('nan')):.4f} | "
                        f"C-idx(profit): {val_metrics.get('c_index_profit', 0):.4f} | "
                        f"C-idx(loss): {val_metrics.get('c_index_loss', 0):.4f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch:3d} | "
                        f"Train NLL: {train_metrics.get('nll', float('nan')):.4f} | "
                        f"Train Total: {train_metrics.get('total', float('nan')):.4f}"
                    )

            # Early stopping only when validation exists
            if val_metrics is not None:
                val_loss = val_metrics["total"]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            else:
                # No validation set: keep last state as best_state
                self.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

        # Restore best model only if validation-based best state exists
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return history

    @torch.no_grad()
    def predict(self, data: Dict) -> Dict:
        """Generate trading signals for all stocks"""
        self.model.eval()

        x = data["x"].to(self.device)
        edge_index = data["edge_index"].to(self.device)
        edge_type = data.get("edge_type")
        edge_weight = data.get("edge_weight")

        if edge_type is not None:
            edge_type = edge_type.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        output = self.model(x, edge_index, edge_type, edge_weight)

        signals = {}
        for k, v in output["signals"].items():
            if torch.is_tensor(v):
                signals[k] = v.detach().cpu().numpy()

        return {
            "signals": signals,
            "embeddings": output["embeddings"].detach().cpu().numpy(),
            "survival_curves": output["survival"].detach().cpu().numpy(),
            "profit_cif": output["cif"][:, 0, :].detach().cpu().numpy(),
            "loss_cif": output["cif"][:, 1, :].detach().cpu().numpy(),
        }


# ============================================================
# 6. Signal Ranker (Multi-Market)
# ============================================================

class MultiMarketSignalRanker:
    """Rank stocks across US + KOSPI markets

    Combines model signals with market regime context
    to produce final entry recommendations with expected returns.
    """

    def __init__(self, config):
        self.config = config
        self.target_return = config.get("target_return", 0.10)
        self.stop_loss = config.get("stop_loss", -0.05)

    def rank(
        self,
        predictions: Dict,
        tickers: List[str],
        market_features: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Produce ranked stock recommendations

        Returns DataFrame with columns:
          ticker, market, entry_score, expected_return, profit_prob,
          loss_prob, median_time, risk_adjusted_score, signal, rank
        """
        signals = predictions["signals"]
        n = len(tickers)

        records = []
        for i, ticker in enumerate(tickers):
            market = "KR" if ticker.endswith((".KS", ".KQ")) else "US"

            record = {
                "ticker": ticker,
                "market": market,
                "entry_score": float(signals["entry_score"][i]),
                "expected_return": float(signals["expected_return"][i]),
                "profit_prob": float(signals["profit_prob"][i]),
                "loss_prob": float(signals["loss_prob"][i]),
                "censor_prob": float(signals["censor_prob"][i]),
                "median_survival_time": float(signals["median_survival_time"][i]),
                "risk_adjusted_score": float(signals["risk_adjusted_score"][i]),
            }

            # Entry price recommendation
            # If entry_score > threshold, recommend entry at current price
            record["entry_price"] = "current"  # placeholder

            # Signal classification
            score = record["risk_adjusted_score"]
            if score > 0.5 and record["profit_prob"] > 0.5:
                record["signal"] = "STRONG_BUY"
            elif score > 0.2 and record["profit_prob"] > 0.4:
                record["signal"] = "BUY"
            elif score < -0.3 or record["loss_prob"] > 0.6:
                record["signal"] = "AVOID"
            else:
                record["signal"] = "HOLD"

            records.append(record)

        df = pd.DataFrame(records)

        # Market regime adjustment
        if market_features:
            df = self._apply_regime_adjustment(df, market_features)

        # Rank within each market
        for market in df["market"].unique():
            mask = df["market"] == market
            df.loc[mask, "rank"] = df.loc[mask, "risk_adjusted_score"].rank(
                ascending=False
            ).astype(int)

        # Global rank
        df["global_rank"] = df["risk_adjusted_score"].rank(ascending=False).astype(int)

        return df.sort_values("global_rank")

    def _apply_regime_adjustment(self, df: pd.DataFrame, market_features: Dict) -> pd.DataFrame:
        """Adjust scores based on market regime

        In bear markets: increase loss_prob weight, be more conservative
        In bull markets: increase profit_prob weight, be more aggressive
        """
        regime = market_features.get("market_regime", 1)  # 0=bear, 1=neutral, 2=bull
        vix = market_features.get("vix_level", 20)

        if regime == 0:  # Bear
            df["expected_return"] *= 0.7
            df["risk_adjusted_score"] *= 0.8
        elif regime == 2:  # Bull
            df["expected_return"] *= 1.1
            df["risk_adjusted_score"] *= 1.1

        # VIX penalty
        if vix > 30:
            df["risk_adjusted_score"] *= 0.7
        elif vix > 25:
            df["risk_adjusted_score"] *= 0.85

        return df

    def generate_report_data(self, ranked_df: pd.DataFrame, top_n: int = 10) -> Dict:
        """Generate data for investment report"""
        top_buys = ranked_df[ranked_df["signal"].isin(["STRONG_BUY", "BUY"])].head(top_n)
        top_avoids = ranked_df[ranked_df["signal"] == "AVOID"].head(5)

        us_picks = top_buys[top_buys["market"] == "US"]
        kr_picks = top_buys[top_buys["market"] == "KR"]

        return {
            "top_picks": top_buys.to_dict("records"),
            "us_picks": us_picks.to_dict("records"),
            "kr_picks": kr_picks.to_dict("records"),
            "avoid_list": top_avoids.to_dict("records"),
            "summary": {
                "total_buy_signals": len(ranked_df[ranked_df["signal"].isin(["STRONG_BUY", "BUY"])]),
                "total_avoid_signals": len(ranked_df[ranked_df["signal"] == "AVOID"]),
                "avg_expected_return": float(top_buys["expected_return"].mean()) if len(top_buys) > 0 else 0,
                "avg_profit_prob": float(top_buys["profit_prob"].mean()) if len(top_buys) > 0 else 0,
            },
        }
