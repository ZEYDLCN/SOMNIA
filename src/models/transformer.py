"""
Temporal Transformer — SOMNIA (Kişisel Uyku Kalitesi için Temporal Transformer)

Adım 4 (bkz. docs/plan.md §4): "gün × özellik" tokenlamalı, iki eksenli
(opsiyonel) self-attention mimarisi, sıfırdan PyTorch ile:

    Girdi: (B, T, F)  — T=window_size gün, F=özellik sayısı
        -> her (gün, özellik) hücresi kendi linear projeksiyonuyla
           d_model boyutuna taşınır (feature embedding)
        -> özellik ekseninde self-attention (TFT tarzı, hangi özelliğin
           o gün için önemli olduğunu öğrenir) + havuzlama -> gün token'ı
        -> döngüsel/öğrenilen zaman encoding eklenir
        -> zaman ekseninde self-attention (N katman) — hangi geçmiş
           günün önemli olduğunu öğrenir
        -> regresyon başlığı -> sleep_quality(t) tahmini

Model, hem özellik-ekseni hem zaman-ekseni attention ağırlıklarını
`return_attn=True` ile döndürür; bunlar adım 6'daki yorumlanabilirlik
modülünde (attention haritaları) kullanılacak. Attention ağırlıklarının
kendisinin nedensellik kanıtı OLMADIĞI unutulmamalı (bkz. docs/plan.md §5) —
burada sadece modelin neye "baktığını" gözlemlemek için saklanıyor.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn


@dataclasses.dataclass
class TransformerConfig:
    n_features: int
    window_size: int
    d_model: int = 32
    n_heads: int = 4
    n_temporal_layers: int = 2
    d_ff: int = 64
    dropout: float = 0.2


class SinusoidalTimeEncoding(nn.Module):
    """Standart sinüsoidal pozisyon encoding'i, T (window) uzunluğu için.
    day_of_week zaten döngüsel özellik olarak girdi feature'larında var
    (dow_sin/dow_cos); bu, ayrıca dizideki GÖRECELİ konumu (t-N .. t-1)
    kodlar."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class FeatureAxisAttention(nn.Module):
    """Her gün için, o günün F özelliği arasında self-attention uygular
    ve ardından ağırlıklı ortalama ile tek bir 'gün token'ı üretir
    (TFT'deki variable-selection network'e benzer, basitleştirilmiş)."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.d_model, num_heads=cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.pool_query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B*T, F, d_model)
        bt = x.shape[0]
        query = self.pool_query.expand(bt, -1, -1)  # (B*T, 1, d_model)
        pooled, attn_weights = self.attn(query, x, x, need_weights=True)
        # pooled: (B*T, 1, d_model), attn_weights: (B*T, 1, F)
        pooled = self.norm(pooled.squeeze(1))
        return pooled, attn_weights.squeeze(1)  # (B*T, F)


class TemporalEncoderLayer(nn.Module):
    """Standart pre-norm Transformer encoder katmanı; attention
    ağırlıklarını dışarı verir (interpretability için)."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.d_model, num_heads=cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d_model)
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=True)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x, attn_weights  # attn_weights: (B, T, T)


class TemporalSleepTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        # Her özellik kendi skaler->d_model projeksiyonuna sahip
        # (plan.md §4: "her özellik kendi linear projeksiyonuyla").
        self.feature_embed = nn.Linear(1, cfg.d_model)
        self.feature_id_embed = nn.Embedding(cfg.n_features, cfg.d_model)

        self.feature_attn = FeatureAxisAttention(cfg)
        self.time_encoding = SinusoidalTimeEncoding(cfg.d_model, max_len=cfg.window_size + 1)

        self.temporal_layers = nn.ModuleList(
            [TemporalEncoderLayer(cfg) for _ in range(cfg.n_temporal_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

        self.temporal_pool_query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.temporal_pool_attn = nn.MultiheadAttention(
            cfg.d_model, num_heads=cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # x: (B, T, F)
        B, T, F = x.shape
        assert F == self.cfg.n_features and T == self.cfg.window_size

        # 1) Feature embedding: her skaler değeri d_model'e taşı, ve
        #    hangi özellik olduğunu bir "feature-id" embedding'i ile belirt.
        tokens = self.feature_embed(x.unsqueeze(-1))  # (B, T, F, d_model)
        feature_ids = torch.arange(F, device=x.device)
        tokens = tokens + self.feature_id_embed(feature_ids)  # broadcast

        # 2) Özellik ekseninde self-attention + havuzlama -> gün token'ı
        tokens_flat = tokens.reshape(B * T, F, self.cfg.d_model)
        day_tokens, feat_attn = self.feature_attn(tokens_flat)
        day_tokens = day_tokens.reshape(B, T, self.cfg.d_model)  # (B, T, d_model)
        feat_attn = feat_attn.reshape(B, T, F)  # her gün için özellik ağırlıkları

        # 3) Zaman encoding'i ekle
        day_tokens = self.time_encoding(day_tokens)

        # 4) Zaman ekseninde self-attention (N katman)
        temporal_attns = []
        h = day_tokens
        for layer in self.temporal_layers:
            h, attn_w = layer(h)
            temporal_attns.append(attn_w)  # (B, T, T)
        h = self.final_norm(h)

        # 5) Dizi boyunca havuzlama (öğrenilen query ile attention pooling)
        query = self.temporal_pool_query.expand(B, -1, -1)
        pooled, pool_attn = self.temporal_pool_attn(query, h, h, need_weights=True)
        pooled = pooled.squeeze(1)  # (B, d_model)
        pool_attn = pool_attn.squeeze(1)  # (B, T) — hangi günün ne kadar ağırlık aldığı

        pred = self.head(pooled).squeeze(-1)  # (B,)

        if not return_attn:
            return pred

        return pred, {
            "feature_attn": feat_attn,  # (B, T, F): gün başına özellik önemi
            "temporal_layer_attn": temporal_attns,  # her katman: (B, T, T)
            "temporal_pool_attn": pool_attn,  # (B, T): son tahmine gün katkısı
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
