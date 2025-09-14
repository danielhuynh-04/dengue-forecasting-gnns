# trainers/trainer_weekly.py
# ------------------------------------------------------------
# Dựng chuỗi thời gian từ *_lagK và tạo model TemporalGAT.
# ------------------------------------------------------------
from __future__ import annotations
import torch
from typing import List, Tuple
from models.temporal_gat import TemporalGAT


def _find_lag_columns(feature_names: List[str], allowed_prefixes=None) -> List[str]:
    if allowed_prefixes is None:
        allowed_prefixes = ("incidence_", "casos_", "precip", "temp", "rel_humid")
    lag_cols = []
    for c in feature_names or []:
        if "_lag" in c and any(c.startswith(p) for p in allowed_prefixes):
            lag_cols.append(c)

    def _lag_k(name: str) -> int:
        try:
            return int(name.split("_lag")[-1])
        except Exception:
            return 10**9

    lag_cols.sort(key=_lag_k)
    return lag_cols


def build_temporal_seq(x: torch.Tensor, feature_names: List[str]) -> Tuple[torch.Tensor | None, int]:
    if not feature_names:
        return None, 0
    lag_cols = _find_lag_columns(feature_names)
    if not lag_cols:
        return None, 0
    name2idx = {n: i for i, n in enumerate(feature_names)}
    col_idx = [name2idx[c] for c in lag_cols]
    seq = x[:, col_idx].unsqueeze(-1)  # [N,T,1]
    return seq, 1


def make_model(
    in_dim: int,
    has_temporal: bool,
    hidden: int = 128,
    heads: tuple[int, int] = (4, 4),
    gat_dropout: float = 0.2,
) -> TemporalGAT:
    t_in_dim = 1 if has_temporal else 0
    return TemporalGAT(
        in_dim=in_dim,
        hidden=hidden,
        out_dim=1,
        heads1=heads[0],
        heads2=heads[1],
        gat_dropout=gat_dropout,
        t_in_dim=t_in_dim,
        t_hidden=64,
        t_heads=4,
        t_dropout=0.1,
    )
