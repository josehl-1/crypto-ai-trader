"""Escáner multi-símbolo: entrena/carga modelos en paralelo y rankea oportunidades."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .data import tf_seconds
from .features import build_features
from .model import ModelRegistry
from .strategy import decide, score

log = logging.getLogger(__name__)


class Scanner:
    def __init__(self, cfg: dict, registry: ModelRegistry | None = None):
        self.cfg = cfg
        self.registry = registry or ModelRegistry(cfg)

    def analyze(self, symbol: str, df: pd.DataFrame, source: str = "") -> dict:
        model = self.registry.get(symbol, df)
        feats = build_features(df).iloc[[-1]]
        p = model.predict_frame(feats).iloc[0]
        price = float(df["close"].iloc[-1])
        bars_day = max(1, int(86400 / tf_seconds(self.cfg["data"]["timeframe"])))
        qvol = float((df["close"] * df["volume"]).tail(bars_day).sum())
        liquid = qvol >= self.cfg["universe"].get("min_quote_volume_24h", 0)
        tradable = bool(model.metrics["tradable"] and liquid)
        sig = decide(p["prob_up"], p["q50"], tradable, self.cfg["strategy"])
        return {
            "symbol": symbol, "bar_time": str(df.index[-1]), "price": price,
            "prob_up": round(float(p["prob_up"]), 4),
            "q10": float(p["q10"]), "q50": float(p["q50"]), "q90": float(p["q90"]),
            "target_price": price * float(np.exp(p["q50"])),
            "atr_pct": float(feats["atr_pct"].iloc[0]),
            "change_24h": float(df["close"].iloc[-1] / df["close"].iloc[-1 - bars_day] - 1) if len(df) > bars_day else 0.0,
            "quote_volume_24h": qvol,
            "auc": model.metrics["auc"], "hit_rate": model.metrics["hit_rate"],
            "holdout_edge": model.metrics["holdout_edge"],
            "tradable": tradable, "signal": sig,
            "score": score(p["prob_up"], p["q10"], p["q50"], p["q90"]),
            "source": source,
        }

    def scan(self, frames: dict[str, pd.DataFrame], sources: dict[str, str] | None = None) -> pd.DataFrame:
        sources = sources or {}
        failed = self.registry.train_many(frames, int(self.cfg["universe"].get("model_workers", 4)))
        rows = []
        for s, df in frames.items():
            if s in failed:
                continue
            try:
                rows.append(self.analyze(s, df, sources.get(s, "")))
            except Exception as e:  # noqa: BLE001
                log.warning("%s: error en análisis: %s", s, e)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
