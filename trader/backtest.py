"""Backtest walk-forward: el modelo se reentrena periódicamente usando SOLO datos pasados
y se simula con las mismas reglas de estrategia y riesgo que el bot en vivo."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .data import tf_seconds
from .features import WARMUP, atr, build_features
from .model import parallel_map, train_model
from .risk import Position, RiskManager
from .strategy import decide

log = logging.getLogger(__name__)


def walk_forward_predictions(symbol: str, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b = cfg["backtest"]
    X_all = build_features(df)
    start, step = WARMUP + int(b["min_train_bars"]), int(b["retrain_every"])
    parts = []
    for i in range(start, len(df), step):
        m = train_model(symbol, df.iloc[:i], cfg)  # sólo pasado
        p = m.predict_frame(X_all.iloc[i:i + step])
        p["tradable"] = m.metrics["tradable"]
        parts.append(p)
    if not parts:
        raise ValueError(f"{symbol}: historia insuficiente para backtest")
    return pd.concat(parts)


def simulate(df: pd.DataFrame, preds: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    fee, slip = cfg["execution"]["fee_rate"], cfg["execution"]["slippage"]
    risk = RiskManager(cfg["risk"], fee)
    a = atr(df)
    cash = float(cfg["risk"]["starting_equity"])
    pos: Position | None = None
    curve, trades = [], []
    for t, p in preds.iterrows():
        o, h, l, c = df.loc[t, ["open", "high", "low", "close"]]
        atr_v = float(a.loc[t])
        if pos is not None:
            ex = risk.check_bar(pos, o, h, l, c, atr_v)
            if ex is None and decide(p.prob_up, p.q50, p.tradable, cfg["strategy"]) == "EXIT":
                ex = (c, "model_exit")
            if ex:
                px = ex[0] * (1 - slip)
                fee_x = pos.qty * px * fee
                cash += pos.qty * px - fee_x
                trades.append({"entry_time": pos.entry_ts, "exit_time": t, "entry": pos.entry_price, "exit": px,
                               "pnl": (px - pos.entry_price) * pos.qty - fee_x - pos.entry_fee, "reason": ex[1],
                               "bars": pos.bars_held})
                pos = None
        elif decide(p.prob_up, p.q50, p.tradable, cfg["strategy"]) == "BUY":
            qty, stop, tp = risk.size(cash, cash, c, atr_v)
            if qty > 0:
                px = c * (1 + slip)
                fee_e = qty * px * fee
                cash -= qty * px + fee_e
                pos = Position("", qty, px, t, stop, tp, fee_e)
        curve.append(cash + (pos.qty * c if pos else 0.0))
    return pd.Series(curve, index=preds.index, name="equity"), pd.DataFrame(trades)


def metrics(equity: pd.Series, trades: pd.DataFrame, df: pd.DataFrame, timeframe: str) -> dict:
    rets = equity.pct_change().dropna()
    per_year = 365 * 86400 / tf_seconds(timeframe)
    dd = (equity / equity.cummax() - 1).min()
    px = df["close"].reindex(equity.index)
    wins = trades[trades["pnl"] > 0] if len(trades) else trades
    losses = trades[trades["pnl"] <= 0] if len(trades) else trades
    return {
        "total_return": round(float(equity.iloc[-1] / equity.iloc[0] - 1), 4),
        "buy_hold": round(float(px.iloc[-1] / px.iloc[0] - 1), 4),
        "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(per_year)), 2) if rets.std() > 0 else 0.0,
        "max_drawdown": round(float(dd), 4),
        "trades": int(len(trades)),
        "win_rate": round(float(len(wins) / len(trades)), 3) if len(trades) else None,
        "profit_factor": round(float(wins["pnl"].sum() / -losses["pnl"].sum()), 2) if len(losses) and losses["pnl"].sum() < 0 else None,
    }


def backtest_symbol(symbol: str, df: pd.DataFrame, cfg: dict) -> dict:
    preds = walk_forward_predictions(symbol, df, cfg)
    eq, trades = simulate(df, preds, cfg)
    return {"symbol": symbol, **metrics(eq, trades, df, cfg["data"]["timeframe"]), "equity": eq, "trade_log": trades}


def _bt_job(args):
    return backtest_symbol(*args)


def backtest_many(frames: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, dict]:
    results = {}
    items = [(s, df, cfg) for s, df in frames.items()]
    for s, r in parallel_map(_bt_job, items, int(cfg["universe"].get("model_workers", 4)), keys=list(frames)):
        if isinstance(r, Exception):
            log.warning("%s: backtest falló: %s", s, r)
        else:
            results[s] = r
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("equity", "trade_log")} for r in results.values()])
    if not table.empty:
        table = table.sort_values("total_return", ascending=False).reset_index(drop=True)
    return table, results
