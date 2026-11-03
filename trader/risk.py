"""Gestión de riesgo: tamaño de posición, stops, take profit, trailing y kill switch."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_ts: str
    stop: float
    take_profit: float
    entry_fee: float = 0.0
    last_checked: str = ""
    bars_held: int = 0


class RiskManager:
    def __init__(self, rcfg: dict, fee_rate: float = 0.001):
        self.r = rcfg
        self.fee_rate = fee_rate

    def size(self, equity: float, cash: float, price: float, atr_value: float):
        """Devuelve (qty, stop, take_profit). Riesgo fijo % del capital hasta el stop."""
        stop_dist = self.r["atr_stop_mult"] * atr_value
        if stop_dist <= 0 or price <= 0 or stop_dist >= price:
            return 0.0, 0.0, 0.0
        qty = equity * self.r["risk_per_trade"] / stop_dist
        qty = min(qty, equity * self.r["max_position_pct"] / price, cash * 0.995 / (price * (1 + self.fee_rate)))
        if qty * price < self.r.get("min_notional", 10):
            return 0.0, 0.0, 0.0
        return qty, price - stop_dist, price + self.r["reward_risk"] * stop_dist

    def can_open(self, n_open: int, day_start_equity: float | None, equity: float) -> tuple[bool, str]:
        if n_open >= self.r["max_open_positions"]:
            return False, "max_open_positions"
        if day_start_equity and equity < day_start_equity * (1 - self.r["max_daily_loss_pct"]):
            return False, "kill_switch_daily_loss"
        return True, ""

    def check_bar(self, pos: Position, o: float, h: float, l: float, c: float, atr_value: float):
        """Evalúa una vela cerrada sobre una posición abierta.
        Devuelve (precio_salida, motivo) o None. Si stop y TP caen en la misma vela, se asume
        el stop (supuesto conservador). Gaps: se rellena al precio de apertura."""
        pos.bars_held += 1
        if l <= pos.stop:
            return min(pos.stop, o), "stop_loss"
        if h >= pos.take_profit:
            return max(pos.take_profit, o), "take_profit"
        if self.r.get("trailing_stop", True):
            pos.stop = max(pos.stop, c - self.r["atr_stop_mult"] * atr_value)
        if pos.bars_held >= self.r["max_holding_bars"]:
            return c, "time_exit"
        return None
