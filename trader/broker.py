"""Ejecución de órdenes: PaperBroker (simulado) y LiveBroker (real vía ccxt)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import ccxt

from .storage import Storage

log = logging.getLogger(__name__)


@dataclass
class Fill:
    qty: float
    price: float
    fee: float


class PaperBroker:
    mode = "paper"

    def __init__(self, cfg: dict, storage: Storage):
        self.fee = cfg["execution"]["fee_rate"]
        self.slip = cfg["execution"]["slippage"]
        self.storage = storage
        if storage.get_state("paper_cash") is None:
            storage.set_state("paper_cash", cfg["risk"]["starting_equity"])

    def cash(self) -> float:
        return float(self.storage.get_state("paper_cash"))

    def _set_cash(self, v):
        self.storage.set_state("paper_cash", v)

    def buy(self, symbol: str, qty: float, ref_price: float) -> Fill:
        price = ref_price * (1 + self.slip)
        qty = min(qty, self.cash() / (price * (1 + self.fee)))
        cost = qty * price
        fee = cost * self.fee
        self._set_cash(self.cash() - cost - fee)
        return Fill(qty, price, fee)

    def sell(self, symbol: str, qty: float, ref_price: float) -> Fill:
        price = ref_price * (1 - self.slip)
        proceeds = qty * price
        fee = proceeds * self.fee
        self._set_cash(self.cash() + proceeds - fee)
        return Fill(qty, price, fee)

    def protect(self, pos):
        """En simulación los stops solo los vigila el bot."""

    def stop_fill(self, pos):
        return None


class LiveBroker:
    """Órdenes de mercado reales. Requiere:
       EXCHANGE_API_KEY, EXCHANGE_API_SECRET (y EXCHANGE_API_PASSWORD si el exchange lo pide)
       CONFIRM_LIVE_TRADING=YES  (salvaguarda explícita)
       execution.max_live_capital > 0 (capital máximo que el bot puede usar)"""
    mode = "live"

    def __init__(self, cfg: dict, storage: Storage):
        if os.environ.get("CONFIRM_LIVE_TRADING") != "YES":
            raise RuntimeError("Modo live bloqueado: exporta CONFIRM_LIVE_TRADING=YES para habilitarlo.")
        if cfg["data"]["source"] != "exchange":
            raise RuntimeError("Modo live bloqueado: no se puede operar con datos sintéticos.")
        self.max_capital = float(cfg["execution"].get("max_live_capital") or 0)
        if self.max_capital <= 0:
            raise RuntimeError("Modo live bloqueado: define execution.max_live_capital en el config.")
        self.storage = storage
        key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
        if not key or not secret:
            raise RuntimeError("Faltan EXCHANGE_API_KEY / EXCHANGE_API_SECRET.")
        params = {"apiKey": key, "secret": secret, "enableRateLimit": True, "options": {"defaultType": "spot"}}
        if os.environ.get("EXCHANGE_API_PASSWORD"):
            params["password"] = os.environ["EXCHANGE_API_PASSWORD"]
        self.ex = getattr(ccxt, cfg["exchange"]["id"])(params)
        if cfg["exchange"].get("sandbox"):
            self.ex.set_sandbox_mode(True)
        self.ex.load_markets()
        self.quote = cfg["exchange"]["quote"]
        self.fee = cfg["execution"]["fee_rate"]
        self.exchange_stops = cfg["execution"].get("exchange_stops", True)
        self.stop_buffer = float(cfg["execution"].get("stop_limit_buffer", 0.01))
        if self.exchange_stops and not self.ex.has.get("createStopLossOrder"):
            raise RuntimeError(f"Modo live bloqueado: {self.ex.id} no admite órdenes de stop vía ccxt.")
        log.warning("LiveBroker activo en %s (sandbox=%s)", self.ex.id, cfg["exchange"].get("sandbox"))

    def cash(self) -> float:
        """Saldo libre, limitado a lo que queda de max_live_capital (las pérdidas lo reducen)."""
        free = float(self.ex.fetch_balance()["free"].get(self.quote, 0) or 0)
        budget = self.max_capital + self.storage.realized_pnl() - self.storage.deployed_capital()
        return max(0.0, min(free, budget))

    def _fill(self, order, amount, ref_price) -> Fill:
        filled = float(order.get("filled") or amount)
        avg = float(order.get("average") or order.get("price") or ref_price)
        fee_info = order.get("fee") or {}
        fee = float(fee_info["cost"]) if fee_info.get("cost") is not None and fee_info.get("currency") == self.quote \
            else filled * avg * self.fee
        return Fill(filled, avg, fee)

    def buy(self, symbol: str, qty: float, ref_price: float) -> Fill:
        qty = min(qty, self.cash() / (ref_price * (1 + self.fee)))
        if qty <= 0:
            return Fill(0.0, ref_price, 0.0)
        amount = float(self.ex.amount_to_precision(symbol, qty))
        order = self.ex.create_order(symbol, "market", "buy", amount)
        log.info("LIVE BUY %s %s -> %s", symbol, amount, order.get("id"))
        return self._fill(order, amount, ref_price)

    def sell(self, symbol: str, qty: float, ref_price: float) -> Fill:
        # si cancelar falla no se vende: el stop pudo ejecutarse y stop_fill lo registrará
        partial = self._release_stop(symbol, ref_price)
        sold = partial.qty if partial else 0.0
        amount = float(self.ex.amount_to_precision(symbol, min(qty - sold, self._free_base(symbol))))
        order = self.ex.create_order(symbol, "market", "sell", amount)
        log.info("LIVE SELL %s %s -> %s", symbol, amount, order.get("id"))
        fill = self._fill(order, amount, ref_price)
        if partial:
            total = partial.qty + fill.qty
            price = (partial.qty * partial.price + fill.qty * fill.price) / total
            return Fill(total, price, partial.fee + fill.fee)
        return fill

    def _free_base(self, symbol: str) -> float:
        base = self.ex.market(symbol)["base"]
        return float(self.ex.fetch_balance()["free"].get(base, 0) or 0)

    # ---------------- stop en el exchange ----------------
    def _stop_order(self, symbol: str) -> tuple[str | None, float]:
        raw = self.storage.get_state(f"stop_order:{symbol}")
        if not raw:
            return None, 0.0
        order_id, trigger = raw.split("|")
        return order_id, float(trigger)

    def protect(self, pos):
        """Mantiene en el exchange un stop-limit de venta en pos.stop; solo se mueve hacia arriba."""
        if not self.exchange_stops:
            return
        symbol = pos.symbol
        order_id, current = self._stop_order(symbol)
        trigger = float(self.ex.price_to_precision(symbol, pos.stop))
        if order_id:
            if trigger <= current or float(self.ex.fetch_order(order_id, symbol).get("filled") or 0) > 0:
                return
            self._release_stop(symbol, pos.stop)
        amount = float(self.ex.amount_to_precision(symbol, min(pos.qty, self._free_base(symbol))))
        limit = float(self.ex.price_to_precision(symbol, trigger * (1 - self.stop_buffer)))
        order = self.ex.create_order(symbol, "limit", "sell", amount, limit, {"stopLossPrice": trigger})
        self.storage.set_state(f"stop_order:{symbol}", f"{order['id']}|{trigger}")
        log.info("LIVE STOP %s %s @ %s (límite %s) -> %s", symbol, amount, trigger, limit, order["id"])

    def stop_fill(self, pos) -> Fill | None:
        """Devuelve el Fill si el stop se ejecutó en el exchange; si ya no está activo, lo olvida."""
        order_id, _ = self._stop_order(pos.symbol)
        if not order_id:
            return None
        order = self.ex.fetch_order(order_id, pos.symbol)
        status = order.get("status")
        if status not in ("closed", "canceled", "expired", "rejected"):
            return None
        self.storage.delete_state(f"stop_order:{pos.symbol}")
        if status == "closed":
            return self._fill(order, pos.qty, pos.stop)
        log.warning("Stop de %s en estado %s (ejecutado: %s): se colocará otro", pos.symbol, status, order.get("filled"))
        return None

    def _release_stop(self, symbol: str, ref_price: float) -> Fill | None:
        """Cancela el stop para liberar el saldo. Devuelve lo que llegara a ejecutarse."""
        order_id, _ = self._stop_order(symbol)
        if not order_id:
            return None
        self.ex.cancel_order(order_id, symbol)
        self.storage.delete_state(f"stop_order:{symbol}")
        order = self.ex.fetch_order(order_id, symbol)
        filled = float(order.get("filled") or 0)
        return self._fill(order, filled, ref_price) if filled > 0 else None


def account_id(cfg: dict) -> str:
    if cfg["execution"]["mode"] != "live":
        return "paper"
    ex = cfg["exchange"]
    return f"live:{ex['id']}:{'sandbox' if ex.get('sandbox') else 'real'}"


def make_broker(cfg: dict, storage: Storage):
    account = account_id(cfg)
    storage.check_account(account)
    broker = LiveBroker(cfg, storage) if cfg["execution"]["mode"] == "live" else PaperBroker(cfg, storage)
    storage.set_state("account", account)
    return broker
