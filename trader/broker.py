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
        base = self.ex.market(symbol)["base"]
        free = float(self.ex.fetch_balance()["free"].get(base, 0) or 0)
        amount = float(self.ex.amount_to_precision(symbol, min(qty, free)))
        order = self.ex.create_order(symbol, "market", "sell", amount)
        log.info("LIVE SELL %s %s -> %s", symbol, amount, order.get("id"))
        return self._fill(order, amount, ref_price)


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
