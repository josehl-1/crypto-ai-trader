"""Tests del modo live con un exchange falso: no usan la red ni envían órdenes."""
from types import SimpleNamespace

import pandas as pd
import pytest

import trader.broker as broker_mod
from trader.bot import TradingBot
from trader.broker import LiveBroker, make_broker
from trader.config import load_config, override
from trader.data import synthetic_ohlcv
from trader.risk import Position
from trader.storage import Storage

PRICE = 2000.0


class FakeExchange:
    id = "fake"

    def __init__(self):
        self.balance = {"USDT": 1000.0, "ETH": 5.0, "BTC": 5.0}
        self.orders = []
        self.sandbox = False

    def set_sandbox_mode(self, value):
        self.sandbox = value

    def load_markets(self):
        return {}

    def fetch_balance(self):
        return {"free": dict(self.balance)}

    def amount_to_precision(self, symbol, amount):
        return f"{amount:.6f}"

    def market(self, symbol):
        return {"base": symbol.split("/")[0]}

    def create_order(self, symbol, type_, side, amount):
        self.orders.append((symbol, type_, side, amount))
        sign = 1 if side == "buy" else -1
        self.balance["USDT"] -= sign * amount * PRICE
        self.balance[symbol.split("/")[0]] += sign * amount
        return {"id": str(len(self.orders)), "filled": amount, "average": PRICE}


@pytest.fixture
def fake(monkeypatch):
    ex = FakeExchange()
    monkeypatch.setattr(broker_mod, "ccxt", SimpleNamespace(binance=lambda params: ex))
    monkeypatch.setenv("CONFIRM_LIVE_TRADING", "YES")
    monkeypatch.setenv("EXCHANGE_API_KEY", "clave-de-prueba")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "secreto-de-prueba")
    return ex


@pytest.fixture
def cfg(tmp_path):
    c = override(load_config(), universe__symbols=["ETH/USDT"], data__source="exchange")
    c["storage"]["db_path"] = str(tmp_path / "t.db")
    c["model"]["model_dir"] = str(tmp_path / "models")
    return c


def live(cfg, capital=100, sandbox=False):
    c = override(cfg, execution__mode="live", execution__max_live_capital=capital, exchange__sandbox=sandbox)
    return c


@pytest.fixture
def storage(cfg):
    return Storage(cfg["storage"]["db_path"])


# ---------------------------------------------------------------- arranque
def test_live_needs_confirmation(cfg, storage, fake, monkeypatch):
    monkeypatch.delenv("CONFIRM_LIVE_TRADING")
    with pytest.raises(RuntimeError, match="CONFIRM_LIVE_TRADING"):
        LiveBroker(live(cfg), storage)


@pytest.mark.parametrize("capital", [0, None, -5])
def test_live_needs_capital_limit(cfg, storage, fake, capital):
    with pytest.raises(RuntimeError, match="max_live_capital"):
        LiveBroker(live(cfg, capital=capital), storage)


def test_live_refuses_synthetic_data(cfg, storage, fake):
    with pytest.raises(RuntimeError, match="sintéticos"):
        LiveBroker(override(live(cfg), data__source="synthetic"), storage)


def test_sandbox_flag_reaches_exchange(cfg, storage, fake):
    LiveBroker(live(cfg, sandbox=True), storage)
    assert fake.sandbox is True


# ---------------------------------------------------------------- límite de capital
def test_cash_limited_by_capital(cfg, storage, fake):
    assert LiveBroker(live(cfg, capital=100), storage).cash() == 100


def test_cash_limited_by_balance(cfg, storage, fake):
    fake.balance["USDT"] = 30.0
    assert LiveBroker(live(cfg, capital=100), storage).cash() == 30


def test_open_positions_and_losses_reduce_capital(cfg, storage, fake):
    b = LiveBroker(live(cfg, capital=100), storage)
    storage.upsert_position(Position("ETH/USDT", 0.02, PRICE, "", 1900, 2200, entry_fee=0.04))
    assert b.cash() == pytest.approx(59.96)
    storage.log_trade("t", "BTC/USDT", "SELL", 1, 1, 0, "stop_loss", -10.0, "live")
    assert b.cash() == pytest.approx(49.96)


def test_buy_cannot_exceed_capital(cfg, storage, fake):
    fill = LiveBroker(live(cfg, capital=100), storage).buy("ETH/USDT", 1.0, PRICE)
    assert fill.qty * fill.price + fill.fee <= 100 + 1e-2
    assert fake.orders[0][:3] == ("ETH/USDT", "market", "buy")


def test_buy_without_capital_sends_no_order(cfg, storage, fake):
    storage.log_trade("t", "BTC/USDT", "SELL", 1, 1, 0, "stop_loss", -100.0, "live")
    assert LiveBroker(live(cfg, capital=100), storage).buy("ETH/USDT", 1.0, PRICE).qty == 0
    assert fake.orders == []


def test_sell_capped_by_free_balance(cfg, storage, fake):
    fake.balance["ETH"] = 0.5
    LiveBroker(live(cfg), storage).sell("ETH/USDT", 1.0, PRICE)
    assert fake.orders == [("ETH/USDT", "market", "sell", 0.5)]


def test_bot_opens_within_capital(cfg, fake):
    c = live(cfg, capital=100)
    c["risk"].update(max_position_pct=0.8, risk_per_trade=0.9)
    bot = TradingBot(c)
    symbols = ["ETH/USDT", "BTC/USDT"]
    scan = pd.DataFrame({"symbol": symbols, "signal": "BUY", "tradable": True, "source": "binance", "price": PRICE,
                         "atr_pct": 0.01, "prob_up": 0.7, "q50": 0.01})
    frames = {s: synthetic_ohlcv(s, "1h", 10) for s in symbols}
    opened = bot._open_positions(frames, scan, {s: PRICE for s in symbols}, "2026-01-01T00:00:00+00:00")
    assert opened == symbols
    assert bot.storage.deployed_capital() <= 100
    assert bot.broker.cash() >= 0


# ---------------------------------------------------------------- una base de datos, una cuenta
def test_live_refuses_paper_database(cfg, storage, fake):
    make_broker(cfg, storage)
    with pytest.raises(RuntimeError, match="paper"):
        make_broker(live(cfg), storage)


def test_paper_refuses_live_database(cfg, storage, fake):
    make_broker(live(cfg), storage)
    with pytest.raises(RuntimeError, match="live:binance:real"):
        make_broker(cfg, storage)


def test_real_refuses_testnet_database(cfg, storage, fake):
    make_broker(live(cfg, sandbox=True), storage)
    with pytest.raises(RuntimeError, match="live:binance:sandbox"):
        make_broker(live(cfg, sandbox=False), storage)


def test_live_refuses_database_without_owner(cfg, storage, fake):
    storage.upsert_position(Position("ETH/USDT", 1, PRICE, "", 1900, 2200))
    with pytest.raises(RuntimeError, match="desconocida"):
        make_broker(live(cfg), storage)


def test_same_account_can_reopen_database(cfg, storage, fake):
    make_broker(live(cfg), storage)
    make_broker(live(cfg), storage)


def test_failed_live_start_does_not_claim_database(cfg, storage, fake):
    with pytest.raises(RuntimeError):
        make_broker(live(cfg, capital=0), storage)
    make_broker(cfg, storage)


# ---------------------------------------------------------------- stops con precio actual
def prices_bot(c, prices):
    bot = TradingBot(c)
    bot.data.fetch_last_prices = lambda symbols: prices
    bot.storage.upsert_position(Position("ETH/USDT", 0.01, PRICE, "", 1900, 2200))
    return bot


@pytest.mark.parametrize("price,reason", [(1899, "stop_loss"), (1900, "stop_loss"), (2200, "take_profit")])
def test_check_stops_closes(cfg, price, reason):
    bot = prices_bot(cfg, {"ETH/USDT": price})
    assert bot.check_stops() == ["ETH/USDT"]
    assert bot.storage.get_positions() == {}
    assert bot.storage.trades().iloc[0]["reason"] == reason


def test_check_stops_holds_between_levels(cfg):
    bot = prices_bot(cfg, {"ETH/USDT": 2000})
    assert bot.check_stops() == []
    assert "ETH/USDT" in bot.storage.get_positions()


def test_check_stops_survives_price_failure(cfg):
    bot = prices_bot(cfg, {})

    def boom(symbols):
        raise ConnectionError("sin red")

    bot.data.fetch_last_prices = boom
    assert bot.check_stops() == []
    assert "ETH/USDT" in bot.storage.get_positions()


def test_check_stops_sends_live_sell(cfg, fake):
    bot = prices_bot(live(cfg), {"ETH/USDT": 1800})
    assert bot.check_stops() == ["ETH/USDT"]
    assert fake.orders == [("ETH/USDT", "market", "sell", 0.01)]


def test_failed_sell_keeps_position_for_retry(cfg, fake):
    bot = prices_bot(live(cfg), {"ETH/USDT": 1800})

    def reject(*args):
        raise ConnectionError("orden rechazada")

    fake.create_order = reject
    assert bot.check_stops() == []
    assert "ETH/USDT" in bot.storage.get_positions()


def test_watch_stops_checks_repeatedly(cfg):
    c = override(cfg, bot__stop_check_seconds=0.01)
    bot = TradingBot(c)
    calls = []
    bot.check_stops = lambda: calls.append(1)
    bot._watch_stops(0.05)
    assert len(calls) >= 2


def test_synthetic_source_has_no_live_prices(cfg):
    bot = TradingBot(override(cfg, data__source="synthetic"))
    assert bot.data.fetch_last_prices(["ETH/USDT"]) == {}
