"""Tests de extremo a extremo con datos sintéticos (no requieren conexión)."""
import numpy as np
import pandas as pd
import pytest

import trader  # noqa: F401
from trader.backtest import backtest_symbol
from trader.bot import TradingBot
from trader.config import load_config, override
from trader.data import drop_open_candle, synthetic_ohlcv
from trader.features import WARMUP, build_features
from trader.model import train_model
from trader.risk import Position, RiskManager
from trader.strategy import decide


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c = override(c, data__source="synthetic", data__history_bars=1500,
                 universe__symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"], universe__model_workers=2)
    c["storage"]["db_path"] = str(tmp_path / "t.db")
    c["model"]["model_dir"] = str(tmp_path / "models")
    return c


def test_features_are_causal():
    df = synthetic_ohlcv("BTC/USDT", "1h", 800)
    full = build_features(df)
    partial = build_features(df.iloc[:600])
    pd.testing.assert_frame_equal(full.iloc[WARMUP:600], partial.iloc[WARMUP:600])  # el futuro no altera el pasado


def test_drop_open_candle():
    df = synthetic_ohlcv("BTC/USDT", "1h", 50)
    last_open = df.index[-1].timestamp()
    assert len(drop_open_candle(df, "1h", now=last_open + 1800)) == 49
    assert len(drop_open_candle(df, "1h", now=last_open + 3600)) == 50


def test_model_outputs(cfg):
    m = train_model("ETH/USDT", synthetic_ohlcv("ETH/USDT", "1h", 1500), cfg)
    p = m.predict_frame(build_features(synthetic_ohlcv("ETH/USDT", "1h", 1500)).iloc[-50:])
    assert p["prob_up"].between(0, 1).all()
    assert (p["q10"] <= p["q50"]).all() and (p["q50"] <= p["q90"]).all()
    assert 0 <= m.metrics["auc"] <= 1 and isinstance(m.metrics["tradable"], bool)


def test_risk_sizing_caps():
    r = RiskManager({"atr_stop_mult": 2, "risk_per_trade": 0.01, "max_position_pct": 0.2, "reward_risk": 2,
                     "min_notional": 10, "max_open_positions": 3, "max_daily_loss_pct": 0.05,
                     "max_holding_bars": 10, "trailing_stop": True})
    qty, stop, tp = r.size(10000, 10000, 100, 1.0)
    assert qty * 100 <= 2000 + 1e-6          # tope 20% del capital
    assert stop == 98 and tp == 104
    assert not r.can_open(3, 10000, 10000)[0]
    assert not r.can_open(0, 10000, 9400)[0]  # kill switch
    pos = Position("X", 1, 100, "", 98, 104)
    assert r.check_bar(pos, 99, 105, 97, 100, 1)[1] == "stop_loss"  # stop y TP en la misma vela -> stop


def test_decide():
    s = {"buy_prob": 0.6, "exit_prob": 0.45, "min_expected_return": 0.003}
    assert decide(0.7, 0.01, True, s) == "BUY"
    assert decide(0.7, 0.01, False, s) == "HOLD"
    assert decide(0.4, 0.01, True, s) == "EXIT"


def test_bot_cycle_and_accounting(cfg):
    bot = TradingBot(cfg)
    s1 = bot.run_cycle()
    assert s1["scanned"] == 3
    s2 = bot.run_cycle()
    eq = bot.storage.equity_curve()
    assert len(eq) == 2 and np.isfinite(eq["equity"]).all()
    assert bot.broker.cash() >= 0


def test_backtest(cfg):
    df = synthetic_ohlcv("SOL/USDT", "1h", 1500)
    r = backtest_symbol("SOL/USDT", df, cfg)
    assert r["equity"].notna().all() and r["max_drawdown"] <= 0
