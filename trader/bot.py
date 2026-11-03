"""Bot de trading: ciclo escanear -> gestionar posiciones -> abrir nuevas -> registrar."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from .broker import make_broker
from .data import MarketData
from .features import atr
from .risk import Position, RiskManager
from .scanner import Scanner
from .storage import Storage

log = logging.getLogger(__name__)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


class TradingBot:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.storage = Storage(cfg["storage"]["db_path"])
        self.data = MarketData(cfg)
        self.scanner = Scanner(cfg)
        self.risk = RiskManager(cfg["risk"], cfg["execution"]["fee_rate"])
        self.broker = make_broker(cfg, self.storage)

    # ------------------------------------------------------------------
    def run_cycle(self) -> dict:
        ts = utcnow()
        positions = self.storage.get_positions()
        symbols = list(dict.fromkeys(self.data.resolve_universe() + list(positions)))
        t0 = time.time()
        frames, sources = self.data.fetch_many(symbols)
        scan = self.scanner.scan(frames, sources)
        self.storage.save_scan(scan, ts)
        log.info("Escaneados %d/%d símbolos en %.1fs", len(scan), len(symbols), time.time() - t0)
        sig = scan.set_index("symbol") if not scan.empty else pd.DataFrame()

        closed = self._manage_positions(frames, sig, ts)
        prices = {s: float(df["close"].iloc[-1]) for s, df in frames.items()}
        opened = self._open_positions(frames, scan, prices, ts)

        positions = self.storage.get_positions()
        exposure = sum(p.qty * prices.get(s, p.entry_price) for s, p in positions.items())
        cash = self.broker.cash()
        self.storage.record_equity(ts, cash + exposure, cash, exposure, len(positions))
        summary = {"ts": ts, "scanned": len(scan), "buy_signals": int((scan["signal"] == "BUY").sum()) if len(scan) else 0,
                   "opened": opened, "closed": closed, "equity": round(cash + exposure, 2), "positions": len(positions)}
        log.info("Ciclo: %s", summary)
        return summary

    def run_forever(self):
        loop = int(self.cfg["bot"]["loop_seconds"])
        log.info("Bot iniciado en modo %s, ciclo cada %ds", self.broker.mode, loop)
        while True:
            start = time.time()
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001  el bot no debe morir por un error puntual
                log.exception("Error en el ciclo")
            time.sleep(max(5, loop - (time.time() - start)))

    # ------------------------------------------------------------------
    def _close(self, pos: Position, ref_price: float, reason: str, ts: str):
        fill = self.broker.sell(pos.symbol, pos.qty, ref_price)
        pnl = (fill.price - pos.entry_price) * fill.qty - fill.fee - pos.entry_fee
        self.storage.log_trade(ts, pos.symbol, "SELL", fill.qty, fill.price, fill.fee, reason, pnl, self.broker.mode)
        self.storage.delete_position(pos.symbol)
        log.info("CERRADA %s @ %.6g (%s) PnL=%.2f", pos.symbol, fill.price, reason, pnl)

    def _manage_positions(self, frames, sig: pd.DataFrame, ts: str) -> list[str]:
        closed = []
        live = self.broker.mode == "live"
        for sym, pos in self.storage.get_positions().items():
            df = frames.get(sym)
            if df is None or df.empty:
                continue
            a = atr(df)
            new = df[df.index > pd.Timestamp(pos.last_checked)] if pos.last_checked else df.tail(1)
            exit_ = None
            for t, bar in new.iterrows():
                exit_ = self.risk.check_bar(pos, bar.open, bar.high, bar.low, bar.close, float(a.loc[t]))
                pos.last_checked = str(t)
                if exit_:
                    break
            if exit_ is None and sym in sig.index and sig.loc[sym, "signal"] == "EXIT":
                exit_ = (float(df["close"].iloc[-1]), "model_exit")
            if exit_:
                price, reason = exit_
                # en real solo podemos vender a mercado ahora (stops gestionados por el bot)
                self._close(pos, float(df["close"].iloc[-1]) if live else price, reason, ts)
                closed.append(sym)
            else:
                self.storage.upsert_position(pos)
        return closed

    def _open_positions(self, frames, scan: pd.DataFrame, prices: dict, ts: str) -> list[str]:
        if scan.empty:
            return []
        positions = self.storage.get_positions()
        cash = self.broker.cash()
        equity = cash + sum(p.qty * prices.get(s, p.entry_price) for s, p in positions.items())
        day_eq = self.storage.day_start_equity(ts[:10])
        cands = scan[(scan["signal"] == "BUY") & scan["tradable"] & ~scan["symbol"].isin(list(positions))]
        if self.cfg["data"]["source"] == "exchange":  # nunca operar con datos de relleno
            cands = cands[cands["source"] != "synthetic"]
        opened = []
        for _, row in cands.iterrows():
            ok, why = self.risk.can_open(len(positions) + len(opened), day_eq, equity)
            if not ok:
                log.info("Sin nuevas entradas: %s", why)
                break
            price = row["price"]
            qty, stop, tp = self.risk.size(equity, cash, price, row["atr_pct"] * price)
            if qty <= 0:
                continue
            fill = self.broker.buy(row["symbol"], qty, price)
            if fill.qty <= 0:
                continue
            cash = self.broker.cash()
            stop_dist = price - stop
            pos = Position(row["symbol"], fill.qty, fill.price, ts, fill.price - stop_dist,
                           fill.price + self.cfg["risk"]["reward_risk"] * stop_dist, fill.fee,
                           str(frames[row["symbol"]].index[-1]), 0)
            self.storage.upsert_position(pos)
            self.storage.log_trade(ts, pos.symbol, "BUY", fill.qty, fill.price, fill.fee,
                                   f"p_up={row['prob_up']:.2f} q50={row['q50']:.4f}", 0.0, self.broker.mode)
            log.info("ABIERTA %s qty=%.6g @ %.6g SL=%.6g TP=%.6g", pos.symbol, pos.qty, pos.entry_price, pos.stop, pos.take_profit)
            opened.append(pos.symbol)
        return opened
