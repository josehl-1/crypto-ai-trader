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
        stopped = self.check_stops(ts)
        positions = self.storage.get_positions()
        symbols = list(dict.fromkeys(self.data.resolve_universe() + list(positions)))
        t0 = time.time()
        frames, sources = self.data.fetch_many(symbols)
        scan = self.scanner.scan(frames, sources)
        self.storage.save_scan(scan, ts)
        log.info("Escaneados %d/%d símbolos en %.1fs", len(scan), len(symbols), time.time() - t0)
        sig = scan.set_index("symbol") if not scan.empty else pd.DataFrame()

        closed = stopped + self._manage_positions(frames, sig, ts)
        prices ={s: float(df["close"].iloc[-1]) for s, df in frames.items()}
        opened = self._open_positions(frames, scan, prices, ts)
        closed += self._sync_exchange_stops(ts)

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
            self._watch_stops(max(5, loop - (time.time() - start)))

    def _watch_stops(self, seconds: float):
        """Espera hasta el próximo ciclo vigilando los stops cada bot.stop_check_seconds."""
        end = time.time() + seconds
        check = float(self.cfg["bot"].get("stop_check_seconds", 30))
        while (left := end - time.time()) > 0:
            time.sleep(min(check, left))
            try:
                self.check_stops()
            except Exception:  # noqa: BLE001  el bot no debe morir por un error puntual
                log.exception("Error vigilando stops")

    def check_stops(self, ts: str | None = None) -> list[str]:
        """Sincroniza los stops del exchange y compara stops, take profits y el límite de pérdida
        diaria con el precio actual, sin esperar a que cierre la vela."""
        ts = ts or utcnow()
        closed = self._sync_exchange_stops(ts)
        positions = self.storage.get_positions()
        if not positions:
            return closed
        try:
            prices = self.data.fetch_last_prices(list(positions))
        except Exception as e:  # noqa: BLE001
            log.warning("Stops sin vigilar: no se pudo obtener el precio actual (%s)", e)
            return closed
        for sym, pos in positions.items():
            price = prices.get(sym)
            if price is None:
                continue
            reason = "stop_loss" if price <= pos.stop else "take_profit" if price >= pos.take_profit else None
            if reason and self._try_close(pos, price, reason, ts):
                closed.append(sym)
        return closed + self._enforce_daily_loss(prices, ts)

    def _sync_exchange_stops(self, ts: str) -> list[str]:
        """Registra los stops ya ejecutados en el exchange y coloca o sube los que falten."""
        closed = []
        for sym, pos in self.storage.get_positions().items():
            try:
                fill = self.broker.stop_fill(pos)
                if fill:
                    self._record_close(pos, fill, "exchange_stop", ts)
                    closed.append(sym)
                else:
                    self.broker.protect(pos)
            except Exception:  # noqa: BLE001  se reintenta en la siguiente comprobación
                log.exception("Stop de %s sin sincronizar con el exchange", sym)
        return closed

    def _enforce_daily_loss(self, prices: dict, ts: str) -> list[str]:
        """Cierra todo si se supera max_daily_loss_pct; can_open bloquea entradas el resto del día."""
        positions = self.storage.get_positions()
        if not self.cfg["risk"].get("close_on_daily_loss", True) or not positions or any(s not in prices for s in positions):
            return []
        try:
            equity = self.broker.cash() + sum(p.qty * prices[s] for s, p in positions.items())
        except Exception as e:  # noqa: BLE001
            log.warning("Límite diario sin vigilar: no se pudo obtener el saldo (%s)", e)
            return []
        if not self.risk.daily_loss_hit(self.storage.day_start_equity(ts[:10]), equity):
            return []
        log.warning("Límite de pérdida diaria superado (capital %.2f): cerrando %d posiciones", equity, len(positions))
        return [s for s, p in positions.items() if self._try_close(p, prices[s], "kill_switch_daily_loss", ts)]

    # ------------------------------------------------------------------
    def _try_close(self, pos: Position, ref_price: float, reason: str, ts: str) -> bool:
        try:
            self._close(pos, ref_price, reason, ts)
            return True
        except Exception:  # noqa: BLE001  se reintenta en la siguiente comprobación
            log.exception("No se pudo cerrar %s (%s)", pos.symbol, reason)
            return False

    def _close(self, pos: Position, ref_price: float, reason: str, ts: str):
        self._record_close(pos, self.broker.sell(pos.symbol, pos.qty, ref_price), reason, ts)

    def _record_close(self, pos: Position, fill, reason: str, ts: str):
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
                if self._try_close(pos, float(df["close"].iloc[-1]) if live else price, reason, ts):
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
            stop_dist = price - stop
            pos = Position(row["symbol"], fill.qty, fill.price, ts, fill.price - stop_dist,
                           fill.price + self.cfg["risk"]["reward_risk"] * stop_dist, fill.fee,
                           str(frames[row["symbol"]].index[-1]), 0)
            self.storage.upsert_position(pos)
            cash = self.broker.cash()
            self.storage.log_trade(ts, pos.symbol, "BUY", fill.qty, fill.price, fill.fee,
                                   f"p_up={row['prob_up']:.2f} q50={row['q50']:.4f}", 0.0, self.broker.mode)
            log.info("ABIERTA %s qty=%.6g @ %.6g SL=%.6g TP=%.6g", pos.symbol, pos.qty, pos.entry_price, pos.stop, pos.take_profit)
            opened.append(pos.symbol)
            try:
                self.broker.protect(pos)
            except Exception:  # noqa: BLE001  se reintenta en la siguiente comprobación
                log.exception("%s abierta sin stop en el exchange", pos.symbol)
        return opened
