"""Persistencia SQLite: señales, posiciones, operaciones y curva de capital."""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .risk import Position

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (scan_ts TEXT, symbol TEXT, bar_time TEXT, price REAL, prob_up REAL,
  q10 REAL, q50 REAL, q90 REAL, target_price REAL, atr_pct REAL, change_24h REAL, quote_volume_24h REAL,
  auc REAL, hit_rate REAL, holdout_edge REAL, tradable INTEGER, signal TEXT, score REAL, source TEXT);
CREATE INDEX IF NOT EXISTS ix_signals_ts ON signals(scan_ts);
CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, qty REAL, entry_price REAL, entry_ts TEXT,
  stop REAL, take_profit REAL, entry_fee REAL, last_checked TEXT, bars_held INTEGER);
CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, side TEXT,
  qty REAL, price REAL, fee REAL, reason TEXT, pnl REAL, mode TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT, equity REAL, cash REAL, exposure REAL, n_positions INTEGER);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


class Storage:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _exec(self, sql, params=()):
        with self.lock, self.conn:
            return self.conn.execute(sql, params)

    def _df(self, sql, params=()):
        with self.lock:
            return pd.read_sql_query(sql, self.conn, params=params)

    # --- estado clave/valor ---
    def get_state(self, key, default=None):
        row = self._exec("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_state(self, key, value):
        self._exec("INSERT OR REPLACE INTO state VALUES (?,?)", (key, str(value)))

    def delete_state(self, key):
        self._exec("DELETE FROM state WHERE key=?", (key,))

    # --- cuenta dueña de esta base de datos ---
    def check_account(self, account: str):
        """Una base de datos pertenece a una sola cuenta: las posiciones de otra (simuladas o de
        testnet) no se deben gestionar con órdenes de esta."""
        bound = self.get_state("account")
        if bound is None and self._has_activity():  # base anterior a este control
            bound = "paper" if self.get_state("paper_cash") is not None else "desconocida"
        if bound not in (None, account):
            raise RuntimeError(f"La base de datos pertenece a la cuenta '{bound}' y se intentó usar con "
                               f"'{account}'. Usa otro storage.db_path para esta cuenta.")

    def _has_activity(self) -> bool:
        return self.get_state("paper_cash") is not None \
            or self._exec("SELECT 1 FROM positions LIMIT 1").fetchone() is not None \
            or self._exec("SELECT 1 FROM trades LIMIT 1").fetchone() is not None

    # --- señales ---
    def save_scan(self, scan: pd.DataFrame, scan_ts: str):
        if scan.empty:
            return
        d = scan.copy()
        d.insert(0, "scan_ts", scan_ts)
        d["tradable"] = d["tradable"].astype(int)
        with self.lock:
            d.to_sql("signals", self.conn, if_exists="append", index=False)

    def latest_scan(self) -> pd.DataFrame:
        return self._df("SELECT * FROM signals WHERE scan_ts=(SELECT MAX(scan_ts) FROM signals) ORDER BY score DESC")

    def signal_history(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        return self._df("SELECT * FROM signals WHERE symbol=? ORDER BY scan_ts DESC LIMIT ?", (symbol, limit))

    # --- posiciones ---
    def get_positions(self) -> dict[str, Position]:
        rows = self._exec("SELECT * FROM positions").fetchall()
        return {r[0]: Position(*r) for r in rows}

    def upsert_position(self, p: Position):
        v = asdict(p)
        self._exec("INSERT OR REPLACE INTO positions VALUES (:symbol,:qty,:entry_price,:entry_ts,:stop,"
                   ":take_profit,:entry_fee,:last_checked,:bars_held)", v)

    def delete_position(self, symbol):
        self._exec("DELETE FROM positions WHERE symbol=?", (symbol,))

    # --- operaciones / equity ---
    def log_trade(self, ts, symbol, side, qty, price, fee, reason, pnl, mode):
        self._exec("INSERT INTO trades (ts,symbol,side,qty,price,fee,reason,pnl,mode) VALUES (?,?,?,?,?,?,?,?,?)",
                   (ts, symbol, side, qty, price, fee, reason, pnl, mode))

    def trades(self) -> pd.DataFrame:
        return self._df("SELECT * FROM trades ORDER BY id DESC")

    def record_equity(self, ts, equity, cash, exposure, n):
        self._exec("INSERT INTO equity VALUES (?,?,?,?,?)", (ts, equity, cash, exposure, n))

    def realized_pnl(self) -> float:
        return float(self._exec("SELECT COALESCE(SUM(pnl), 0) FROM trades").fetchone()[0])

    def deployed_capital(self) -> float:
        return sum(p.qty * p.entry_price + p.entry_fee for p in self.get_positions().values())

    def equity_curve(self) -> pd.DataFrame:
        return self._df("SELECT * FROM equity ORDER BY ts")

    def day_start_equity(self, day: str) -> float | None:
        row = self._exec("SELECT equity FROM equity WHERE ts>=? ORDER BY ts LIMIT 1", (day,)).fetchone()
        return row[0] if row else None
