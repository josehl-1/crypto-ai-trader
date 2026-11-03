"""Descarga concurrente de velas OHLCV (ccxt async) + generador sintético para pruebas."""
from __future__ import annotations

import asyncio
import logging
import time
import zlib

import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
from scipy.signal import lfilter

log = logging.getLogger(__name__)

STABLES = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "USDD", "PYUSD", "USDE", "EUR", "EURI", "AEUR"}


def tf_seconds(tf: str) -> int:
    return int(ccxt.Exchange.parse_timeframe(tf))


def ohlcv_to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "time"
    return df.drop(columns="ts").astype(float)


def drop_open_candle(df: pd.DataFrame, timeframe: str, now: float | None = None) -> pd.DataFrame:
    """Elimina la vela que aún no ha cerrado: el bot solo decide con velas cerradas."""
    now_ts = pd.Timestamp(now or time.time(), unit="s", tz="UTC")
    return df[df.index + pd.Timedelta(seconds=tf_seconds(timeframe)) <= now_ts]


def synthetic_ohlcv(symbol: str, timeframe: str, bars: int, now: float | None = None) -> pd.DataFrame:
    """Serie sintética determinista por símbolo (régimenes de tendencia, clusters de volatilidad
    y algo de momentum). Anclada a 2024-01-01 para que cada llamada añada velas nuevas de forma
    consistente con el paso del tiempo. SOLO para pruebas: no representa el mercado real."""
    sec = tf_seconds(timeframe)
    now = now or time.time()
    origin = 1704067200
    last_open = int(now // sec) * sec - sec
    n = max((last_open - origin) // sec + 1, bars)
    origin = last_open - (n - 1) * sec
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    base_vol = 0.006 * np.sqrt(sec / 3600) * rng.uniform(0.6, 1.6)
    vol_mult = np.exp(lfilter([1], [1, -0.985], rng.normal(0, 0.06, n)))
    regime_id = np.cumsum(rng.random(n) < 1 / 250)
    drifts = rng.normal(0, base_vol * 0.12, regime_id.max() + 1)
    eps = rng.standard_t(5, n) / np.sqrt(5 / 3) * base_vol * vol_mult
    r = lfilter([1], [1, -0.12], eps) + drifts[regime_id]
    p0 = float(np.exp(rng.uniform(np.log(0.2), np.log(40000))))
    close = p0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[p0], close[:-1]])
    wick = np.abs(rng.normal(0, 0.5, (2, n))) * base_vol * vol_mult
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    volume = rng.lognormal(0, 0.4, n) * (1 + 3 * np.abs(r) / base_vol) * (2e6 / p0)
    idx = pd.to_datetime(origin + np.arange(n) * sec, unit="s", utc=True)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    df.index.name = "time"
    return df.tail(bars)


class MarketData:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ex_id = cfg["exchange"]["id"]
        self.quote = cfg["exchange"]["quote"]
        self.tf = cfg["data"]["timeframe"]
        self.bars = int(cfg["data"]["history_bars"])
        self.source = cfg["data"]["source"]
        # nunca rellenar con datos falsos si se opera con dinero real
        self.fallback = cfg["data"].get("synthetic_fallback", False) and cfg["execution"]["mode"] == "paper"
        self.concurrency = int(cfg["universe"].get("max_concurrency", 8))

    # ---------------- universo ----------------
    def resolve_universe(self) -> list[str]:
        ucfg = self.cfg["universe"]
        top_n = int(ucfg.get("top_n") or 0)
        if top_n <= 0 or self.source != "exchange":
            return list(ucfg["symbols"])
        try:
            ex = getattr(ccxt, self.ex_id)({"enableRateLimit": True})
            markets = ex.load_markets()
            tickers = ex.fetch_tickers()
            rows = []
            for s, t in tickers.items():
                m = markets.get(s)
                if not m or not m.get("spot") or not m.get("active", True):
                    continue
                if m["quote"] != self.quote or m["base"] in STABLES:
                    continue
                qv = t.get("quoteVolume") or ((t.get("baseVolume") or 0) * (t.get("last") or 0))
                rows.append((s, qv or 0))
            rows.sort(key=lambda x: x[1], reverse=True)
            symbols = [s for s, _ in rows[:top_n]]
            log.info("Universo automático: top %d por volumen -> %s", top_n, symbols)
            return symbols
        except Exception as e:  # noqa: BLE001
            log.warning("No se pudo obtener el universo automático (%s); usando lista del config", e)
            return list(ucfg["symbols"])

    # ---------------- descarga ----------------
    def fetch_many(self, symbols: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        if self.source == "synthetic":
            return ({s: synthetic_ohlcv(s, self.tf, self.bars) for s in symbols}, {s: "synthetic" for s in symbols})
        return asyncio.run(self._fetch_many_async(symbols))

    def fetch_one(self, symbol: str) -> tuple[pd.DataFrame | None, str]:
        frames, sources = self.fetch_many([symbol])
        return frames.get(symbol), sources.get(symbol, "none")

    async def _fetch_many_async(self, symbols):
        frames, sources = {}, {}
        ex = getattr(ccxt_async, self.ex_id)({"enableRateLimit": True})
        try:
            try:
                await ex.load_markets()
                sem = asyncio.Semaphore(self.concurrency)
                results = await asyncio.gather(*[self._fetch_symbol(ex, s, sem) for s in symbols], return_exceptions=True)
            except Exception as e:  # noqa: BLE001  exchange inaccesible
                log.error("Exchange %s inaccesible: %s", self.ex_id, e)
                results = [e] * len(symbols)
        finally:
            await ex.close()
        for s, r in zip(symbols, results):
            if isinstance(r, pd.DataFrame) and len(r) > 0:
                frames[s], sources[s] = r, self.ex_id
            elif self.fallback:
                log.warning("%s: fallo de descarga (%s) -> datos SINTÉTICOS", s, r)
                frames[s], sources[s] = synthetic_ohlcv(s, self.tf, self.bars), "synthetic"
            else:
                log.warning("%s: sin datos (%s), se omite", s, r)
        return frames, sources

    async def _fetch_symbol(self, ex, symbol: str, sem: asyncio.Semaphore) -> pd.DataFrame:
        if symbol not in ex.markets:
            raise ValueError(f"{symbol} no existe en {self.ex_id}")
        ms = tf_seconds(self.tf) * 1000
        now_ms = int(time.time() * 1000)
        since = now_ms - (self.bars + 1) * ms
        rows: list = []
        async with sem:
            for _ in range(50):  # límite de páginas
                batch = await ex.fetch_ohlcv(symbol, self.tf, since=since, limit=1000)
                if not batch:
                    break
                rows.extend(batch)
                since = batch[-1][0] + ms
                if since >= now_ms - ms:
                    break
        df = drop_open_candle(ohlcv_to_df(rows), self.tf)
        return df.tail(self.bars)
