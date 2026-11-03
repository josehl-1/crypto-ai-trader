"""Ingeniería de features técnicas. Todas son causales (solo usan datos pasados)."""
import numpy as np
import pandas as pd

WARMUP = 200  # velas descartadas al inicio (EMA200, etc.)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)
    lr = np.log(c).diff()
    for n in (1, 3, 6, 12, 24, 48):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    for n in (12, 24, 72):
        f[f"vol_{n}"] = lr.rolling(n).std()
    f["vol_ratio"] = f["vol_12"] / f["vol_72"]
    f["rsi_14"] = rsi(c, 14)
    f["rsi_6"] = rsi(c, 6)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    for n in (9, 21, 50, 200):
        f[f"ema_{n}_dist"] = c / c.ewm(span=n, adjust=False).mean() - 1
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20)
    f["bb_width"] = 4 * s20 / m20
    f["atr_pct"] = atr(df) / c
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_k"] = (c - lo14) / (hi14 - lo14)
    lv = np.log1p(v)
    f["volume_z"] = (lv - lv.rolling(48).mean()) / lv.rolling(48).std()
    f["range_pct"] = (h - l) / c
    f["close_pos"] = (c - l) / (h - l)
    f["dist_high_48"] = c / h.rolling(48).max() - 1
    f["dist_low_48"] = c / l.rolling(48).min() - 1
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour + df.index.minute / 60
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return f.replace([np.inf, -np.inf], np.nan)


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    return np.log(close.shift(-horizon) / close)
