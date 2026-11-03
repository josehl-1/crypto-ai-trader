"""Modelos de IA por símbolo:
 - Clasificador gradient boosting -> P(retorno a H velas > umbral)
 - 3 regresores cuantílicos (10/50/90) -> forecast del retorno con banda de incertidumbre
Validación fuera de muestra con purga temporal (sin fuga de información)."""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from .features import WARMUP, build_features, forward_log_return

log = logging.getLogger(__name__)
QUANTILES = (0.1, 0.5, 0.9)
_common = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
               l2_regularization=1.0, random_state=7)


def _clf():
    return HistGradientBoostingClassifier(**_common)


def _qreg(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **_common)


@dataclass
class SymbolModel:
    symbol: str
    timeframe: str
    horizon: int
    features: list
    clf: object
    qmodels: dict
    metrics: dict = field(default_factory=dict)
    trained_at: float = 0.0
    last_bar: str = ""

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features]
        out = pd.DataFrame(index=X.index)
        out["prob_up"] = self.clf.predict_proba(X)[:, 1]
        qs = np.column_stack([self.qmodels[q].predict(X) for q in QUANTILES])
        qs.sort(axis=1)  # evita cruce de cuantiles
        out["q10"], out["q50"], out["q90"] = qs[:, 0], qs[:, 1], qs[:, 2]
        return out


def make_dataset(df: pd.DataFrame, horizon: int):
    X = build_features(df).iloc[WARMUP:]
    y = forward_log_return(df["close"], horizon).reindex(X.index)
    return X, y


def train_model(symbol: str, df: pd.DataFrame, cfg: dict) -> SymbolModel:
    m, s = cfg["model"], cfg["strategy"]
    H, thr = int(m["horizon"]), float(m["label_threshold"])
    fee = float(cfg["execution"]["fee_rate"]) + float(cfg["execution"]["slippage"])
    X, y = make_dataset(df, H)
    keep = y.notna()
    X, y = X[keep], y[keep]
    if len(X) < 400:
        raise ValueError(f"{symbol}: historia insuficiente ({len(X)} filas)")

    # ---- validación fuera de muestra (holdout final + purga de H velas) ----
    split = int(len(X) * (1 - m["test_size"]))
    Xtr, ytr, Xte, yte = X.iloc[: split - H], y.iloc[: split - H], X.iloc[split:], y.iloc[split:]
    clf = _clf().fit(Xtr, ytr > thr)
    qm = {q: _qreg(q).fit(Xtr, ytr) for q in QUANTILES}
    tmp = SymbolModel(symbol, cfg["data"]["timeframe"], H, list(X.columns), clf, qm)
    p = tmp.predict_frame(Xte)
    lab = (yte > thr).values
    auc = roc_auc_score(lab, p["prob_up"]) if 0 < lab.mean() < 1 else 0.5
    sig = (p["prob_up"] >= s["buy_prob"]) & (p["q50"] >= s["min_expected_return"])
    n_sig = int(sig.sum())
    edge = float(yte[sig].mean() - 2 * fee) if n_sig else 0.0
    metrics = {
        "auc": round(float(auc), 4),
        "accuracy": round(float(((p["prob_up"] > 0.5).values == lab).mean()), 4),
        "base_rate": round(float(lab.mean()), 4),
        "holdout_signals": n_sig,
        "hit_rate": round(float((yte[sig] > 0).mean()), 4) if n_sig else None,
        "holdout_edge": round(edge, 5),
        "q_coverage": round(float(((yte >= p["q10"]) & (yte <= p["q90"])).mean()), 4),
        "n_train": len(Xtr), "n_test": len(Xte),
    }
    tradable = auc >= m["min_auc"]
    if m.get("require_positive_holdout_edge", True):
        tradable = tradable and n_sig >= m.get("min_holdout_signals", 10) and edge > 0
    metrics["tradable"] = bool(tradable)

    # ---- modelo final entrenado con todos los datos disponibles ----
    clf = _clf().fit(X, y > thr)
    qm = {q: _qreg(q).fit(X, y) for q in QUANTILES}
    return SymbolModel(symbol, cfg["data"]["timeframe"], H, list(X.columns), clf, qm, metrics,
                       time.time(), str(df.index[-1]))


def forecast_path(last_time, price: float, pred: pd.Series, horizon: int, tf_sec: int) -> pd.DataFrame:
    """Trayectoria de forecast interpolada (mediana lineal, bandas ~ sqrt(t))."""
    k = np.arange(0, horizon + 1)
    frac = k / horizon
    med = pred["q50"] * frac
    lo = med + (pred["q10"] - pred["q50"]) * np.sqrt(frac)
    hi = med + (pred["q90"] - pred["q50"]) * np.sqrt(frac)
    idx = pd.DatetimeIndex([last_time + pd.Timedelta(seconds=int(tf_sec * i)) for i in k])
    return pd.DataFrame({"median": price * np.exp(med), "low": price * np.exp(lo), "high": price * np.exp(hi)}, index=idx)


class ModelRegistry:
    """Cache en memoria + disco. Reentrena cuando el modelo supera retrain_hours."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = Path(cfg["model"]["model_dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, SymbolModel] = {}
        self.lock = threading.Lock()

    def _path(self, symbol: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", symbol)
        return self.dir / f"{safe}_{self.cfg['data']['timeframe']}_h{self.cfg['model']['horizon']}.joblib"

    def _fresh(self, m: SymbolModel | None) -> bool:
        return (m is not None and m.horizon == self.cfg["model"]["horizon"]
                and time.time() - m.trained_at < self.cfg["model"]["retrain_hours"] * 3600)

    def cached(self, symbol: str) -> SymbolModel | None:
        """Modelo vigente (memoria o disco) o None si hay que reentrenar."""
        m = self.mem.get(symbol)
        if not self._fresh(m):
            p = self._path(symbol)
            m = None
            if p.exists():
                try:
                    m = joblib.load(p)
                except Exception:  # noqa: BLE001
                    m = None
        if self._fresh(m):
            with self.lock:
                self.mem[symbol] = m
            return m
        return None

    def put(self, m: SymbolModel):
        joblib.dump(m, self._path(m.symbol))
        with self.lock:
            self.mem[m.symbol] = m

    def get(self, symbol: str, df: pd.DataFrame, force: bool = False) -> SymbolModel:
        m = None if force else self.cached(symbol)
        if m is None:
            log.info("Entrenando modelo %s (%d velas)", symbol, len(df))
            m = train_model(symbol, df, self.cfg)
            self.put(m)
        return m

    def train_many(self, frames: dict[str, pd.DataFrame], workers: int = 4, force: bool = False) -> dict[str, str]:
        """Entrena en paralelo (procesos, evita el GIL) los modelos ausentes o caducados.
        Devuelve {símbolo: error} de los que fallaron."""
        todo = {s: df for s, df in frames.items() if force or self.cached(s) is None}
        errors: dict[str, str] = {}
        if not todo:
            return errors
        log.info("Entrenando %d modelos con %d procesos...", len(todo), workers)
        t0 = time.time()
        for sym, res in parallel_map(_train_job, [(s, df, self.cfg) for s, df in todo.items()], workers, keys=list(todo)):
            if isinstance(res, Exception):
                errors[sym] = str(res)
                log.warning("%s: entrenamiento falló: %s", sym, res)
            else:
                self.put(res)
        log.info("Entrenamiento completado en %.1fs", time.time() - t0)
        return errors


def _train_job(args):
    return train_model(*args)


def parallel_map(fn, items: list, workers: int, keys: list):
    """Ejecuta fn(item) en procesos; si no es posible (entorno restringido) usa hilos.
    Produce (clave, resultado | excepción)."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import multiprocessing as mp

    workers = max(1, min(workers, len(items)))
    try:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) if workers > 1 \
            else ThreadPoolExecutor(max_workers=1)
    except Exception:  # noqa: BLE001
        pool = ThreadPoolExecutor(max_workers=workers)
    with pool:
        futs = [pool.submit(fn, it) for it in items]
        for k, f in zip(keys, futs):
            try:
                yield k, f.result()
            except Exception as e:  # noqa: BLE001
                yield k, e
