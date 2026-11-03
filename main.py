#!/usr/bin/env python3
"""CLI de Crypto AI Trader.

  python main.py scan                 # escanea todo el universo y muestra el ranking
  python main.py bot --once           # un ciclo del bot (paper por defecto)
  python main.py bot                  # bot en bucle continuo
  python main.py backtest             # backtest walk-forward de todo el universo
  python main.py dashboard            # abre el dashboard web (Streamlit)
  Añade --synthetic para usar datos simulados (sin conexión al exchange).
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from trader.config import load_config, override


def main():
    ap = argparse.ArgumentParser(description="Crypto AI Trader")
    ap.add_argument("command", choices=["scan", "bot", "backtest", "dashboard"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="usar datos sintéticos (pruebas offline)")
    ap.add_argument("--symbols", nargs="*", help="sobrescribe la lista de símbolos")
    ap.add_argument("--top", type=int, help="auto-seleccionar los N pares con más volumen")
    ap.add_argument("--once", action="store_true", help="bot: ejecutar un solo ciclo")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("ccxt", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if args.synthetic:
        cfg = override(cfg, data__source="synthetic")
    if args.symbols:
        cfg = override(cfg, universe__symbols=args.symbols, universe__top_n=0)
    if args.top:
        cfg = override(cfg, universe__top_n=args.top)
    pd.set_option("display.width", 200, "display.max_columns", 30)

    if args.command == "dashboard":
        app = Path(__file__).parent / "dashboard" / "app.py"
        extra = ["--", "--synthetic"] if args.synthetic else []
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *extra]))

    if args.command == "scan":
        from trader.data import MarketData
        from trader.scanner import Scanner
        md = MarketData(cfg)
        frames, sources = md.fetch_many(md.resolve_universe())
        scan = Scanner(cfg).scan(frames, sources)
        cols = ["symbol", "price", "prob_up", "q50", "target_price", "auc", "holdout_edge", "tradable", "signal", "score", "source"]
        print(scan[cols].to_string(index=False) if not scan.empty else "Sin resultados")

    elif args.command == "bot":
        from trader.bot import TradingBot
        bot = TradingBot(cfg)
        print(bot.run_cycle()) if args.once else bot.run_forever()

    elif args.command == "backtest":
        from trader.backtest import backtest_many
        from trader.data import MarketData
        md = MarketData(cfg)
        frames, _ = md.fetch_many(md.resolve_universe())
        table, _ = backtest_many(frames, cfg)
        print(table.to_string(index=False))
        if not table.empty:
            print(f"\nMedia: retorno {table.total_return.mean():.2%} | buy&hold {table.buy_hold.mean():.2%} | "
                  f"maxDD {table.max_drawdown.mean():.2%} | trades {table.trades.sum()}")


if __name__ == "__main__":
    main()
