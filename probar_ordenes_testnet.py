#!/usr/bin/env python3
"""Comprueba en la testnet (dinero ficticio) las órdenes del modo live, sin esperar a una señal:
compra, stop en el exchange, subida del stop, cancelación y venta.

  .venv\\Scripts\\python.exe probar_ordenes_testnet.py

No comprueba que un stop llegue a dispararse: eso depende de que el precio caiga hasta él.
La salida, con las claves ocultas, queda en data/testnet-ordenes.log.
"""
import os
import sys

from probar_testnet import ask_key, clipboard
from trader.broker import LiveBroker, account_id
from trader.config import ROOT, load_config
from trader.risk import Position
from trader.storage import Storage

CONFIG = ROOT / "config.testnet.yaml"
LOG = ROOT / "data" / "testnet-ordenes.log"
SYMBOL = "ETH/USDT"
NOTIONAL = 20.0  # USDT ficticios


class CheckFailed(Exception):
    pass


def open_ids(ex) -> set[str]:
    return {str(o["id"]) for o in ex.fetch_open_orders(SYMBOL)}


def run_checks(broker: LiveBroker, out=print) -> bool:
    """Recorre el ciclo de órdenes y deja la cuenta como estaba. Devuelve True si todo funcionó."""
    ex = broker.ex
    if "testnet" not in ex.urls["api"]["public"]:
        raise RuntimeError("Esta prueba solo se ejecuta contra una testnet.")

    def check(ok, message):
        out(f"{'OK   ' if ok else 'FALLO'} {message}")
        if not ok:
            raise CheckFailed(message)

    pos = None
    try:
        price = float(ex.fetch_ticker(SYMBOL)["last"])
        out(f"Precio de {SYMBOL} en la testnet: {price}")

        fill = broker.buy(SYMBOL, NOTIONAL / price, price)
        check(fill.qty > 0, f"Compra a mercado: {fill.qty} a {fill.price}")
        pos = Position(SYMBOL, fill.qty, fill.price, "", price * 0.90, price * 1.10, fill.fee)
        broker.storage.upsert_position(pos)

        broker.protect(pos)
        first_id, first_trigger = broker._stop_order(SYMBOL)
        check(first_id and first_id in open_ids(ex), f"Stop visible en el exchange, disparo a {first_trigger}")
        check(broker.stop_fill(pos) is None, "El stop sigue activo y sin ejecutar")

        pos.stop = price * 0.95
        broker.protect(pos)
        second_id, second_trigger = broker._stop_order(SYMBOL)
        ids = open_ids(ex)
        check(second_id != first_id and second_id in ids and first_id not in ids, f"Stop subido a {second_trigger}")

        closed = broker.sell(SYMBOL, pos.qty, price)
        pos = None
        check(closed.qty > 0, f"Stop cancelado y venta a mercado: {closed.qty} a {closed.price}")
        check(second_id not in open_ids(ex), "No queda ningún stop de esta prueba abierto")
        return True
    except CheckFailed:
        return False
    except Exception as e:  # noqa: BLE001  se informa y se limpia la cuenta
        out(f"FALLO {type(e).__name__}: {e}")
        return False
    finally:
        if pos is not None:
            try:
                broker.sell(SYMBOL, pos.qty, price)
                out("Limpieza: stop cancelado y posición de prueba vendida")
            except Exception as e:  # noqa: BLE001
                out(f"Limpieza incompleta, revisa la cuenta de testnet: {type(e).__name__}: {e}")


def main():
    cfg = load_config(CONFIG)
    if account_id(cfg) != "live:binance:sandbox":
        sys.exit(f"{CONFIG.name} no apunta a la testnet de Binance: no se ejecuta nada.")
    print(f"Prueba de órdenes en la testnet de Binance: dinero ficticio, unos {NOTIONAL:.0f} USDT.")
    api_key = ask_key("API Key")
    secret = ask_key("Secret Key", other=api_key)
    clipboard("")
    os.environ.update(EXCHANGE_API_KEY=api_key, EXCHANGE_API_SECRET=secret, CONFIRM_LIVE_TRADING="YES")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as fh:
        def out(line):
            line = str(line).replace(api_key, "***").replace(secret, "***")
            print(line)
            fh.write(line + "\n")
            fh.flush()

        try:
            passed = run_checks(LiveBroker(cfg, Storage(":memory:")), out)
        except Exception as e:  # noqa: BLE001
            out(f"FALLO {type(e).__name__}: {e}")
            passed = False
        out("RESULTADO: todo correcto" if passed else "RESULTADO: hubo fallos")
    print("\nTerminado. Vuelve al chat y escribe: listo")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
