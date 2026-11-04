#!/usr/bin/env python3
"""Asistente para ejecutar el bot con DINERO REAL en Binance.

  .venv\\Scripts\\python.exe iniciar_real.py

Comprueba la clave y el saldo sin operar, pide confirmación expresa, ejecuta un ciclo y pregunta
antes de seguir. Las claves se leen del portapapeles y no se guardan. La salida, con las claves
ocultas, queda en data/real-run.log.
"""
import logging
import os
import sys

from probar_testnet import ask_key, clipboard
from trader.bot import TradingBot
from trader.broker import LiveBroker, account_id
from trader.config import ROOT, load_config, override
from trader.storage import Storage

CONFIG = ROOT / "config.live.yaml"
LOG = ROOT / "data" / "real-run.log"
CONFIRM_PHRASE = "ACEPTO USAR DINERO REAL"
CONTINUE_WORD = "SEGUIR"
BINANCE_SOURCE = "la Gestión de API de Binance"

log = logging.getLogger("trader.asistente")


class RedactingFormatter(logging.Formatter):
    def __init__(self, secrets: list[str]):
        super().__init__("%(asctime)s %(message)s", "%H:%M:%S")
        self.secrets = secrets

    def format(self, record):
        text = super().format(record)
        for secret in self.secrets:
            text = text.replace(secret, "***")
        return text


def setup_logging(secrets: list[str]) -> list[logging.Handler]:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(LOG, encoding="utf-8")]
    root = logging.getLogger("trader")
    root.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(RedactingFormatter(secrets))
        root.addHandler(handler)
    return handlers


def minimum_capital(cfg: dict) -> float:
    """Por debajo de esto ninguna posición alcanza el mínimo por operación."""
    return cfg["risk"]["min_notional"] / cfg["risk"]["max_position_pct"]


def ask_capital(minimum: float) -> int:
    while True:
        text = input(f"\n¿Cuánto USDT puede usar el bot como máximo? Mínimo {minimum:.0f}. "
                     "Escribe solo el número entero, sin puntos ni comas: ").strip()
        if text.isdigit() and int(text) >= minimum:
            return int(text)
        print("Ese valor no sirve. Ejemplo: 100")


def key_problems(broker: LiveBroker) -> list[str]:
    """Permisos de la clave. Si no se pueden leer, se trata como un problema."""
    rights = broker.ex.sapiGetAccountApiRestrictions()
    problems = []
    if rights.get("enableWithdrawals") is not False:
        problems.append("la clave tiene los RETIROS activados: desactívalos o crea otra clave")
    if rights.get("enableSpotAndMarginTrading") is not True:
        problems.append("la clave no tiene permiso de operaciones Spot")
    return problems


def describe(summary: dict) -> str:
    opened = ", ".join(summary["opened"]) or "ninguna"
    closed = ", ".join(summary["closed"]) or "ninguna"
    return (f"Pares analizados: {summary['scanned']} | Señales de compra: {summary['buy_signals']}\n"
            f"Compras: {opened} | Ventas: {closed}\n"
            f"Posiciones abiertas: {summary['positions']} | Capital del bot: {summary['equity']} USDT")


def run() -> int:
    cfg = load_live_config()
    if account_id(cfg) != "live:binance:real":
        print(f"{CONFIG.name} no apunta a la cuenta real de Binance: no se ejecuta nada.")
        return 1

    print("=" * 70)
    print("BOT CON DINERO REAL")
    print("Puedes perder el dinero que pongas a su disposición. Software experimental,")
    print("no es asesoría financiera. Hasta confirmar no se envía ninguna orden.")
    print("Para salir en cualquier momento pulsa Ctrl+C.")
    print("=" * 70)

    minimum = minimum_capital(cfg)
    capital = ask_capital(minimum)
    cfg = override(cfg, execution__max_live_capital=capital)
    api_key = ask_key("API Key", source=BINANCE_SOURCE)
    secret = ask_key("Secret Key", other=api_key, source=BINANCE_SOURCE)
    clipboard(" ")
    os.environ.update(EXCHANGE_API_KEY=api_key, EXCHANGE_API_SECRET=secret, CONFIRM_LIVE_TRADING="YES")

    handlers = setup_logging([api_key, secret])
    try:
        return operate(cfg, capital, minimum)
    finally:
        for handler in handlers:
            logging.getLogger("trader").removeHandler(handler)
            handler.close()


def operate(cfg: dict, capital: int, minimum: float) -> int:
    log.info("Comprobando la clave y el saldo, sin operar...")
    try:
        storage = Storage(cfg["storage"]["db_path"])
        storage.check_account(account_id(cfg))
        broker = LiveBroker(cfg, Storage(":memory:"))
        problems = key_problems(broker)
        free = float(broker.ex.fetch_balance()["free"].get(cfg["exchange"]["quote"], 0) or 0)
    except Exception as e:  # noqa: BLE001
        log.error("No se pudo comprobar la cuenta, no se envió ninguna orden: %s: %s", type(e).__name__, e)
        return 1
    if problems:
        for problem in problems:
            log.error("PROBLEMA: %s.", problem)
        log.error("No se envió ninguna orden.")
        return 1

    positions = storage.get_positions()
    log.info("Clave correcta: sin retiros y con operaciones Spot.")
    log.info("USDT libre en Binance: %.2f | Límite para el bot: %d | Posiciones del bot: %d", free, capital, len(positions))
    if min(free, capital) < minimum and not positions:
        log.error("Con %.2f USDT disponibles el bot no podría abrir ninguna posición. No se envió ninguna orden.",
                  min(free, capital))
        return 1

    print(f"\nA partir de aquí el bot puede comprar y vender con hasta {capital} USDT de tu dinero real.")
    if input(f"Para aceptar escribe exactamente: {CONFIRM_PHRASE}\n> ").strip() != CONFIRM_PHRASE:
        log.info("No confirmado. No se envió ninguna orden.")
        return 0

    bot = TradingBot(cfg)
    log.info("Ejecutando un ciclo; puede tardar unos minutos...")
    summary = bot.run_cycle()
    log.info("Resultado del ciclo:\n%s", describe(summary))

    print("\nRevisa en Binance que cada compra tenga su stop en órdenes abiertas.")
    answer = input(f"Para dejar el bot funcionando escribe {CONTINUE_WORD}; para terminar pulsa Enter: ").strip()
    if answer != CONTINUE_WORD:
        log.info("Bot detenido. Las posiciones abiertas siguen en Binance solo con su stop.")
        return 0
    log.info("Bot en marcha. Ctrl+C lo detiene, pero NO cierra las posiciones. Mantén el PC encendido.")
    bot.run_forever()
    return 0


def load_live_config() -> dict:
    return load_config(CONFIG)


def main():
    try:
        code = run()
    except (KeyboardInterrupt, EOFError):
        print("\nDetenido. Si había posiciones abiertas, siguen en Binance solo con su stop.")
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
