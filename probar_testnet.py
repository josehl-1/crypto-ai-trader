#!/usr/bin/env python3
"""Asistente para probar un ciclo del bot en la testnet (dinero ficticio).

  .venv\\Scripts\\python.exe probar_testnet.py

Lee las claves del portapapeles sin mostrarlas ni guardarlas. La salida, con las claves
ocultas, queda en data/testnet-run.log.
"""
import getpass
import os
import subprocess
import sys
from pathlib import Path

from trader.broker import account_id
from trader.config import ROOT, load_config

CONFIG = ROOT / "config.testnet.yaml"
LOG = ROOT / "data" / "testnet-run.log"
MIN_KEY_LENGTH = 20


def clipboard(text: str | None = None) -> str:
    """Lee el portapapeles de Windows o lo reemplaza si se pasa text."""
    cmd = "Get-Clipboard -Raw" if text is None else f"Set-Clipboard -Value '{text}'"
    done = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    return (done.stdout or "").strip()


def ask_key(name: str, other: str = "") -> str:
    while True:
        getpass.getpass(f"\nCopia tu {name} desde la web de la testnet y pulsa Enter aquí (no hace falta pegarla)... ")
        key = clipboard()
        if len(key) >= MIN_KEY_LENGTH and not any(c.isspace() for c in key) and key != other:
            print(f"{name} recibida ({len(key)} caracteres).")
            return key
        print(f"Eso no parece tu {name}. Cópiala de nuevo e inténtalo otra vez.")


def main():
    if account_id(load_config(CONFIG)) != "live:binance:sandbox":
        sys.exit(f"{CONFIG.name} no apunta a la testnet de Binance: no se ejecuta nada.")
    print("Prueba en la testnet de Binance: dinero ficticio, un solo ciclo.")
    api_key = ask_key("API Key")
    secret = ask_key("Secret Key", other=api_key)
    clipboard("")

    env = {**os.environ, "EXCHANGE_API_KEY": api_key, "EXCHANGE_API_SECRET": secret, "CONFIRM_LIVE_TRADING": "YES",
           "PYTHONIOENCODING": "utf-8"}
    print("\nEjecutando el bot; puede tardar unos minutos...\n")
    proc = subprocess.Popen([sys.executable, str(ROOT / "main.py"), "bot", "--once", "--config", str(CONFIG)], env=env,
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(LOG), "w", encoding="utf-8") as fh:
        for line in proc.stdout:
            line = line.replace(api_key, "***").replace(secret, "***")
            print(line, end="")
            fh.write(line)
    code = proc.wait()
    print(f"\nTerminado (código {code}). Vuelve al chat y escribe: listo")
    sys.exit(code)


if __name__ == "__main__":
    main()
