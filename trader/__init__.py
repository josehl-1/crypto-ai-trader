"""Crypto AI Trader: escaneo multi-símbolo, forecasting con ML y bot de trading."""
import os

# Los modelos se entrenan en paralelo (un hilo por símbolo); evitamos que cada uno
# lance además su propio pool OpenMP, lo que satura la CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "1.0.0"
