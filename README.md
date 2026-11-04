# Crypto AI Trader

Sistema automatizado de trading de criptomonedas con:

- **Escáner multi-crypto concurrente**: descarga en paralelo (asyncio + ccxt) decenas de pares y entrena un modelo de IA por símbolo usando varios procesos a la vez.
- **Forecasting con IA**: por cada crypto, un clasificador gradient boosting estima `P(subida)` a *H* velas, y tres regresores cuantílicos (q10/q50/q90) dan el forecast del precio con su banda de incertidumbre.
- **Validación honesta**: holdout fuera de muestra con purga temporal. Un símbolo solo se marca como **operable** si su AUC supera el mínimo y la estrategia tuvo ventaja positiva (después de comisiones) en datos que el modelo no vio.
- **Bot de trading estructurado**: el ciclo es escanear → gestionar posiciones (stop, take profit, trailing, salida por tiempo o por el modelo) → abrir las mejores señales → registrar. Usa paper trading por defecto y trading real vía ccxt solo si lo habilitas de forma explícita.
- **Gestión de riesgo**: riesgo fijo por operación (1% hasta el stop), stop por ATR, límite por posición, máximo de posiciones abiertas y un *kill switch* por pérdida diaria.
- **Dashboard de forecasting** (Streamlit): ranking del escáner, gráfico de velas con cono de forecast, curva de capital y posiciones del bot, y backtests interactivos.
- **Backtest walk-forward**: reentrena el modelo periódicamente usando solo el pasado y aplica las mismas reglas que el bot en vivo.

## Instalación

```bash
cd crypto-ai-trader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py scan                    # escanea las 20 cryptos del config y muestra el ranking
python main.py scan --top 50           # escanea automáticamente los 50 pares con más volumen
python main.py backtest                # backtest walk-forward de todo el universo
python main.py bot --once              # un ciclo del bot (paper)
python main.py bot                     # bot continuo (cada 5 min por defecto)
python main.py dashboard               # dashboard en http://localhost:8501
```

Añade `--synthetic` a cualquier comando para probar sin conexión con datos simulados.

Recomendado: deja el bot corriendo en una terminal y el dashboard en otra. Ambos comparten la base de datos `data/trader.db`.

## Arquitectura

```
config.yaml           parámetros (exchange, universo, modelo, estrategia, riesgo, ejecución)
main.py               CLI
trader/
  data.py             descarga OHLCV concurrente (ccxt async), universo top-N, datos sintéticos
  features.py         ~30 indicadores causales (retornos, volatilidad, RSI, MACD, EMAs, Bollinger, ATR...)
  model.py            clasificador + regresores cuantílicos, validación con purga, registro/cache de modelos
  scanner.py          entrena en paralelo y rankea todas las cryptos
  strategy.py         reglas BUY / HOLD / EXIT (compartidas por el bot y el backtest)
  risk.py             tamaño de posición, stops, TP, trailing, kill switch
  broker.py           PaperBroker y LiveBroker (órdenes de mercado vía ccxt)
  bot.py              ciclo del bot
  backtest.py         backtest walk-forward con comisiones y slippage
  storage.py          SQLite: señales, posiciones, operaciones, capital
dashboard/app.py      dashboard Streamlit
tests/                tests de extremo a extremo (pytest)
```

## Trading real (bajo tu responsabilidad)

1. Prueba primero en **testnet** con `config.testnet.yaml` (usa `data/testnet.db`) y claves de testnet. Para dinero real se usa `config.live.yaml` (usa `data/live.db`).
2. En `config.live.yaml` define `execution.max_live_capital`: el capital máximo que el bot puede usar. Con 0 el modo live no arranca. Cada cuenta necesita su propio `storage.db_path`: el bot no arranca con una base de datos de otra cuenta (paper, testnet o real).
3. Exporta tus credenciales (con permiso de trading y **sin** permiso de retiro):

```bash
export EXCHANGE_API_KEY=...
export EXCHANGE_API_SECRET=...
export CONFIRM_LIVE_TRADING=YES
```

4. Ejecuta `python main.py bot --config config.testnet.yaml` (o `config.live.yaml`).

Protecciones del modo live: cada posición tiene una orden stop-limit en el exchange, que sigue activa aunque el bot esté detenido y sube con el trailing stop. El bot compara además stops y take profits con el precio actual cada `bot.stop_check_seconds` y vende a mercado si hace falta. Si se supera `risk.max_daily_loss_pct` (día UTC), cierra todas las posiciones y no abre nuevas ese día. En modo live el bot no arranca con datos sintéticos.

Limitaciones: solo opera en spot y solo en largo. Un stop-limit no garantiza la venta: si el precio cae por debajo del límite (`execution.stop_limit_buffer`), la orden queda sin ejecutar hasta que el bot, si está en marcha, venda a mercado. Con el bot detenido no funcionan el take profit, el trailing stop, la salida por tiempo ni el límite de pérdida diaria.

## Aviso importante

Esto es software experimental, no asesoría financiera. Los mercados de criptomonedas son en gran parte impredecibles y la mayoría de los modelos que funcionan en un backtest no funcionan en vivo. El sistema está diseñado para ser conservador: solo opera los símbolos donde el modelo demostró ventaja fuera de muestra, y en muchas condiciones de mercado no abrirá ninguna operación. Eso es intencional. Usa paper trading durante semanas antes de arriesgar dinero, y nunca arriesgues lo que no puedas perder.
