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

1. Prueba primero en **testnet**: pon `exchange.sandbox: true` en el config y usa claves de testnet.
2. Pon `execution.mode: live` en `config.yaml`.
3. Exporta tus credenciales (con permiso de trading y **sin** permiso de retiro):

```bash
export EXCHANGE_API_KEY=...
export EXCHANGE_API_SECRET=...
export CONFIRM_LIVE_TRADING=YES
```

4. Ejecuta `python main.py bot`.

Limitaciones del modo live: solo opera en spot y solo en largo. Los stops y take profits los vigila el bot en cada ciclo y los ejecuta a mercado; no se colocan como órdenes en el exchange, así que si el bot está detenido nada protege tus posiciones. En modo live el bot nunca opera con datos sintéticos.

## Aviso importante

Esto es software experimental, no asesoría financiera. Los mercados de criptomonedas son en gran parte impredecibles y la mayoría de los modelos que funcionan en un backtest no funcionan en vivo. El sistema está diseñado para ser conservador: solo opera los símbolos donde el modelo demostró ventaja fuera de muestra, y en muchas condiciones de mercado no abrirá ninguna operación. Eso es intencional. Usa paper trading durante semanas antes de arriesgar dinero, y nunca arriesgues lo que no puedas perder.
