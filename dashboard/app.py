"""Forecasting Dashboard — streamlit run dashboard/app.py [-- --synthetic]"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from trader.backtest import backtest_many  # noqa: E402
from trader.bot import TradingBot, utcnow  # noqa: E402
from trader.config import load_config, override  # noqa: E402
from trader.data import MarketData, tf_seconds  # noqa: E402
from trader.features import build_features  # noqa: E402
from trader.model import ModelRegistry, forecast_path  # noqa: E402
from trader.scanner import Scanner  # noqa: E402
from trader.storage import Storage  # noqa: E402

UP, DOWN, NEUTRAL, ACCENT = "#16a34a", "#dc2626", "#94a3b8", "#2563eb"
st.set_page_config(page_title="Crypto AI Trader", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- setup
base_cfg = load_config()
st.sidebar.title("📈 Crypto AI Trader")
synthetic = st.sidebar.toggle("Datos sintéticos (offline)", value="--synthetic" in sys.argv or base_cfg["data"]["source"] == "synthetic")
cfg = override(base_cfg, data__source="synthetic") if synthetic else base_cfg
st.sidebar.caption(f"Exchange: **{cfg['exchange']['id']}** · TF: **{cfg['data']['timeframe']}** · "
                   f"Horizonte: **{cfg['model']['horizon']} velas** · Modo: **{cfg['execution']['mode'].upper()}**")
if synthetic:
    st.sidebar.warning("Usando datos SIMULADOS: útil para probar el sistema, no para decidir operaciones.")


@st.cache_resource
def resources(source: str):
    c = override(base_cfg, data__source=source)
    reg = ModelRegistry(c)
    return MarketData(c), Scanner(c, reg), reg, Storage(c["storage"]["db_path"])


md, scanner, registry, storage = resources(cfg["data"]["source"])


@st.cache_data(ttl=300, show_spinner="Descargando velas...")
def get_frames(symbols: tuple, source: str):
    return md.fetch_many(list(symbols))


universe = tuple(md.resolve_universe())
st.sidebar.write(f"Universo: **{len(universe)} pares**")
if st.sidebar.button("🔄 Escanear ahora", use_container_width=True, type="primary"):
    with st.spinner(f"Escaneando {len(universe)} cryptos (entrenando modelos si hace falta)..."):
        get_frames.clear()
        frames, sources = get_frames(universe, cfg["data"]["source"])
        storage.save_scan(scanner.scan(frames, sources), utcnow())
    st.sidebar.success("Escaneo completado")

tab_scan, tab_fc, tab_bot, tab_bt = st.tabs(["🔎 Escáner", "📈 Forecast", "🤖 Bot / Portafolio", "🧪 Backtest"])

# ---------------------------------------------------------------- escáner
with tab_scan:
    scan = storage.latest_scan()
    if scan.empty:
        st.info("Aún no hay escaneos. Pulsa **Escanear ahora** en la barra lateral (o ejecuta `python main.py bot`).")
    else:
        st.caption(f"Último escaneo: {scan['scan_ts'].iloc[0]} · fuente: {', '.join(scan['source'].unique())}")
        c = st.columns(5)
        c[0].metric("Cryptos escaneadas", len(scan))
        c[1].metric("Señales BUY", int((scan["signal"] == "BUY").sum()))
        c[2].metric("Señales EXIT", int((scan["signal"] == "EXIT").sum()))
        c[3].metric("Modelos operables", int(scan["tradable"].sum()), help="AUC ≥ mínimo y ventaja positiva fuera de muestra")
        c[4].metric("Prob. media de subida", f"{scan['prob_up'].mean():.1%}")

        fig = go.Figure(go.Bar(
            x=scan["symbol"], y=scan["prob_up"],
            marker_color=[UP if s == "BUY" else DOWN if s == "EXIT" else NEUTRAL for s in scan["signal"]],
            hovertemplate="%{x}<br>P(sube)=%{y:.1%}<extra></extra>"))
        fig.add_hline(y=cfg["strategy"]["buy_prob"], line_dash="dot", annotation_text="umbral compra")
        fig.add_hline(y=cfg["strategy"]["exit_prob"], line_dash="dot", annotation_text="umbral salida")
        fig.update_layout(height=320, margin=dict(t=20, b=10), yaxis_tickformat=".0%", yaxis_title="P(subida)")
        st.plotly_chart(fig, use_container_width=True)

        only = st.multiselect("Filtrar señal", ["BUY", "HOLD", "EXIT"], default=["BUY", "HOLD", "EXIT"])
        view = scan[scan["signal"].isin(only)].copy()
        view["exp_ret"] = (np.exp(view["q50"]) - 1) * 100
        view["change_24h"] *= 100
        view["atr_pct"] *= 100
        view["holdout_edge"] *= 100
        view["tradable"] = view["tradable"].astype(bool)
        st.dataframe(
            view[["symbol", "signal", "score", "price", "prob_up", "exp_ret", "target_price", "change_24h",
                  "atr_pct", "auc", "hit_rate", "holdout_edge", "tradable", "quote_volume_24h"]],
            use_container_width=True, hide_index=True,
            column_config={
                "prob_up": st.column_config.ProgressColumn("P(sube)", format="%.2f", min_value=0, max_value=1),
                "exp_ret": st.column_config.NumberColumn("Ret. esperado", format="%.2f%%"),
                "change_24h": st.column_config.NumberColumn("24h", format="%.2f%%"),
                "atr_pct": st.column_config.NumberColumn("ATR %", format="%.2f%%"),
                "price": st.column_config.NumberColumn("Precio", format="%.6g"),
                "target_price": st.column_config.NumberColumn("Objetivo q50", format="%.6g"),
                "holdout_edge": st.column_config.NumberColumn("Edge holdout", format="%.2f%%"),
                "hit_rate": st.column_config.NumberColumn("Hit rate", format="%.2f"),
                "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
                "quote_volume_24h": st.column_config.NumberColumn("Vol. 24h", format="%.3e"),
            })

# ---------------------------------------------------------------- forecast
with tab_fc:
    sym = st.selectbox("Crypto", universe, index=0)
    frames, sources = get_frames(universe, cfg["data"]["source"])
    df = frames.get(sym)
    if df is None:
        st.error(f"Sin datos para {sym}")
    else:
        with st.spinner("Cargando/entrenando modelo..."):
            model = registry.get(sym, df)
        X = build_features(df)
        last = model.predict_frame(X.iloc[[-1]]).iloc[0]
        price = float(df["close"].iloc[-1])
        H = model.horizon
        path = forecast_path(df.index[-1], price, last, H, tf_seconds(cfg["data"]["timeframe"]))
        hist = model.predict_frame(X.iloc[-300:])
        m = model.metrics

        c = st.columns(6)
        c[0].metric("Precio", f"{price:,.6g}", f"{df['close'].iloc[-1] / df['close'].iloc[-25] - 1:+.2%} (24 velas)")
        c[1].metric(f"P(sube) en {H} velas", f"{last['prob_up']:.1%}")
        c[2].metric("Forecast q50", f"{price * np.exp(last['q50']):,.6g}", f"{np.exp(last['q50']) - 1:+.2%}")
        c[3].metric("Rango 80%", f"{np.exp(last['q10']) - 1:+.1%} / {np.exp(last['q90']) - 1:+.1%}")
        c[4].metric("AUC fuera de muestra", f"{m['auc']:.3f}", help="0.5 = azar")
        c[5].metric("Operable", "Sí ✅" if m["tradable"] else "No ⛔")

        view = df.iloc[-200:]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=view.index, open=view.open, high=view.high, low=view.low, close=view.close,
                                     name=sym, increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
        for n, color in ((21, "#f59e0b"), (50, "#8b5cf6")):
            fig.add_trace(go.Scatter(x=view.index, y=df["close"].ewm(span=n, adjust=False).mean().iloc[-200:],
                                     name=f"EMA{n}", line=dict(width=1, color=color)), 1, 1)
        fig.add_trace(go.Scatter(x=path.index, y=path["high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=path.index, y=path["low"], mode="lines", fill="tonexty", fillcolor="rgba(37,99,235,0.18)",
                                 line=dict(width=0), name="Banda 10–90%"), 1, 1)
        fig.add_trace(go.Scatter(x=path.index, y=path["median"], mode="lines", name="Forecast mediana",
                                 line=dict(color=ACCENT, width=2, dash="dash")), 1, 1)
        hv = hist.iloc[-200:]
        fig.add_trace(go.Scatter(x=hv.index, y=hv["prob_up"], name="P(sube)", line=dict(color=ACCENT)), 2, 1)
        fig.add_hline(y=cfg["strategy"]["buy_prob"], line_dash="dot", line_color=UP, row=2, col=1)
        fig.add_hline(y=cfg["strategy"]["exit_prob"], line_dash="dot", line_color=DOWN, row=2, col=1)
        fig.add_trace(go.Bar(x=view.index, y=view.volume, name="Volumen", marker_color=NEUTRAL), 3, 1)
        fig.update_layout(height=720, xaxis_rangeslider_visible=False, margin=dict(t=20, b=10),
                          legend=dict(orientation="h", y=1.02))
        fig.update_yaxes(title_text="P(sube)", tickformat=".0%", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("La serie P(sube) histórica se calcula con el modelo actual (incluye datos de entrenamiento): "
                   "úsala como contexto visual. Las métricas de calidad (AUC, hit rate, edge) son fuera de muestra.")
        with st.expander("Métricas del modelo (holdout)"):
            st.json(m)
        if st.button("Reentrenar este modelo"):
            registry.get(sym, df, force=True)
            st.rerun()

# ---------------------------------------------------------------- bot
with tab_bot:
    st.write(f"Modo de ejecución: **{cfg['execution']['mode'].upper()}** — para correr el bot de forma continua: "
             f"`python main.py bot{' --synthetic' if synthetic else ''}`")
    if st.button("▶️ Ejecutar un ciclo del bot ahora"):
        with st.spinner("Ejecutando ciclo..."):
            try:
                st.success(TradingBot(cfg).run_cycle())
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    eq = storage.equity_curve()
    positions = storage.get_positions()
    trades = storage.trades()
    if not eq.empty:
        start = cfg["risk"]["starting_equity"]
        cur = eq["equity"].iloc[-1]
        closed = trades[trades["side"] == "SELL"]
        c = st.columns(4)
        c[0].metric("Capital", f"{cur:,.2f}", f"{cur / start - 1:+.2%}")
        c[1].metric("Efectivo", f"{eq['cash'].iloc[-1]:,.2f}")
        c[2].metric("Posiciones abiertas", len(positions))
        c[3].metric("Win rate", f"{(closed['pnl'] > 0).mean():.0%}" if len(closed) else "—", f"{len(closed)} cerradas")
        fig = go.Figure(go.Scatter(x=pd.to_datetime(eq["ts"]), y=eq["equity"], mode="lines+markers",
                                   line=dict(color=ACCENT)))
        fig.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="Capital")
        st.plotly_chart(fig, use_container_width=True)
    if positions:
        latest = storage.latest_scan().set_index("symbol")
        rows = []
        for s, p in positions.items():
            px = float(latest.loc[s, "price"]) if s in latest.index else p.entry_price
            rows.append({"symbol": s, "qty": p.qty, "entrada": p.entry_price, "actual": px, "stop": p.stop,
                         "take_profit": p.take_profit, "PnL no realizado": (px - p.entry_price) * p.qty - p.entry_fee,
                         "velas": p.bars_held, "desde": p.entry_ts})
        st.subheader("Posiciones abiertas")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.subheader("Historial de operaciones")
    st.dataframe(trades, hide_index=True, use_container_width=True) if not trades.empty else st.caption("Sin operaciones todavía.")

# ---------------------------------------------------------------- backtest
with tab_bt:
    st.write("Backtest **walk-forward**: el modelo se reentrena cada "
             f"{cfg['backtest']['retrain_every']} velas usando solo el pasado, con comisiones, slippage y las mismas reglas de riesgo.")
    pick = st.multiselect("Cryptos", universe, default=list(universe[:4]))
    if st.button("Ejecutar backtest", disabled=not pick):
        frames, _ = get_frames(universe, cfg["data"]["source"])
        with st.spinner(f"Backtesting {len(pick)} símbolos..."):
            table, results = backtest_many({s: frames[s] for s in pick if s in frames}, cfg)
        st.session_state["bt"] = (table, results)
    if "bt" in st.session_state:
        table, results = st.session_state["bt"]
        pct = st.column_config.NumberColumn(format="%.2f%%")
        shown = table.copy()
        for col in ("total_return", "buy_hold", "max_drawdown", "win_rate"):
            shown[col] = shown[col].astype(float) * 100
        st.dataframe(shown, hide_index=True, use_container_width=True,
                     column_config={"total_return": pct, "buy_hold": pct, "max_drawdown": pct, "win_rate": pct})
        fig = go.Figure()
        for s, r in results.items():
            fig.add_trace(go.Scatter(x=r["equity"].index, y=r["equity"] / r["equity"].iloc[0] - 1, name=s))
        fig.update_layout(height=380, yaxis_tickformat=".1%", yaxis_title="Retorno estrategia", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
