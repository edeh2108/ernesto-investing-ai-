# -*- coding: utf-8 -*-
"""
InvestAI — App Streamlit (Bono Sem. 13, iDeSo · UNMSM-FISI)
Lee directamente de MongoDB Atlas (mismas colecciones que la API FastAPI):
    precios_ohlcv, predicciones, predicciones_rnn, predicciones_lstm,
    sentimiento_resumen, sentimiento_noticias

No usa fetch()/API: pymongo se conecta directo a Atlas.
Requiere el secret MONGO_URI configurado en Streamlit Cloud
(Settings -> Secrets), formato:

    MONGO_URI = "mongodb+srv://usuario:password@cluster.mongodb.net/"
"""

import math
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# ─────────────────────────────────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InvestAI — Streamlit",
    page_icon="📈",
    layout="wide",
)

TICKERS = ["FSM", "VOLCABC1.LM", "ABX.TO", "BVN", "BHP"]
EMPRESAS = {
    "FSM": "Fortuna Silver Mines Inc.",
    "VOLCABC1.LM": "Volcan Compañía Minera S.A.A.",
    "ABX.TO": "Barrick Gold Corporation",
    "BVN": "Compañía de Minas Buenaventura S.A.A.",
    "BHP": "BHP Group Limited",
}

DB_NOMBRE = "investai"
COL_PRECIOS = "precios_ohlcv"
COL_SVC = "predicciones"
COL_RNN = "predicciones_rnn"
COL_LSTM = "predicciones_lstm"
COL_SENT_RES = "sentimiento_resumen"
COL_SENT_NOT = "sentimiento_noticias"

EXCLUIR = {"_id": 0}


# ─────────────────────────────────────────────────────────────────────────
# Conexión a MongoDB (cacheada — una sola conexión por sesión de servidor)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def obtener_cliente():
    try:
        uri = st.secrets["MONGO_URI"]
    except Exception:
        st.error(
            "⚠️ No se encontró el secret **MONGO_URI**. "
            "Configúralo en Streamlit Cloud → Settings → Secrets:\n\n"
            '`MONGO_URI = "mongodb+srv://usuario:password@cluster.mongodb.net/"`'
        )
        st.stop()
    try:
        cliente = MongoClient(uri, serverSelectionTimeoutMS=8000)
        cliente.admin.command("ping")
        return cliente
    except ConnectionFailure as e:
        st.error(f"❌ No se pudo conectar a MongoDB Atlas: {e}")
        st.stop()


cliente = obtener_cliente()
db = cliente[DB_NOMBRE]
col_precios = db[COL_PRECIOS]
col_svc = db[COL_SVC]
col_rnn = db[COL_RNN]
col_lstm = db[COL_LSTM]
col_sent_res = db[COL_SENT_RES]
col_sent_not = db[COL_SENT_NOT]


# ─────────────────────────────────────────────────────────────────────────
# Helpers de lectura (cacheados 60s para no golpear MongoDB en cada rerun)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def leer_mercado(ticker: str, ultimos: int | None = None):
    docs = list(col_precios.find({"ticker": ticker}, EXCLUIR).sort("fecha", 1))
    if not docs:
        return None
    df = pd.DataFrame(docs)
    df["fecha"] = pd.to_datetime(df["fecha"])
    if ultimos:
        df = df.tail(ultimos)
    return df.reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def leer_svc(ticker: str):
    return col_svc.find_one({"ticker": ticker, "modelo": "SVC"}, EXCLUIR)


@st.cache_data(ttl=60, show_spinner=False)
def leer_rnn(ticker: str):
    docs = list(col_rnn.find({"ticker": ticker}, EXCLUIR).sort("arquitectura", 1))
    return {d["arquitectura"]: d for d in docs}


@st.cache_data(ttl=60, show_spinner=False)
def leer_lstm(ticker: str):
    return col_lstm.find_one({"ticker": ticker}, EXCLUIR)


@st.cache_data(ttl=60, show_spinner=False)
def leer_sentimiento(ticker: str):
    return col_sent_res.find_one({"ticker": ticker}, EXCLUIR)


@st.cache_data(ttl=60, show_spinner=False)
def leer_noticias(ticker: str, limite: int = 15):
    return list(
        col_sent_not.find({"ticker": ticker}, EXCLUIR)
        .sort("fecha_publicacion", -1)
        .limit(limite)
    )


def sent_a_senal(s):
    return {"POSITIVO": "BUY", "NEGATIVO": "SELL"}.get(s, "HOLD")


def color_senal(s):
    return {"BUY": "🟢", "SELL": "🔴"}.get(s, "🟡")


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📈 Invest**AI**")
st.sidebar.caption("UNMSM · FISI · iDeSo 2026-II · Grupo 10")
ticker = st.sidebar.selectbox(
    "Ticker minero",
    TICKERS,
    format_func=lambda t: f"{t} — {EMPRESAS.get(t, t)}",
)
pagina = st.sidebar.radio(
    "Módulo",
    [
        "🏠 Dashboard",
        "📊 Mercado",
        "🤖 Clasificador SVC",
        "🧠 Consola RNN",
        "📉 Regresor LSTM",
        "💬 Sentimiento NLP",
        "⚡ Señales Broker (5 tickers)",
        "💼 Portafolio simulado",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Fuente: Yahoo Finance → MongoDB Atlas (lectura directa, sin API intermedia)")


# ─────────────────────────────────────────────────────────────────────────
# Página: Dashboard consolidado
# ─────────────────────────────────────────────────────────────────────────
if pagina == "🏠 Dashboard":
    st.title(f"🏠 Dashboard — {ticker}")
    st.caption(EMPRESAS.get(ticker, ticker))

    mer = leer_mercado(ticker)
    svc = leer_svc(ticker)
    rnn = leer_rnn(ticker)
    lstm = leer_lstm(ticker)
    sent = leer_sentimiento(ticker)

    if mer is None:
        st.warning("Sin datos en MongoDB para este ticker. Ejecuta el Notebook 1 (Ingesta).")
    else:
        ultimo = mer.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precio actual", f"${ultimo['cierre']:.4f}")
        c2.metric(
            "Señal SVC",
            svc.get("señal", "—") if svc else "—",
            f"{(svc.get('confianza', 0) * 100):.1f}% conf." if svc else "",
        )
        rnn_sen = [d.get("senal") for d in rnn.values()] if rnn else []
        buy_r = rnn_sen.count("BUY")
        consenso_rnn = "BUY" if buy_r > len(rnn_sen) / 2 else "SELL" if rnn_sen else "—"
        c3.metric("Consenso RNN", consenso_rnn)
        if lstm:
            pf7 = lstm.get("predicciones_futuras", {}).get("7", {})
            c4.metric("LSTM +7d", f"${pf7.get('precio_final', 0):.4f}" if pf7 else "—")
        else:
            c4.metric("LSTM +7d", "—")
        c5.metric("Sentimiento", sent.get("sentimiento_global", "—") if sent else "—")

        # Candlestick 90 días
        s = mer.tail(90)
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=s["fecha"], open=s["apertura"], high=s["maximo"],
                    low=s["minimo"], close=s["cierre"], name=ticker,
                    increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                ),
                go.Scatter(x=s["fecha"], y=s.get("sma_20"), name="SMA-20",
                           line=dict(color="#f59e0b", width=1.3)),
            ]
        )
        fig.update_layout(
            template="plotly_dark", height=380, margin=dict(t=20, b=20, l=40, r=20),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Señal consolidada (voto mayoritario)
        votos = [svc.get("señal") if svc else None, consenso_rnn if consenso_rnn != "—" else None,
                 sent_a_senal(sent.get("sentimiento_global")) if sent else None]
        votos = [v for v in votos if v]
        n_buy = votos.count("BUY")
        n_sell = votos.count("SELL")
        estrategia = "BUY" if n_buy > n_sell else "SELL" if n_sell > n_buy else "HOLD"
        st.subheader(f"Señal consolidada del sistema: {color_senal(estrategia)} **{estrategia}**")
        st.caption(f"{n_buy} BUY · {n_sell} SELL de {len(votos)} modelos con señal disponible")


# ─────────────────────────────────────────────────────────────────────────
# Página: Mercado
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "📊 Mercado":
    st.title(f"📊 Visualización de Mercado — {ticker}")
    mer = leer_mercado(ticker)
    if mer is None:
        st.warning("Sin datos. Ejecuta el Notebook 1 (Ingesta) primero.")
    else:
        u = mer.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Último cierre", f"${u['cierre']:.4f}")
        c2.metric("RSI-14", f"{u.get('rsi_14', 0):.2f}" if pd.notna(u.get("rsi_14")) else "—")
        c3.metric("SMA-20", f"${u.get('sma_20', 0):.4f}" if pd.notna(u.get("sma_20")) else "—")
        c4.metric("SMA-50", f"${u.get('sma_50', 0):.4f}" if pd.notna(u.get("sma_50")) else "—")
        c5.metric("EMA-12", f"${u.get('ema_12', 0):.4f}" if pd.notna(u.get("ema_12")) else "—")

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=mer["fecha"], open=mer["apertura"], high=mer["maximo"],
                    low=mer["minimo"], close=mer["cierre"], name=ticker,
                    increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                ),
                go.Scatter(x=mer["fecha"], y=mer.get("sma_20"), name="SMA-20",
                           line=dict(color="#f59e0b", width=1.3)),
                go.Scatter(x=mer["fecha"], y=mer.get("sma_50"), name="SMA-50",
                           line=dict(color="#3b82f6", width=1.3, dash="dot")),
                go.Scatter(x=mer["fecha"], y=mer.get("ema_12"), name="EMA-12",
                           line=dict(color="#a78bfa", width=1.1)),
            ]
        )
        fig.update_layout(template="plotly_dark", height=460, xaxis_rangeslider_visible=False,
                           margin=dict(t=20, b=20, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)

        fig_rsi = go.Figure(go.Scatter(x=mer["fecha"], y=mer.get("rsi_14"),
                                        line=dict(color="#f59e0b")))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="#ef4444")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="#22c55e")
        fig_rsi.update_layout(template="plotly_dark", height=200, title="RSI-14",
                               margin=dict(t=30, b=20, l=40, r=20))
        st.plotly_chart(fig_rsi, use_container_width=True)

        with st.expander("📋 Últimos 15 registros"):
            st.dataframe(
                mer.tail(15).sort_values("fecha", ascending=False)[
                    ["fecha", "apertura", "maximo", "minimo", "cierre", "volumen", "sma_20", "rsi_14"]
                ],
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────
# Página: Clasificador SVC
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "🤖 Clasificador SVC":
    st.title(f"🤖 Clasificador SVC — {ticker}")
    doc = leer_svc(ticker)
    if not doc:
        st.warning("Sin predicción SVC. Ejecuta el Notebook 2.")
    else:
        senal = doc.get("señal", "—")
        conf = doc.get("confianza", 0) or 0
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Señal vigente", f"{color_senal(senal)} {senal}")
            st.progress(min(max(conf, 0), 1), text=f"Confianza: {conf*100:.1f}%")
        with c2:
            hp = doc.get("hiperparametros", {})
            st.write("**Hiperparámetros (GridSearchCV):**")
            st.json(hp)
        m = doc.get("metricas", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{(m.get('accuracy') or 0)*100:.1f}%")
        c2.metric("Precision", f"{(m.get('precision') or 0)*100:.1f}%")
        c3.metric("Recall", f"{(m.get('recall') or 0)*100:.1f}%")
        c4.metric("F1-score", f"{(m.get('f1') or 0)*100:.1f}%")
        st.caption(f"Último cierre: {doc.get('ultimo_cierre')} · Fecha ref.: {doc.get('fecha_referencia')}")


# ─────────────────────────────────────────────────────────────────────────
# Página: Consola RNN
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "🧠 Consola RNN":
    st.title(f"🧠 Consola de Modelos RNN — {ticker}")
    modelos = leer_rnn(ticker)
    if not modelos:
        st.warning("Sin predicciones RNN. Ejecuta el Notebook 3.")
    else:
        cols = st.columns(3)
        for col, arq in zip(cols, ["SimpleRNN", "GRU", "LSTM"]):
            d = modelos.get(arq, {})
            senal = d.get("senal", "—")
            with col:
                st.markdown(f"### {arq}")
                st.markdown(f"## {color_senal(senal)} {senal}")
                st.progress(min(max(d.get("confianza", 0) or 0, 0), 1))
                st.caption(f"Épocas: {d.get('epocas_reales', '—')}")

        filas = [
            {"Arquitectura": a, "Señal": d.get("senal"), "Confianza": d.get("confianza")}
            for a, d in modelos.items()
        ]
        st.dataframe(pd.DataFrame(filas), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# Página: Regresor LSTM
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "📉 Regresor LSTM":
    st.title(f"📉 Regresor LSTM — {ticker}")
    doc = leer_lstm(ticker)
    if not doc:
        st.warning("Sin predicciones LSTM. Ejecuta el Notebook 4.")
    else:
        m = doc.get("metricas", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Último precio", f"${doc.get('ultimo_precio', 0):.4f}")
        c2.metric("RMSE", f"${m.get('rmse_usd', 0):.4f}")
        c3.metric("MAE", f"${m.get('mae_usd', 0):.4f}")
        c4.metric("R²", f"{m.get('r2', 0):.4f}")

        horizonte = st.radio("Horizonte", [7, 14, 30, 60], horizontal=True)
        pf = doc.get("predicciones_futuras", {}).get(str(horizonte))
        if pf:
            fechas = [doc.get("fecha_ultimo")] + pf.get("fechas", [])
            precios = [doc.get("ultimo_precio")] + pf.get("predicciones", [])
            bsup = [doc.get("ultimo_precio")] + pf.get("banda_superior", [])
            binf = [doc.get("ultimo_precio")] + pf.get("banda_inferior", [])
            fig = go.Figure([
                go.Scatter(x=fechas, y=bsup, line=dict(width=0), showlegend=False),
                go.Scatter(x=fechas, y=binf, fill="tonexty", fillcolor="rgba(31,224,196,0.08)",
                           line=dict(width=0), name="Banda 95%"),
                go.Scatter(x=fechas, y=precios, name="Predicción LSTM",
                           line=dict(color="#1fe0c4", width=2.5), mode="lines+markers"),
            ])
            fig.update_layout(template="plotly_dark", height=420, margin=dict(t=20, b=20, l=40, r=20))
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# Página: Sentimiento NLP
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "💬 Sentimiento NLP":
    st.title(f"💬 Análisis de Sentimiento NLP — {ticker}")
    resumen = leer_sentimiento(ticker)
    noticias = leer_noticias(ticker)

    if not resumen:
        st.warning("Sin resumen de sentimiento. Ejecuta el Notebook 5.")
    else:
        sent = resumen.get("sentimiento_global", "—")
        comp = resumen.get("compound_promedio", 0) or 0
        dist = resumen.get("distribucion", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentimiento global", sent)
        c2.metric("Compound promedio", f"{comp:+.4f}")
        c3.metric("Noticias analizadas", resumen.get("n_noticias", 0))
        c4.metric("Positivas / Negativas", f"{dist.get('positivo', 0)} / {dist.get('negativo', 0)}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=comp,
            gauge={"axis": {"range": [-1, 1]},
                   "bar": {"color": "#1fe0c4" if comp >= 0 else "#ef4444"}},
        ))
        fig.update_layout(template="plotly_dark", height=250, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📰 Noticias recientes con score VADER")
    if not noticias:
        st.info("Sin noticias almacenadas para este ticker.")
    for n in noticias:
        comp = n.get("vader_compound", 0) or 0
        emoji = "🟢" if comp >= 0.05 else "🔴" if comp <= -0.05 else "🟡"
        st.markdown(
            f"{emoji} **{n.get('titulo', 'Sin título')}**  \n"
            f"<span style='color:gray;font-size:0.85em'>{n.get('publisher','—')} · "
            f"{n.get('fecha_pub_str','—')} · compound: {comp:+.4f}</span>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# Página: Señales Broker (todos los tickers)
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "⚡ Señales Broker (5 tickers)":
    st.title("⚡ Panel de Señales — Todos los Tickers")
    filas = []
    for t in TICKERS:
        svc = leer_svc(t)
        rnn = leer_rnn(t)
        sent = leer_sentimiento(t)
        mer = leer_mercado(t, ultimos=1)
        precio = mer.iloc[-1]["cierre"] if mer is not None and not mer.empty else None
        rnn_sen = [d.get("senal") for d in rnn.values()] if rnn else []
        consenso_rnn = "BUY" if rnn_sen.count("BUY") > len(rnn_sen) / 2 else ("SELL" if rnn_sen else "—")
        filas.append({
            "Ticker": t,
            "Precio": f"${precio:.4f}" if precio else "—",
            "SVC": svc.get("señal", "—") if svc else "—",
            "Conf. SVC": f"{(svc.get('confianza',0) or 0)*100:.1f}%" if svc else "—",
            "RNN consenso": consenso_rnn,
            "NLP": sent_a_senal(sent.get("sentimiento_global")) if sent else "—",
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────
# Página: Portafolio simulado
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "💼 Portafolio simulado":
    st.title("💼 Gestión de Portafolio Simulado ($100,000)")
    pesos = {"FSM": 20000, "VOLCABC1.LM": 15000, "ABX.TO": 25000, "BVN": 25000, "BHP": 15000}
    filas = []
    for t, val in pesos.items():
        svc = leer_svc(t)
        senal = svc.get("señal", "—") if svc else "—"
        filas.append({"Ticker": t, "Empresa": EMPRESAS[t], "Señal SVC": senal, "Valor USD": val})
    df = pd.DataFrame(filas)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=df["Ticker"], values=df["Valor USD"], hole=0.5))
        fig.update_layout(template="plotly_dark", height=320, title="Distribución del portafolio")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(df, use_container_width=True, hide_index=True)


st.sidebar.divider()
st.sidebar.caption(f"Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
