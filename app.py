# -*- coding: utf-8 -*-
"""
InvestAI — App Streamlit AUTÓNOMA (Bono Sem. 13, iDeSo · UNMSM-FISI)

A diferencia de la versión anterior, esta app NO depende de Colab/ngrok/
notebooks: ella misma hace todo el pipeline end-to-end cuando el usuario
lo solicita:

    yfinance (descarga) -> indicadores técnicos -> SVC / RNN / LSTM (TF)
    -> VADER (sentimiento) -> MongoDB Atlas (persistencia)
    -> Streamlit (visualización)

Reglas de rendimiento (Streamlit Community Cloud, free tier ~1 CPU/1GB RAM):
  - La ingesta OHLCV + indicadores + SVC + NLP son ligeros -> se ejecutan
    automáticamente (con caché por antigüedad) para el ticker activo, y
    también para los 5 tickers en las páginas "Broker" y "Portafolio".
  - Los modelos RNN (SimpleRNN/GRU/LSTM clasificadores) y el Regresor LSTM
    usan TensorFlow y SOLO se entrenan para el ticker seleccionado, y SOLO
    cuando el usuario pulsa el botón correspondiente (nunca para los 5
    tickers de golpe).

Requiere el secret MONGO_URI en Streamlit Cloud (Settings -> Secrets):
    MONGO_URI = "mongodb+srv://usuario:password@cluster.mongodb.net/"
"""

import math
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, ConfigurationError

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Config general
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="InvestAI — Streamlit (Autónomo)", page_icon="📈", layout="wide")

TICKERS = ["FSM", "VOLCABC1.LM", "ABX.TO", "BVN", "BHP"]
EMPRESAS_META = {
    "FSM":         {"nombre": "Fortuna Silver Mines Inc.",              "moneda": "USD"},
    "VOLCABC1.LM": {"nombre": "Volcan Compañía Minera S.A.A.",          "moneda": "PEN"},
    "ABX.TO":      {"nombre": "Barrick Gold Corporation",               "moneda": "CAD"},
    "BVN":         {"nombre": "Compañía de Minas Buenaventura S.A.A.",  "moneda": "USD"},
    "BHP":         {"nombre": "BHP Group Limited",                      "moneda": "USD"},
}

PERIOD = "1y"
AUTO_ADJUST = True
SMA_VENTANAS = [20, 50]
EMA_VENTANAS = [12, 26]
RSI_PERIODO = 14

FRESCURA_MERCADO_H = 6      # re-descargar OHLCV si el último dato tiene más de 6h
FRESCURA_NLP_H = 3          # re-analizar noticias si el resumen tiene más de 3h
MIN_REGISTROS_SVC = 80
UMBRAL_POS, UMBRAL_NEG = 0.05, -0.05

# Hiperparámetros TF reducidos respecto a los notebooks originales,
# para que el entrenamiento tome ~20-60s en el free tier (1 ticker a la vez)
VENTANA_RNN = 20
EPOCHS_RNN = 30
PATIENCE_RNN = 7
VENTANA_LSTM = 60
HORIZONTES_LSTM = [7, 14, 30, 60]
EPOCHS_LSTM = 60
PATIENCE_LSTM = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DB_NOMBRE = "investai"
COL_PRECIOS = "precios_ohlcv"
COL_SVC = "predicciones"
COL_RNN = "predicciones_rnn"
COL_LSTM = "predicciones_lstm"
COL_SENT_RES = "sentimiento_resumen"
COL_SENT_NOT = "sentimiento_noticias"
COL_METRICAS = "metricas_modelos"
EXCLUIR = {"_id": 0}


# ─────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────
def nan_a_none(v):
    if v is None:
        return None
    try:
        v = float(v)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
    except (TypeError, ValueError):
        return None


def _forzar_dns_publico():
    """Evita ConfigurationError resolviendo el SRV con DNS públicos."""
    try:
        import dns.resolver
        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = ["8.8.8.8", "1.1.1.1", "8.8.4.4"]
        resolver.timeout = 5
        resolver.lifetime = 5
        dns.resolver.default_resolver = resolver
    except Exception:
        pass


_forzar_dns_publico()


@st.cache_resource(show_spinner=False)
def obtener_cliente():
    try:
        uri = st.secrets["MONGO_URI"]
    except Exception:
        st.error(
            "⚠️ Falta el secret **MONGO_URI**. Configúralo en Streamlit Cloud → "
            "Settings → Secrets:\n\n"
            '`MONGO_URI = "mongodb+srv://usuario:password@cluster.mongodb.net/"`'
        )
        st.stop()
    try:
        cliente = MongoClient(uri, serverSelectionTimeoutMS=8000)
        cliente.admin.command("ping")
        return cliente
    except ConfigurationError as e:
        st.error(
            "❌ **No se pudo resolver el DNS SRV de MongoDB Atlas.**\n\n"
            f"Detalle: {e}\n\n"
            "Solución: en Atlas → Connect → Drivers, usa la cadena de conexión "
            "*estándar* (`mongodb://...`, sin `+srv`) con los hosts del shard "
            "separados por coma, y reemplaza el secret `MONGO_URI` por esa."
        )
        st.stop()
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
col_metricas = db[COL_METRICAS]

for _col, _keys, _kw in [
    (col_precios, [("ticker", 1), ("fecha", 1)], dict(unique=True)),
    (col_svc, [("ticker", 1)], dict(unique=True)),
    (col_rnn, [("ticker", 1), ("arquitectura", 1)], dict(unique=True)),
    (col_lstm, [("ticker", 1)], dict(unique=True)),
    (col_sent_res, [("ticker", 1)], dict(unique=True)),
    (col_sent_not, [("ticker", 1), ("uuid", 1)], dict(unique=True, sparse=True)),
]:
    try:
        _col.create_index(_keys, **_kw)
    except Exception:
        pass


def _reciente(doc, campo, horas):
    if not doc or not doc.get(campo):
        return False
    ts = doc[campo]
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts) < timedelta(hours=horas)


# ─────────────────────────────────────────────────────────────────────────
# 1) INGESTA — yfinance -> indicadores -> MongoDB (ligero, automático)
# ─────────────────────────────────────────────────────────────────────────
def calcular_sma(s, v):
    return s.rolling(window=v, min_periods=v).mean()


def calcular_ema(s, v):
    return s.ewm(span=v, adjust=False, min_periods=v).mean()


def calcular_rsi(s, periodo=14):
    delta = s.diff(1)
    gan = delta.clip(lower=0)
    per = (-delta).clip(lower=0)
    mg = gan.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    mp = per.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    rs = mg / mp.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ultimo_doc_mercado(ticker):
    return col_precios.find_one({"ticker": ticker}, sort=[("fecha", -1)])


def ingesta_ticker(ticker: str, forzar: bool = False) -> bool:
    """Descarga + indicadores + upsert Mongo. Devuelve True si hubo escritura."""
    if not forzar and _reciente(_ultimo_doc_mercado(ticker), "ingestado_en", FRESCURA_MERCADO_H):
        return False

    df = yf.download(ticker, period=PERIOD, auto_adjust=AUTO_ADJUST, progress=False)
    if df is None or df.empty:
        return False
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "apertura", "High": "maximo", "Low": "minimo",
                             "Close": "cierre", "Volume": "volumen"})
    cols = ["apertura", "maximo", "minimo", "cierre", "volumen"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index)
    df = df[df["cierre"].notna() & (df["cierre"] > 0)]
    if len(df) < 30:
        return False

    cierre = df["cierre"]
    for v in SMA_VENTANAS:
        df[f"sma_{v}"] = calcular_sma(cierre, v)
    for v in EMA_VENTANAS:
        df[f"ema_{v}"] = calcular_ema(cierre, v)
    df[f"rsi_{RSI_PERIODO}"] = calcular_rsi(cierre, RSI_PERIODO)

    meta = EMPRESAS_META.get(ticker, {"nombre": ticker, "moneda": "USD"})
    ts = datetime.now(timezone.utc)
    ops = []
    for fecha, fila in df.iterrows():
        doc = {
            "ticker": ticker, "nombre": meta["nombre"], "moneda": meta["moneda"],
            "fecha": fecha.to_pydatetime().replace(tzinfo=timezone.utc),
            "fecha_str": fecha.strftime("%Y-%m-%d"),
            "apertura": nan_a_none(fila.get("apertura")), "maximo": nan_a_none(fila.get("maximo")),
            "minimo": nan_a_none(fila.get("minimo")), "cierre": nan_a_none(fila.get("cierre")),
            "volumen": int(fila["volumen"]) if pd.notna(fila.get("volumen")) else None,
            "sma_20": nan_a_none(fila.get("sma_20")), "sma_50": nan_a_none(fila.get("sma_50")),
            "ema_12": nan_a_none(fila.get("ema_12")), "ema_26": nan_a_none(fila.get("ema_26")),
            "rsi_14": nan_a_none(fila.get("rsi_14")),
            "ingestado_en": ts, "fuente": "Yahoo Finance (yfinance)",
        }
        ops.append(UpdateOne({"ticker": ticker, "fecha": doc["fecha"]}, {"$set": doc}, upsert=True))
    if ops:
        col_precios.bulk_write(ops, ordered=False)
    return True


@st.cache_data(ttl=300, show_spinner=False)
def leer_mercado(ticker: str, ultimos: int | None = None):
    docs = list(col_precios.find({"ticker": ticker}, EXCLUIR).sort("fecha", 1))
    if not docs:
        return None
    df = pd.DataFrame(docs)
    df["fecha"] = pd.to_datetime(df["fecha"])
    if ultimos:
        df = df.tail(ultimos)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# 2) SVC — ligero, automático por ticker
# ─────────────────────────────────────────────────────────────────────────
def _leer_docs_mercado(ticker):
    docs = list(col_precios.find({"ticker": ticker}, {"_id": 0}).sort("fecha", 1))
    if not docs:
        return None
    df = pd.DataFrame(docs)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha").sort_index()
    return df[~df.index.duplicated(keep="last")]


def entrenar_svc(ticker: str, forzar: bool = False) -> bool:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    if not forzar and _reciente(col_svc.find_one({"ticker": ticker}), "actualizado_en", FRESCURA_MERCADO_H):
        return False

    df_raw = _leer_docs_mercado(ticker)
    if df_raw is None or len(df_raw) < MIN_REGISTROS_SVC:
        return False

    d = df_raw.copy()
    d["cierre_manana"] = d["cierre"].shift(-1)
    d["target"] = (d["cierre_manana"] > d["cierre"]).astype(int)
    d["retorno_1d"] = d["cierre"].pct_change(1)
    d["rango_intradia"] = (d["maximo"] - d["minimo"]) / d["cierre"]
    d["precio_sobre_sma20"] = d["cierre"] / d["sma_20"] - 1
    d["precio_sobre_sma50"] = d["cierre"] / d["sma_50"] - 1
    d["cruce_ema"] = d["ema_12"] / d["ema_26"] - 1

    feats = ["sma_20", "sma_50", "ema_12", "ema_26", "rsi_14", "retorno_1d",
             "rango_intradia", "precio_sobre_sma20", "precio_sobre_sma50", "cruce_ema"]
    fila_vig = d.iloc[[-1]].copy()
    d_mod = d.dropna(subset=feats + ["target"]).copy()
    if len(d_mod) < MIN_REGISTROS_SVC:
        return False

    corte = int(len(d_mod) * 0.8)
    tr, te = d_mod.iloc[:corte], d_mod.iloc[corte:]
    if len(te) < 5:
        return False

    Xtr, ytr = tr[feats].values, tr["target"].values
    Xte, yte = te[feats].values, te["target"].values
    Xvig = fila_vig[feats].values

    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42))])
    n_splits = min(5, max(2, len(Xtr) // 20))
    grid = GridSearchCV(
        pipe,
        {"svc__kernel": ["linear", "rbf"], "svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", "auto"]},
        cv=TimeSeriesSplit(n_splits=n_splits), scoring="f1_macro", n_jobs=-1, refit=True,
    )
    grid.fit(Xtr, ytr)
    modelo = grid.best_estimator_
    ypred = modelo.predict(Xte)

    acc, prec = accuracy_score(yte, ypred), precision_score(yte, ypred, zero_division=0)
    rec, f1 = recall_score(yte, ypred, zero_division=0), f1_score(yte, ypred, zero_division=0)
    cm = confusion_matrix(yte, ypred, labels=[0, 1]).tolist()

    prob = modelo.predict_proba(Xvig)[0]
    pred = modelo.predict(Xvig)[0]
    senal = "BUY" if pred == 1 else "SELL"
    conf = float(prob[1]) if pred == 1 else float(prob[0])

    meta = EMPRESAS_META.get(ticker, {})
    ts = datetime.now(timezone.utc)
    col_svc.update_one(
        {"ticker": ticker},
        {"$set": {
            "ticker": ticker, "nombre": meta.get("nombre", ticker), "modelo": "SVC",
            "señal": senal, "confianza": nan_a_none(conf),
            "ultimo_cierre": nan_a_none(fila_vig["cierre"].iloc[0]),
            "fecha_referencia": fila_vig.index[0].strftime("%Y-%m-%d"),
            "hiperparametros": grid.best_params_,
            "metricas": {"accuracy": nan_a_none(acc), "precision": nan_a_none(prec),
                         "recall": nan_a_none(rec), "f1": nan_a_none(f1)},
            "matriz_confusion": cm, "actualizado_en": ts,
        }},
        upsert=True,
    )
    col_metricas.insert_one({
        "ticker": ticker, "modelo": "SVC", "metricas": {"accuracy": nan_a_none(acc), "f1": nan_a_none(f1)},
        "hiperparametros": grid.best_params_, "entrenado_en": ts,
    })
    return True


@st.cache_data(ttl=120, show_spinner=False)
def leer_svc(ticker: str):
    return col_svc.find_one({"ticker": ticker, "modelo": "SVC"}, EXCLUIR)


# ─────────────────────────────────────────────────────────────────────────
# 3) RNN clasificadores (TensorFlow) — SOLO 1 ticker, SOLO on-demand
# ─────────────────────────────────────────────────────────────────────────
COLS_FEATURES_RNN = [
    "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14",
    "retorno_1d", "retorno_3d", "rango_intradia",
    "precio_vs_sma20", "precio_vs_sma50", "cruce_ema",
]


def _construir_features_rnn(df):
    d = df.copy()
    d["retorno_1d"] = d["cierre"].pct_change(1)
    d["retorno_3d"] = d["cierre"].pct_change(3)
    d["rango_intradia"] = (d["maximo"] - d["minimo"]) / d["cierre"]
    d["precio_vs_sma20"] = d["cierre"] / d["sma_20"] - 1
    d["precio_vs_sma50"] = d["cierre"] / d["sma_50"] - 1
    d["cruce_ema"] = d["ema_12"] / d["ema_26"] - 1
    d["target"] = (d["cierre"].shift(-1) > d["cierre"]).astype(int)
    return d


def _construir_modelo_rnn(arq, ventana, n_feat):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    tf.random.set_seed(42)
    m = Sequential(name=f"{arq}_Clasificador")
    m.add(Input(shape=(ventana, n_feat)))
    if arq == "SimpleRNN":
        m.add(SimpleRNN(64)); m.add(Dropout(0.2))
    elif arq == "GRU":
        m.add(GRU(64, return_sequences=True)); m.add(Dropout(0.2))
        m.add(GRU(32)); m.add(Dropout(0.2))
    elif arq == "LSTM":
        m.add(LSTM(64, return_sequences=True)); m.add(Dropout(0.2))
        m.add(LSTM(32)); m.add(Dropout(0.2))
    m.add(Dense(16, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def entrenar_rnn_arquitectura(ticker: str, arq: str):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    df_raw = _leer_docs_mercado(ticker)
    if df_raw is None or len(df_raw) < MIN_REGISTROS_SVC + VENTANA_RNN:
        return None

    d = _construir_features_rnn(df_raw).dropna(subset=COLS_FEATURES_RNN + ["target"]).copy()
    if len(d) < MIN_REGISTROS_SVC + VENTANA_RNN:
        return None

    fila_vig = d.iloc[[-1]]
    d_mod = d.iloc[:-1]
    X_all, y_all = d_mod[COLS_FEATURES_RNN].values, d_mod["target"].values

    scaler = MinMaxScaler()
    X_all_n = scaler.fit_transform(X_all)

    Xv, yv = [], []
    for i in range(VENTANA_RNN, len(X_all_n)):
        Xv.append(X_all_n[i - VENTANA_RNN:i]); yv.append(y_all[i])
    Xv, yv = np.array(Xv), np.array(yv)
    if len(Xv) < 30:
        return None

    corte = int(len(Xv) * 0.8)
    Xtr, Xte = Xv[:corte], Xv[corte:]
    ytr, yte = yv[:corte], yv[corte:]
    if len(Xte) < 5:
        return None

    modelo = _construir_modelo_rnn(arq, VENTANA_RNN, len(COLS_FEATURES_RNN))
    cbs = [EarlyStopping(monitor="val_loss", patience=PATIENCE_RNN, restore_best_weights=True, verbose=0),
           ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6, verbose=0)]
    hist = modelo.fit(Xtr, ytr, epochs=EPOCHS_RNN, batch_size=BATCH_SIZE,
                       validation_split=0.1, callbacks=cbs, verbose=0)

    yprob = modelo.predict(Xte, verbose=0).flatten()
    ypred = (yprob >= 0.5).astype(int)
    metr = {"accuracy": nan_a_none(accuracy_score(yte, ypred)),
            "precision": nan_a_none(precision_score(yte, ypred, zero_division=0)),
            "recall": nan_a_none(recall_score(yte, ypred, zero_division=0)),
            "f1": nan_a_none(f1_score(yte, ypred, zero_division=0))}
    cm = confusion_matrix(yte, ypred, labels=[0, 1]).tolist()

    X_vig_seq = X_all_n[-VENTANA_RNN:].reshape(1, VENTANA_RNN, len(COLS_FEATURES_RNN))
    prob_vig = float(modelo.predict(X_vig_seq, verbose=0).flatten()[0])
    pred_vig = 1 if prob_vig >= 0.5 else 0
    senal = "BUY" if pred_vig == 1 else "SELL"
    conf = prob_vig if pred_vig == 1 else 1 - prob_vig

    meta = EMPRESAS_META.get(ticker, {})
    ts = datetime.now(timezone.utc)
    doc = {
        "ticker": ticker, "nombre": meta.get("nombre", ticker), "modelo": "RNN",
        "arquitectura": arq, "senal": senal, "confianza": nan_a_none(conf),
        "acc": metr["accuracy"], "prec": metr["precision"], "rec": metr["recall"], "f1": metr["f1"],
        "matriz_confusion": cm, "epocas": len(hist.history["loss"]),
        "ultimo_cierre": nan_a_none(float(d["cierre"].iloc[-1])),
        "fecha_referencia": fila_vig.index[0].strftime("%Y-%m-%d"),
        "actualizado_en": ts,
    }
    col_rnn.update_one({"ticker": ticker, "arquitectura": arq}, {"$set": doc}, upsert=True)
    return doc


@st.cache_data(ttl=120, show_spinner=False)
def leer_rnn(ticker: str):
    docs = list(col_rnn.find({"ticker": ticker}, EXCLUIR).sort("arquitectura", 1))
    return {d["arquitectura"]: d for d in docs}


# ─────────────────────────────────────────────────────────────────────────
# 4) Regresor LSTM (TensorFlow) — SOLO 1 ticker, SOLO on-demand
# ─────────────────────────────────────────────────────────────────────────
def _leer_serie_cierre(ticker):
    df = _leer_docs_mercado(ticker)
    if df is None:
        return None
    return df["cierre"].dropna()


def _crear_ventanas(serie_norm, ventana):
    X, y = [], []
    for i in range(ventana, len(serie_norm)):
        X.append(serie_norm[i - ventana:i, 0]); y.append(serie_norm[i, 0])
    return np.array(X).reshape(-1, ventana, 1), np.array(y)


def _construir_modelo_lstm_reg(ventana):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
    from tensorflow.keras.optimizers import Adam

    m = Sequential(name="LSTM_Regressor", layers=[
        Input(shape=(ventana, 1)),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32), Dropout(0.2),
        Dense(16, activation="relu"), Dense(1, activation="linear"),
    ])
    m.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
    return m


def _predecir_horizonte(modelo, ult_ventana_norm, scaler, horizonte, rmse_usd):
    actual = ult_ventana_norm.copy().reshape(-1, 1)
    preds = []
    for _ in range(horizonte):
        entrada = actual[-VENTANA_LSTM:].reshape(1, VENTANA_LSTM, 1)
        p = modelo.predict(entrada, verbose=0)[0, 0]
        preds.append(p)
        actual = np.append(actual, [[p]], axis=0)
    preds_usd = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().tolist()
    return {
        "predicciones": [nan_a_none(p) for p in preds_usd],
        "banda_superior": [nan_a_none(p + 1.96 * rmse_usd) for p in preds_usd],
        "banda_inferior": [nan_a_none(p - 1.96 * rmse_usd) for p in preds_usd],
        "precio_final": nan_a_none(preds_usd[-1]),
    }


def entrenar_lstm_regresor(ticker: str):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    serie = _leer_serie_cierre(ticker)
    min_req = VENTANA_LSTM * 2
    if serie is None or len(serie) < min_req:
        return None

    valores = serie.values.reshape(-1, 1).astype(float)
    corte = int(len(valores) * 0.85)
    train_raw, test_raw = valores[:corte], valores[corte:]

    scaler = MinMaxScaler((0, 1))
    train_n = scaler.fit_transform(train_raw)
    test_n = scaler.transform(test_raw)
    todo_n = np.concatenate([train_n, test_n], axis=0)

    X, y = _crear_ventanas(todo_n, VENTANA_LSTM)
    n_train = corte - VENTANA_LSTM
    if n_train <= 0:
        return None
    Xtr, Xte, ytr, yte = X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    if len(Xte) < 5:
        return None

    modelo = _construir_modelo_lstm_reg(VENTANA_LSTM)
    cbs = [EarlyStopping(monitor="val_loss", patience=PATIENCE_LSTM, restore_best_weights=True, verbose=0),
           ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6, verbose=0)]
    hist = modelo.fit(Xtr, ytr, epochs=EPOCHS_LSTM, batch_size=BATCH_SIZE,
                       validation_split=0.1, callbacks=cbs, verbose=0)

    ypred_n = modelo.predict(Xte, verbose=0).flatten()
    yreal = scaler.inverse_transform(yte.reshape(-1, 1)).flatten()
    ypred = scaler.inverse_transform(ypred_n.reshape(-1, 1)).flatten()
    rmse = float(np.sqrt(mean_squared_error(yreal, ypred)))
    mae = float(mean_absolute_error(yreal, ypred))
    r2 = float(r2_score(yreal, ypred))
    rmse_pct = rmse / float(np.mean(yreal)) * 100 if np.mean(yreal) else 0.0

    ult_ventana = todo_n[-VENTANA_LSTM:].flatten()
    fecha_base = serie.index[-1]
    preds_fut = {}
    for h in HORIZONTES_LSTM:
        ph = _predecir_horizonte(modelo, ult_ventana, scaler, h, rmse)
        fechas_fut = [(fecha_base + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(h)]
        preds_fut[str(h)] = {**ph, "fechas": fechas_fut}

    meta = EMPRESAS_META.get(ticker, {})
    ts = datetime.now(timezone.utc)
    doc = {
        "ticker": ticker, "nombre": meta.get("nombre", ticker), "moneda": meta.get("moneda", "USD"),
        "modelo": "LSTM_Regressor", "ventana_dias": VENTANA_LSTM, "horizontes": HORIZONTES_LSTM,
        "ultimo_precio": nan_a_none(float(serie.iloc[-1])), "fecha_ultimo": serie.index[-1].strftime("%Y-%m-%d"),
        "metricas": {"rmse_usd": nan_a_none(rmse), "rmse_pct": nan_a_none(rmse_pct),
                     "mae_usd": nan_a_none(mae), "r2": nan_a_none(r2)},
        "predicciones_futuras": preds_fut, "epocas_reales": len(hist.history["loss"]),
        "actualizado_en": ts,
    }
    col_lstm.update_one({"ticker": ticker}, {"$set": doc}, upsert=True)
    return doc


@st.cache_data(ttl=120, show_spinner=False)
def leer_lstm(ticker: str):
    return col_lstm.find_one({"ticker": ticker}, EXCLUIR)


# ─────────────────────────────────────────────────────────────────────────
# 5) NLP / Sentimiento — ligero, automático
# ─────────────────────────────────────────────────────────────────────────
def clasificar_sentimiento(compound):
    if compound >= UMBRAL_POS:
        return "POSITIVO"
    if compound <= UMBRAL_NEG:
        return "NEGATIVO"
    return "NEUTRO"


def _extraer_noticia(item):
    """Soporta el formato clásico y el formato nuevo (nested 'content') de yfinance."""
    if "content" in item:
        c = item["content"]
        return {
            "uuid": item.get("id", c.get("id", "")),
            "titulo": c.get("title", ""),
            "publisher": (c.get("provider") or {}).get("displayName", ""),
            "link": (c.get("canonicalUrl") or {}).get("url", ""),
            "fecha": c.get("pubDate"),
        }
    return {
        "uuid": item.get("uuid", ""), "titulo": item.get("title", ""),
        "publisher": item.get("publisher", ""), "link": item.get("link", ""),
        "fecha": item.get("providerPublishTime"),
    }


def analizar_sentimiento(ticker: str, forzar: bool = False) -> bool:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    if not forzar and _reciente(col_sent_res.find_one({"ticker": ticker}), "actualizado_en", FRESCURA_NLP_H):
        return False

    try:
        noticias_raw = yf.Ticker(ticker).news or []
    except Exception:
        noticias_raw = []

    analizador = SentimentIntensityAnalyzer()
    ts = datetime.now(timezone.utc)
    docs, scores = [], []

    for item in noticias_raw:
        n = _extraer_noticia(item)
        titulo = n["titulo"] or ""
        sc = analizador.polarity_scores(titulo) if titulo.strip() else {"pos": 0, "neg": 0, "neu": 0, "compound": 0}
        comp = nan_a_none(sc["compound"])
        if comp is not None:
            scores.append(comp)

        fecha_pub = n["fecha"]
        if isinstance(fecha_pub, (int, float)):
            fecha_pub = datetime.fromtimestamp(fecha_pub, tz=timezone.utc)
        elif isinstance(fecha_pub, str):
            try:
                fecha_pub = datetime.fromisoformat(fecha_pub.replace("Z", "+00:00"))
            except ValueError:
                fecha_pub = None

        docs.append({
            "ticker": ticker, "uuid": n["uuid"] or titulo[:80], "titulo": titulo,
            "publisher": n["publisher"], "link": n["link"],
            "fecha_publicacion": fecha_pub,
            "fecha_pub_str": fecha_pub.strftime("%Y-%m-%d") if fecha_pub else None,
            "vader_pos": nan_a_none(sc["pos"]), "vader_neg": nan_a_none(sc["neg"]),
            "vader_neu": nan_a_none(sc["neu"]), "vader_compound": comp,
            "sentimiento": clasificar_sentimiento(sc["compound"]),
            "analizado_en": ts,
        })

    if docs:
        ops = [UpdateOne({"ticker": d["ticker"], "uuid": d["uuid"]}, {"$set": d}, upsert=True) for d in docs]
        col_sent_not.bulk_write(ops, ordered=False)

    comp_prom = float(np.mean(scores)) if scores else None
    sent_glob = clasificar_sentimiento(comp_prom) if comp_prom is not None else "NEUTRO"
    n_pos = sum(1 for d in docs if d["sentimiento"] == "POSITIVO")
    n_neg = sum(1 for d in docs if d["sentimiento"] == "NEGATIVO")
    n_neu = sum(1 for d in docs if d["sentimiento"] == "NEUTRO")

    meta = EMPRESAS_META.get(ticker, {})
    col_sent_res.update_one(
        {"ticker": ticker},
        {"$set": {
            "ticker": ticker, "nombre": meta.get("nombre", ticker),
            "n_noticias": len(docs), "compound_promedio": nan_a_none(comp_prom),
            "sentimiento_global": sent_glob,
            "distribucion": {"positivo": n_pos, "negativo": n_neg, "neutro": n_neu},
            "actualizado_en": ts,
        }},
        upsert=True,
    )
    return True


@st.cache_data(ttl=120, show_spinner=False)
def leer_sentimiento(ticker: str):
    return col_sent_res.find_one({"ticker": ticker}, EXCLUIR)


@st.cache_data(ttl=120, show_spinner=False)
def leer_noticias(ticker: str, limite: int = 15):
    return list(col_sent_not.find({"ticker": ticker}, EXCLUIR).sort("fecha_publicacion", -1).limit(limite))


def sent_a_senal(s):
    return {"POSITIVO": "BUY", "NEGATIVO": "SELL"}.get(s, "HOLD")


def color_senal(s):
    return {"BUY": "🟢", "SELL": "🔴"}.get(s, "🟡")


def limpiar_cache_lecturas():
    leer_mercado.clear(); leer_svc.clear(); leer_rnn.clear()
    leer_lstm.clear(); leer_sentimiento.clear(); leer_noticias.clear()


def pipeline_ligero(ticker: str, forzar: bool = False):
    """Ingesta + SVC + NLP — rápido, se puede correr por cada ticker sin drama."""
    ingesta_ticker(ticker, forzar=forzar)
    entrenar_svc(ticker, forzar=forzar)
    analizar_sentimiento(ticker, forzar=forzar)
    limpiar_cache_lecturas()


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📈 Invest**AI** · Autónomo")
st.sidebar.caption("UNMSM · FISI · iDeSo 2026-II · Grupo 10 — sin Colab/ngrok")
ticker = st.sidebar.selectbox("Ticker minero", TICKERS, format_func=lambda t: f"{t} — {EMPRESAS_META.get(t, {}).get('nombre', t)}")
pagina = st.sidebar.radio(
    "Módulo",
    ["🏠 Dashboard", "📊 Mercado", "🤖 Clasificador SVC", "🧠 Consola RNN (TensorFlow)",
     "📉 Regresor LSTM (TensorFlow)", "💬 Sentimiento NLP",
     "⚡ Señales Broker (5 tickers)", "💼 Portafolio simulado"],
)
st.sidebar.divider()
forzar_actualizacion = st.sidebar.checkbox("🔄 Forzar re-descarga/re-cálculo (ignora caché)", value=False)
st.sidebar.caption("Ingesta + SVC + NLP se ejecutan automáticamente. RNN/LSTM solo al pulsar su botón.")


# ─────────────────────────────────────────────────────────────────────────
# Página: Dashboard
# ─────────────────────────────────────────────────────────────────────────
if pagina == "🏠 Dashboard":
    st.title(f"🏠 Dashboard — {ticker}")
    st.caption(EMPRESAS_META.get(ticker, {}).get("nombre", ticker))

    with st.spinner("Actualizando datos de mercado, SVC y sentimiento…"):
        pipeline_ligero(ticker, forzar=forzar_actualizacion)

    mer = leer_mercado(ticker)
    svc = leer_svc(ticker)
    rnn = leer_rnn(ticker)
    lstm = leer_lstm(ticker)
    sent = leer_sentimiento(ticker)

    if mer is None:
        st.warning("No se pudo descargar/leer datos de mercado para este ticker todavía.")
    else:
        ultimo = mer.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precio actual", f"${ultimo['cierre']:.4f}")
        c2.metric("Señal SVC", svc.get("señal", "—") if svc else "—",
                  f"{(svc.get('confianza', 0) * 100):.1f}% conf." if svc else "")
        rnn_sen = [d.get("senal") for d in rnn.values()] if rnn else []
        consenso_rnn = "BUY" if rnn_sen.count("BUY") > len(rnn_sen) / 2 else ("SELL" if rnn_sen else "— (sin entrenar)")
        c3.metric("Consenso RNN", consenso_rnn)
        if lstm:
            pf7 = lstm.get("predicciones_futuras", {}).get("7", {})
            c4.metric("LSTM +7d", f"${pf7.get('precio_final', 0):.4f}" if pf7 else "—")
        else:
            c4.metric("LSTM +7d", "— (sin entrenar)")
        c5.metric("Sentimiento", sent.get("sentimiento_global", "—") if sent else "—")

        if not rnn or not lstm:
            st.info("Los modelos RNN y/o LSTM aún no se han entrenado para este ticker. "
                    "Ve a **🧠 Consola RNN** o **📉 Regresor LSTM** en el menú lateral.")

        s = mer.tail(90)
        fig = go.Figure(data=[
            go.Candlestick(x=s["fecha"], open=s["apertura"], high=s["maximo"], low=s["minimo"],
                           close=s["cierre"], name=ticker,
                           increasing_line_color="#22c55e", decreasing_line_color="#ef4444"),
            go.Scatter(x=s["fecha"], y=s.get("sma_20"), name="SMA-20", line=dict(color="#f59e0b", width=1.3)),
        ])
        fig.update_layout(template="plotly_dark", height=380, margin=dict(t=20, b=20, l=40, r=20),
                           xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        votos = [svc.get("señal") if svc else None,
                 consenso_rnn if consenso_rnn.startswith(("B", "S")) else None,
                 sent_a_senal(sent.get("sentimiento_global")) if sent else None]
        votos = [v for v in votos if v]
        n_buy, n_sell = votos.count("BUY"), votos.count("SELL")
        estrategia = "BUY" if n_buy > n_sell else "SELL" if n_sell > n_buy else "HOLD"
        st.subheader(f"Señal consolidada: {color_senal(estrategia)} **{estrategia}**")
        st.caption(f"{n_buy} BUY · {n_sell} SELL de {len(votos)} modelos con señal disponible")


# ─────────────────────────────────────────────────────────────────────────
# Página: Mercado
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "📊 Mercado":
    st.title(f"📊 Visualización de Mercado — {ticker}")
    with st.spinner("Descargando de Yahoo Finance y calculando indicadores…"):
        ingesta_ticker(ticker, forzar=forzar_actualizacion)
        limpiar_cache_lecturas()
    mer = leer_mercado(ticker)

    if mer is None:
        st.warning("No se pudo obtener datos de mercado.")
    else:
        u = mer.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Último cierre", f"${u['cierre']:.4f}")
        c2.metric("RSI-14", f"{u.get('rsi_14', 0):.2f}" if pd.notna(u.get("rsi_14")) else "—")
        c3.metric("SMA-20", f"${u.get('sma_20', 0):.4f}" if pd.notna(u.get("sma_20")) else "—")
        c4.metric("SMA-50", f"${u.get('sma_50', 0):.4f}" if pd.notna(u.get("sma_50")) else "—")
        c5.metric("EMA-12", f"${u.get('ema_12', 0):.4f}" if pd.notna(u.get("ema_12")) else "—")

        fig = go.Figure(data=[
            go.Candlestick(x=mer["fecha"], open=mer["apertura"], high=mer["maximo"], low=mer["minimo"],
                           close=mer["cierre"], name=ticker,
                           increasing_line_color="#22c55e", decreasing_line_color="#ef4444"),
            go.Scatter(x=mer["fecha"], y=mer.get("sma_20"), name="SMA-20", line=dict(color="#f59e0b", width=1.3)),
            go.Scatter(x=mer["fecha"], y=mer.get("sma_50"), name="SMA-50", line=dict(color="#3b82f6", width=1.3, dash="dot")),
            go.Scatter(x=mer["fecha"], y=mer.get("ema_12"), name="EMA-12", line=dict(color="#a78bfa", width=1.1)),
        ])
        fig.update_layout(template="plotly_dark", height=460, xaxis_rangeslider_visible=False,
                           margin=dict(t=20, b=20, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)

        fig_rsi = go.Figure(go.Scatter(x=mer["fecha"], y=mer.get("rsi_14"), line=dict(color="#f59e0b")))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="#ef4444")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="#22c55e")
        fig_rsi.update_layout(template="plotly_dark", height=200, title="RSI-14", margin=dict(t=30, b=20, l=40, r=20))
        st.plotly_chart(fig_rsi, use_container_width=True)

        with st.expander("📋 Últimos 15 registros"):
            st.dataframe(
                mer.tail(15).sort_values("fecha", ascending=False)[
                    ["fecha", "apertura", "maximo", "minimo", "cierre", "volumen", "sma_20", "rsi_14"]],
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────
# Página: Clasificador SVC
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "🤖 Clasificador SVC":
    st.title(f"🤖 Clasificador SVC — {ticker}")
    with st.spinner("Entrenando SVC (GridSearchCV + TimeSeriesSplit)…"):
        ingesta_ticker(ticker, forzar=forzar_actualizacion)
        entrenar_svc(ticker, forzar=forzar_actualizacion)
        limpiar_cache_lecturas()

    doc = leer_svc(ticker)
    if not doc:
        st.warning("No hay suficientes registros históricos todavía para entrenar el SVC.")
    else:
        senal, conf = doc.get("señal", "—"), doc.get("confianza", 0) or 0
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Señal vigente", f"{color_senal(senal)} {senal}")
            st.progress(min(max(conf, 0), 1), text=f"Confianza: {conf*100:.1f}%")
        with c2:
            st.write("**Hiperparámetros (GridSearchCV):**")
            st.json(doc.get("hiperparametros", {}))
        m = doc.get("metricas", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{(m.get('accuracy') or 0)*100:.1f}%")
        c2.metric("Precision", f"{(m.get('precision') or 0)*100:.1f}%")
        c3.metric("Recall", f"{(m.get('recall') or 0)*100:.1f}%")
        c4.metric("F1-score", f"{(m.get('f1') or 0)*100:.1f}%")
        st.caption(f"Último cierre: {doc.get('ultimo_cierre')} · Fecha ref.: {doc.get('fecha_referencia')} · "
                   f"Entrenado: {doc.get('actualizado_en')}")


# ─────────────────────────────────────────────────────────────────────────
# Página: Consola RNN (TensorFlow, on-demand, 1 ticker)
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "🧠 Consola RNN (TensorFlow)":
    st.title(f"🧠 Consola de Modelos RNN — {ticker}")
    st.caption("Entrena SimpleRNN, GRU y LSTM (clasificación BUY/SELL) SOLO para este ticker. "
               f"~{EPOCHS_RNN} épocas máx. por arquitectura, toma ~15-40s cada una.")

    ingesta_ticker(ticker, forzar=False)  # asegura datos de mercado sin forzar descarga pesada

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        entrenar_1 = st.selectbox("Arquitectura a entrenar", ["SimpleRNN", "GRU", "LSTM"])
    with col_btn2:
        st.write("")
        st.write("")
        disparar = st.button(f"🚀 Entrenar {entrenar_1} para {ticker}", type="primary")

    if disparar:
        with st.spinner(f"Entrenando {entrenar_1} para {ticker} (TensorFlow)… puede tardar hasta 1 minuto."):
            res = entrenar_rnn_arquitectura(ticker, entrenar_1)
            leer_rnn.clear()
        if res is None:
            st.error("No hay suficientes registros históricos para entrenar esta arquitectura en este ticker.")
        else:
            st.success(f"✅ {entrenar_1} entrenado: señal {res['senal']} ({res['confianza']*100:.1f}% confianza)")

    modelos = leer_rnn(ticker)
    st.divider()
    if not modelos:
        st.info("Aún no se ha entrenado ningún modelo RNN para este ticker. Usa el botón de arriba.")
    else:
        cols = st.columns(3)
        for col, arq in zip(cols, ["SimpleRNN", "GRU", "LSTM"]):
            d = modelos.get(arq)
            with col:
                st.markdown(f"### {arq}")
                if not d:
                    st.caption("No entrenado aún")
                    continue
                senal = d.get("senal", "—")
                st.markdown(f"## {color_senal(senal)} {senal}")
                st.progress(min(max(d.get("confianza", 0) or 0, 0), 1))
                st.caption(f"Acc: {(d.get('acc') or 0)*100:.1f}% · F1: {(d.get('f1') or 0)*100:.1f}% · "
                           f"Épocas: {d.get('epocas', '—')}")


# ─────────────────────────────────────────────────────────────────────────
# Página: Regresor LSTM (TensorFlow, on-demand, 1 ticker)
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "📉 Regresor LSTM (TensorFlow)":
    st.title(f"📉 Regresor LSTM — {ticker}")
    st.caption(f"Ventana {VENTANA_LSTM} días · Horizontes {HORIZONTES_LSTM} · "
               f"~{EPOCHS_LSTM} épocas máx. Toma ~30-90s. Solo para {ticker}.")

    ingesta_ticker(ticker, forzar=False)

    if st.button(f"🚀 Entrenar Regresor LSTM para {ticker}", type="primary"):
        with st.spinner("Entrenando LSTM Regressor (TensorFlow)… puede tardar hasta 2 minutos."):
            res = entrenar_lstm_regresor(ticker)
            leer_lstm.clear()
        if res is None:
            st.error("No hay suficientes registros históricos (~120 días) para entrenar el regresor en este ticker.")
        else:
            st.success(f"✅ LSTM entrenado — RMSE: ${res['metricas']['rmse_usd']:.4f} "
                       f"({res['metricas']['rmse_pct']:.2f}%) · R²: {res['metricas']['r2']:.4f}")

    doc = leer_lstm(ticker)
    st.divider()
    if not doc:
        st.info("Aún no se ha entrenado el regresor LSTM para este ticker. Usa el botón de arriba.")
    else:
        m = doc.get("metricas", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Último precio", f"${doc.get('ultimo_precio', 0):.4f}")
        c2.metric("RMSE", f"${m.get('rmse_usd', 0):.4f}")
        c3.metric("MAE", f"${m.get('mae_usd', 0):.4f}")
        c4.metric("R²", f"{m.get('r2', 0):.4f}")

        horizonte = st.radio("Horizonte", HORIZONTES_LSTM, horizontal=True)
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
    with st.spinner("Descargando noticias de Yahoo Finance y analizando con VADER…"):
        analizar_sentimiento(ticker, forzar=forzar_actualizacion)
        limpiar_cache_lecturas()

    resumen = leer_sentimiento(ticker)
    noticias = leer_noticias(ticker)

    if not resumen:
        st.warning("No se encontraron noticias para este ticker en Yahoo Finance en este momento.")
    else:
        sent, comp = resumen.get("sentimiento_global", "—"), resumen.get("compound_promedio", 0) or 0
        dist = resumen.get("distribucion", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentimiento global", sent)
        c2.metric("Compound promedio", f"{comp:+.4f}")
        c3.metric("Noticias analizadas", resumen.get("n_noticias", 0))
        c4.metric("Positivas / Negativas", f"{dist.get('positivo', 0)} / {dist.get('negativo', 0)}")

        fig = go.Figure(go.Indicator(mode="gauge+number", value=comp,
                                      gauge={"axis": {"range": [-1, 1]},
                                             "bar": {"color": "#1fe0c4" if comp >= 0 else "#ef4444"}}))
        fig.update_layout(template="plotly_dark", height=250, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📰 Noticias recientes con score VADER")
    if not noticias:
        st.info("Sin noticias disponibles en este momento (Yahoo Finance no siempre devuelve noticias para todos los tickers).")
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
# Página: Broker (5 tickers — solo pipeline ligero, RNN/LSTM si ya existen)
# ─────────────────────────────────────────────────────────────────────────
elif pagina == "⚡ Señales Broker (5 tickers)":
    st.title("⚡ Panel de Señales — Todos los Tickers")
    st.caption("Ingesta + SVC + NLP se ejecutan para los 5 tickers (ligero). "
               "RNN/LSTM solo aparecen si ya los entrenaste antes desde sus módulos dedicados.")

    if st.button("🔄 Actualizar los 5 tickers ahora"):
        prog = st.progress(0.0)
        for i, t in enumerate(TICKERS):
            pipeline_ligero(t, forzar=forzar_actualizacion)
            prog.progress((i + 1) / len(TICKERS))
        st.success("Actualizado.")

    filas = []
    for t in TICKERS:
        svc, rnn, sent = leer_svc(t), leer_rnn(t), leer_sentimiento(t)
        mer = leer_mercado(t, ultimos=1)
        precio = mer.iloc[-1]["cierre"] if mer is not None and not mer.empty else None
        rnn_sen = [d.get("senal") for d in rnn.values()] if rnn else []
        consenso_rnn = "BUY" if rnn_sen.count("BUY") > len(rnn_sen) / 2 else ("SELL" if rnn_sen else "— (sin entrenar)")
        filas.append({
            "Ticker": t, "Precio": f"${precio:.4f}" if precio else "—",
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
        filas.append({"Ticker": t, "Empresa": EMPRESAS_META[t]["nombre"],
                       "Señal SVC": svc.get("señal", "—") if svc else "—", "Valor USD": val})
    df = pd.DataFrame(filas)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=df["Ticker"], values=df["Valor USD"], hole=0.5))
        fig.update_layout(template="plotly_dark", height=320, title="Distribución del portafolio")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(df, use_container_width=True, hide_index=True)


st.sidebar.divider()
st.sidebar.caption(f"Última recarga: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
