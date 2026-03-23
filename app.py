# ════════════════════════════════════════════════════════════════════════════════
# ATHELTICA DASHBOARD — FULL VERSION (CONSISTENTE COM UNTITLED13)
# ════════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta

# Google Sheets
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials

# Stats
from scipy import stats

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="ATHELTICA", layout="wide")

TRAINING_URL = "https://docs.google.com/spreadsheets/d/1RE4SISd53WmAgQo8J-k2SE_OG0w5m4dbgLHvZHPxKvw/edit"

VALID_TYPES = ['Bike', 'Row', 'Run', 'Ski', 'WeightTraining']

# ════════════════════════════════════════════════════════════════════════════════
# GOOGLE AUTH
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_gc():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    return gspread.authorize(creds)

# ════════════════════════════════════════════════════════════════════════════════
# PREPROCESSOR (IGUAL UNTITLED13)
# ════════════════════════════════════════════════════════════════════════════════

class DataPreprocessor:

    def remover_picos_zscore(self, series, threshold=3.0):
        valores = pd.to_numeric(series, errors='coerce')
        validos = valores[valores.notna()]
        if len(validos) < 4:
            return valores
        z = np.abs(stats.zscore(validos))
        valores.loc[validos.index[z > threshold]] = np.nan
        return valores

    def remover_zeros_invalidos(self, df, cols):
        for c in cols:
            if c in df.columns:
                df.loc[df[c] == 0, c] = np.nan
        return df

    def limpar_training(self, df):

        df = df.copy()
        df['Data'] = pd.to_datetime(df['start_date_local'].astype(str).str[:10])
        df = df.dropna(subset=['Data']).sort_values('Data')

        df = df[df['type'].isin(VALID_TYPES)]

        if 'icu_eftp' in df.columns:
            df['icu_eftp'] = self.remover_picos_zscore(df['icu_eftp'], 3.5)

        df = self.remover_zeros_invalidos(df, ['moving_time', 'icu_eftp'])

        df = df[df['moving_time'] > 60]

        df = df.drop_duplicates(subset=['Data', 'type', 'moving_time'])

        return df.reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# LOAD + PREPROCESS
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_data(days_back):

    gc = get_gc()
    ws = gc.open_by_url(TRAINING_URL).worksheet("intervals.icu_activities-export")

    df = get_as_dataframe(ws, evaluate_formulas=True)

    processor = DataPreprocessor()
    df = processor.limpar_training(df)

    df = df[df['Data'] >= datetime.now() - timedelta(days=days_back)]

    return df

# ════════════════════════════════════════════════════════════════════════════════
# LOAD CALCULATION (CRÍTICO)
# ════════════════════════════════════════════════════════════════════════════════

def compute_load(df):

    df = df.copy()

    # duração em minutos
    df['duration_min'] = df['moving_time'] / 60

    # RPE fallback
    if 'rpe' not in df.columns:
        df['rpe'] = 0

    df['load'] = df['duration_min'] * df['rpe']

    return df

# ════════════════════════════════════════════════════════════════════════════════
# PMC
# ════════════════════════════════════════════════════════════════════════════════

def compute_pmc(df, tau_ctl=42, tau_atl=7):

    df = df.copy().sort_values('Data')

    ctl, atl = [], []
    ctl_val, atl_val = 0, 0

    for load in df['load']:
        ctl_val += (load - ctl_val) / tau_ctl
        atl_val += (load - atl_val) / tau_atl

        ctl.append(ctl_val)
        atl.append(atl_val)

    df['CTL'] = ctl
    df['ATL'] = atl
    df['TSB'] = df['CTL'] - df['ATL']

    return df

# ════════════════════════════════════════════════════════════════════════════════
# FTLM (ADICIONADO)
# ════════════════════════════════════════════════════════════════════════════════

def compute_ftlm(df, tau=21):

    df = df.copy().sort_values('Data')

    ftlm = []
    val = 0

    for load in df['load']:
        val += (load - val) / tau
        ftlm.append(val)

    df['FTLM'] = ftlm
    return df

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

st.sidebar.title("⚙️ Config")

days = st.sidebar.selectbox("Período", [30, 60, 90, 180, 365], index=2)

modalidades = st.sidebar.multiselect(
    "Modalidades",
    VALID_TYPES,
    default=VALID_TYPES
)

# ════════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════

df = load_data(days)
df = df[df['type'].isin(modalidades)]

df = compute_load(df)
df = compute_pmc(df)
df = compute_ftlm(df)

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "📊 Visão Geral",
    "📈 PMC",
    "📉 FTLM",
    "📦 Volume",
    "⚡ Performance",
    "🔍 Debug"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — VISÃO GERAL
# ════════════════════════════════════════════════════════════════════════════════

with tabs[0]:

    st.subheader("Resumo")

    col1, col2, col3 = st.columns(3)

    col1.metric("Sessões", len(df))
    col2.metric("Carga Total", int(df['load'].sum()))
    col3.metric("CTL Atual", int(df['CTL'].iloc[-1]))

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PMC
# ════════════════════════════════════════════════════════════════════════════════

with tabs[1]:

    st.subheader("PMC")

    st.line_chart(df.set_index('Data')[['CTL', 'ATL', 'TSB']])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — FTLM
# ════════════════════════════════════════════════════════════════════════════════

with tabs[2]:

    st.subheader("FTLM")

    st.line_chart(df.set_index('Data')['FTLM'])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — VOLUME
# ════════════════════════════════════════════════════════════════════════════════

with tabs[3]:

    df_vol = df.groupby('Data')['duration_min'].sum()

    st.line_chart(df_vol)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════

with tabs[4]:

    if 'icu_eftp' in df.columns:
        st.line_chart(df.set_index('Data')['icu_eftp'])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — DEBUG
# ════════════════════════════════════════════════════════════════════════════════

with tabs[5]:

    st.write("Carga total:", df['load'].sum())
    st.write("CTL final:", df['CTL'].iloc[-1])
    st.write("ATL final:", df['ATL'].iloc[-1])
    st.write("TSB final:", df['TSB'].iloc[-1])
    st.write("FTLM final:", df['FTLM'].iloc[-1])
