# ════════════════════════════════════════════════════════════════════════════════
# ATHELTICA DASHBOARD — Streamlit App (CONSISTENTE COM UNTITLED13)
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

# Stats / ML (igual original)
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
# PREPROCESSOR (IDÊNTICO AO UNTITLED13)
# ════════════════════════════════════════════════════════════════════════════════

class DataPreprocessor:

    def remover_picos_zscore(self, series, threshold=3.0):
        valores = pd.to_numeric(series, errors='coerce')
        validos = valores[valores.notna()]

        if len(validos) < 4:
            return valores

        z_scores = np.abs(stats.zscore(validos))
        valores.loc[validos.index[z_scores > threshold]] = np.nan
        return valores

    def remover_zeros_invalidos(self, df, cols):
        for c in cols:
            if c in df.columns:
                df.loc[df[c] == 0, c] = np.nan
        return df

    def preencher_faltantes(self, df, col):
        for idx, row in df.iterrows():
            if pd.isna(row[col]):
                data_ref = row['Data']

                last7 = df[(df['Data'] < data_ref) &
                           (df['Data'] >= data_ref - timedelta(days=7))][col].dropna()

                if len(last7) >= 2:
                    df.at[idx, col] = last7.median()
                    continue

                last14 = df[(df['Data'] < data_ref) &
                            (df['Data'] >= data_ref - timedelta(days=14))][col].dropna()

                if len(last14) >= 3:
                    df.at[idx, col] = last14.median()
                    continue

                df.at[idx, col] = df[col].mean()

        return df

    def limpar_training(self, df):

        df = df.copy()

        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values('Data')

        df = df[df['type'].isin(VALID_TYPES)]

        if 'icu_eftp' in df.columns:
            df['icu_eftp'] = self.remover_picos_zscore(df['icu_eftp'], 3.5)

        if 'AllWorkFTP' in df.columns:
            df['AllWorkFTP'] = self.remover_picos_zscore(df['AllWorkFTP'], 3.5)

        df = self.remover_zeros_invalidos(df, ['moving_time', 'icu_eftp'])

        df = df[df['moving_time'] > 60]

        df = df.drop_duplicates(subset=['Data', 'type', 'moving_time'])

        return df.reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# LOAD + PREPROCESS (UNIFICADO)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_data(days_back):

    gc = get_gc()

    ws = gc.open_by_url(TRAINING_URL).worksheet("intervals.icu_activities-export")
    df = get_as_dataframe(ws, evaluate_formulas=True)

    df['Data'] = pd.to_datetime(df['start_date_local'].astype(str).str[:10])
    df = df.dropna(subset=['Data'])

    df = df[df['Data'] >= datetime.now() - timedelta(days=days_back)]

    processor = DataPreprocessor()
    df = processor.limpar_training(df)

    return df

# ════════════════════════════════════════════════════════════════════════════════
# PMC
# ════════════════════════════════════════════════════════════════════════════════

def compute_pmc(df, tau_ctl=42, tau_atl=7):

    df = df.copy().sort_values('Data')

    if 'icu_training_load' not in df.columns:
        df['icu_training_load'] = 0

    df['load'] = df['icu_training_load'].fillna(0)

    ctl = []
    atl = []

    ctl_val = 0
    atl_val = 0

    for load in df['load']:
        ctl_val += (load - ctl_val) * (1 / tau_ctl)
        atl_val += (load - atl_val) * (1 / tau_atl)

        ctl.append(ctl_val)
        atl.append(atl_val)

    df['CTL'] = ctl
    df['ATL'] = atl
    df['TSB'] = df['CTL'] - df['ATL']

    return df

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

st.sidebar.title("⚙️ Config")

days = st.sidebar.selectbox("Período", [30, 60, 90, 180, 365], index=2)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

st.title("🏃 ATHELTICA DASHBOARD (CONSISTENTE)")

df = load_data(days)

st.write(f"Registros após limpeza: {len(df)}")

df_pmc = compute_pmc(df)

# ════════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÃO PMC
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("📈 PMC (CTL / ATL / TSB)")

st.line_chart(df_pmc.set_index('Data')[['CTL', 'ATL', 'TSB']])

# ════════════════════════════════════════════════════════════════════════════════
# DEBUG (IMPORTANTE)
# ════════════════════════════════════════════════════════════════════════════════

with st.expander("🔍 Debug"):
    st.write("Carga total:", df_pmc['load'].sum())
    st.write("CTL final:", df_pmc['CTL'].iloc[-1])
    st.write("ATL final:", df_pmc['ATL'].iloc[-1])
    st.write("TSB final:", df_pmc['TSB'].iloc[-1])
