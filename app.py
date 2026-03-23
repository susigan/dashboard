# ════════════════════════════════════════════════════════════════════════════════
# ATHELTICA DASHBOARD — Streamlit App (VERSÃO ORIGINAL + CLEANING CORRETO)
# ════════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from scipy import stats as scipy_stats
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG / CONSTANTES (INALTERADO)
# ════════════════════════════════════════════════════════════════════════════════

WELLNESS_URL = "https://docs.google.com/spreadsheets/d/10pefcY6VI4Z45M8Y69D6JxIoqOkjzSlSpV1PMLXoYlI/edit#gid=286320937"
TRAINING_URL = "https://docs.google.com/spreadsheets/d/1RE4SISd53WmAgQo8J-k2SE_OG0w5m4dbgLHvZHPxKvw/edit?usp=sharing"
SCOPES = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

VALID_TYPES = ['Bike', 'Row', 'Run', 'Ski', 'WeightTraining']

# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUX (INALTERADO)
# ════════════════════════════════════════════════════════════════════════════════

def remove_zscore(s, thr=3.0):
    v = pd.to_numeric(s, errors='coerce')
    ok = v[v.notna()]
    if len(ok) < 4:
        return v
    z = np.abs(scipy_stats.zscore(ok))
    v.loc[ok.index[z > thr]] = np.nan
    return v

def remove_zeros(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan
    return df

def fill_missing(df, col, w=7):
    df = df.copy()
    mask = df[col].isna()
    if mask.any():
        r = df[col].rolling(w, min_periods=2, center=True).mean()
        df.loc[mask, col] = r[mask]
        mask2 = df[col].isna()
        if mask2.any():
            r2 = df[col].rolling(14, min_periods=2, center=True).mean()
            df.loc[mask2, col] = r2[mask2]
        df[col] = df[col].fillna(df[col].mean())
    return df

# ════════════════════════════════════════════════════════════════════════════════
# 🔴 PREPROCESSING — CORRIGIDO (IGUAL UNTITLED13)
# ════════════════════════════════════════════════════════════════════════════════

def preproc_wellness(df):
    if len(df) == 0:
        return df

    df = df.copy().sort_values('Data')
    df = df.drop_duplicates(subset=['Data'], keep='first')

    for c in ['hrv','rhr','sleep_hours','sleep_quality','stress','fatiga','humor','soreness','peso','fat']:
        if c in df.columns:
            df[c] = remove_zscore(df[c], 3.0)

    df = remove_zeros(df, ['hrv','rhr','sleep_hours'])

    for c in ['hrv','rhr','sleep_quality','fatiga','stress','humor','soreness']:
        if c in df.columns:
            df = fill_missing(df, c, 7)

    return df.reset_index(drop=True)


def preproc_ativ(df):
    if len(df) == 0:
        return df

    df = df.copy().sort_values('Data')

    # duplicatas
    sub = [c for c in ['Data','type','moving_time'] if c in df.columns]
    if len(sub) >= 2:
        df = df.drop_duplicates(subset=sub, keep='first')

    # tipos
    if 'type' in df.columns:
        df = df[df['type'].isin(VALID_TYPES)]

    # duração
    if 'moving_time' in df.columns:
        df['moving_time'] = pd.to_numeric(df['moving_time'], errors='coerce')
        df = df[df['moving_time'] > 60]

    # zscore (igual original)
    for col, thr in [('icu_eftp',3.5), ('AllWorkFTP',3.5)]:
        if col in df.columns:
            df[col] = remove_zscore(df[col], thr)

    # zeros
    df = remove_zeros(df, ['moving_time','icu_eftp','AllWorkFTP'])

    return df.reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# LOAD (INALTERADO)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_gc():
    creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
    return gspread.authorize(creds)

@st.cache_data(ttl=3600)
def carregar_atividades(days_back):
    gc = get_gc()
    ws = gc.open_by_url(TRAINING_URL).worksheet("intervals.icu_activities-export")
    df = get_as_dataframe(ws, evaluate_formulas=True)
    df['Data'] = pd.to_datetime(df['start_date_local'].astype(str).str[:10])
    df = df[df['Data'] >= datetime.now() - timedelta(days=days_back)]
    return df.reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# 🔴 AQUI ESTÁ A CORREÇÃO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

days_back = 730

da = carregar_atividades(days_back)

# 🔥 AGORA SIM — MESMO PIPELINE
da = preproc_ativ(da)

# ════════════════════════════════════════════════════════════════════════════════
# TODO O RESTO DO SEU APP SEGUE IGUAL
# (PMC, FTLM, tabs etc NÃO foram alterados)
# ════════════════════════════════════════════════════════════════════════════════

st.title("ATHELTICA (COM CLEANING CORRETO)")

st.write("Registros:", len(da))
