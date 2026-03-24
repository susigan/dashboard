# ════════════════════════════════════════════════════════════════════════════════
# data_loader.py — ATHELTICA Dashboard — Google Sheets auth + load + preprocess
# ════════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
from config import WELLNESS_URL, TRAINING_URL, SCOPES, MAPA_WELLNESS, MAPA_TRAINING, VALID_TYPES, TYPE_MAP, ANNUAL_SPREADSHEET_ID, ANNUAL_SHEETS
from utils.helpers import detectar_col, br_float, parse_date, norm_tipo, remove_zscore, remove_zeros, fill_missing

@st.cache_data(ttl=3600, show_spinner="A carregar wellness...")
def carregar_wellness(days_back):
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        ws  = gc.open_by_url(WELLNESS_URL).worksheet("Respostas ao formulário 1")
        df  = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]
        cd  = detectar_col(df, ['Data', 'data', 'Date', 'Carimbo de data/hora'])
        if cd:
            df['Data'] = df[cd].apply(parse_date)
            df = df.dropna(subset=['Data']).sort_values('Data')
        for var, lst in MAPA_WELLNESS.items():
            col = detectar_col(df, lst)
            if col: df[var] = df[col].apply(br_float)
        dm = datetime.now() - timedelta(days=days_back)
        return df[df['Data'] >= dm].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro ao carregar wellness: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="A carregar atividades...")
def carregar_atividades(days_back):
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        ws  = gc.open_by_url(TRAINING_URL).worksheet("intervals.icu_activities-export")
        df  = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]
        cd  = detectar_col(df, ['Date', 'start_date_local', 'date', 'data', 'Data'])
        if cd:
            df['Data'] = df[cd].apply(lambda x: parse_date(str(x)[:10]))
            df = df.dropna(subset=['Data']).sort_values('Data')
        TEXTO = ['type', 'name', 'date', 'start_date_local']
        for var, lst in MAPA_TRAINING.items():
            col = detectar_col(df, lst)
            if col: df[var] = df[col] if var in TEXTO else df[col].apply(br_float)
        if 'type' in df.columns: df['type'] = df['type'].apply(norm_tipo)
        dm = datetime.now() - timedelta(days=days_back)
        return df[df['Data'] >= dm].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro ao carregar atividades: {e}")
        return pd.DataFrame()

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preproc_wellness(df):
    """Limpeza wellness: z-score, zeros, rolling fill."""
    if len(df) == 0: return df
    df = df.copy().sort_values('Data')
    df = df.drop_duplicates(subset=['Data'], keep='first')
    for c in [c for c in ['hrv', 'rhr', 'sleep_hours', 'sleep_quality',
                           'stress', 'fatiga', 'humor', 'soreness', 'peso', 'fat'] if c in df.columns]:
        df[c] = remove_zscore(df[c], 3.0)
    df = remove_zeros(df, ['hrv', 'rhr', 'sleep_hours'])
    for c in [c for c in ['hrv', 'rhr', 'sleep_quality', 'fatiga',
                           'stress', 'humor', 'soreness'] if c in df.columns]:
        df = fill_missing(df, c, 7)
    return df.reset_index(drop=True)

def preproc_ativ(df):
    """Limpeza atividades: deduplicação, tipos válidos, duração mínima."""
    if len(df) == 0: return df
    df = df.copy().sort_values('Data')
    sub = [c for c in ['Data', 'type', 'moving_time'] if c in df.columns]
    if len(sub) >= 2: df = df.drop_duplicates(subset=sub, keep='first')
    if 'type' in df.columns:       df = df[df['type'].isin(VALID_TYPES)]
    if 'moving_time' in df.columns: df = df[pd.to_numeric(df['moving_time'], errors='coerce') > 60]
    if 'icu_eftp'    in df.columns: df['icu_eftp'] = remove_zscore(df['icu_eftp'], 3.5)
    df = remove_zeros(df, ['moving_time', 'icu_eftp'])
    return df.reset_index(drop=True)

def filtrar_datas(df, di, df_):
    """Filtra DataFrame pelo intervalo [di, df_]."""
    if len(df) == 0: return df
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'])
    return df[(df['Data'].dt.date >= di) & (df['Data'].dt.date <= df_)].reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner="A carregar dados anuais (Aquecimentos)...")
def carregar_annual():
    """
    Carrega AquecSki, AquecBike, AquecRow da planilha Annual via gviz CSV.
    Igual ao código original Python (SPREADSHEET_ID = 1AEKhDrda9xhxRQA_1ty3z3oPELzH6oANa6L0cysJSMk).
    Não precisa de autenticação — planilha pública via gviz.
    """
    dfs = {}
    COLUNAS_NAO_NUM = ['Mês', 'Fase', 'DATA', 'Treino_antes', 'Atividade', 'Mes']
    for aba in ANNUAL_SHEETS:
        url = (f"https://docs.google.com/spreadsheets/d/{ANNUAL_SPREADSHEET_ID}"
               f"/gviz/tq?tqx=out:csv&sheet={aba}")
        try:
            df = pd.read_csv(url)
            df.columns = [str(c).strip() for c in df.columns]

            # Renomear colunas AquecRow (igual ao original)
            if aba == "AquecRow":
                mapa = {'Unnamed: 2': 'DATA', 'Unnamed: 4': 'HR_140W', 'Unnamed: 5': 'HR_160W',
                        'Unnamed: 6': 'HR_180W', 'Unnamed: 7': 'HR_200W',
                        'Unnamed: 8': 'HR_Pwr_140w', 'Unnamed: 9': 'HR_Pwr_160w',
                        'Unnamed: 10': 'HR_Pwr_180w', 'Unnamed: 11': 'O2_140W',
                        'Unnamed: 12': 'O2_160W', 'Unnamed: 13': 'O2_180W',
                        'Treino antes': 'Treino_antes', 'Drag Factor': 'Drag_Factor'}
                df = df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})

            # Limpar e converter
            df = df.dropna(axis=1, how='all')
            df = df.replace(['', ' ', 'nan', 'NaN', 'null'], np.nan)
            for col in df.columns:
                if col not in COLUNAS_NAO_NUM:
                    df[col] = pd.to_numeric(df[col], errors='coerce').replace({0.0: np.nan, 0: np.nan})
            df['Atividade'] = aba
            dfs[aba] = df
        except Exception as e:
            dfs[aba] = pd.DataFrame()

    # DataFrame unificado
    validos = [d for d in dfs.values() if len(d) > 0]
    df_all = pd.concat(validos, ignore_index=True) if validos else pd.DataFrame()
    return dfs, df_all



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_visao_geral.py
# ════════════════════════════════════════════════════════════════════════════
