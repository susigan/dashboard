# ════════════════════════════════════════════════════════════════════════════════
# app_bundle.py — ATHELTICA Dashboard (ficheiro único gerado por build.py)
# Gerado em: 2026-03-23 20:35
# NÃO EDITAR DIRECTAMENTE — edita os módulos e corre python build.py
# ════════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from scipy import stats as scipy_stats
from scipy.stats import pearsonr, linregress, spearmanr, theilslopes, kruskal, mannwhitneyu
from itertools import combinations
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: config.py
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# config.py — ATHELTICA Dashboard
# Constantes, cores, mapeamentos de colunas
# Editar aqui para alterar URLs, cores ou mapeamentos sem tocar noutros ficheiros.
# ════════════════════════════════════════════════════════════════════════════════

WELLNESS_URL  = "https://docs.google.com/spreadsheets/d/10pefcY6VI4Z45M8Y69D6JxIoqOkjzSlSpV1PMLXoYlI/edit#gid=286320937"
FOOD_URL      = "https://docs.google.com/spreadsheets/d/10pefcY6VI4Z45M8Y69D6JxIoqOkjzSlSpV1PMLXoYlI/edit?usp=sharing"
TRAINING_URL  = "https://docs.google.com/spreadsheets/d/1RE4SISd53WmAgQo8J-k2SE_OG0w5m4dbgLHvZHPxKvw/edit?usp=sharing"
# Planilha Annual — AquecSki, AquecBike, AquecRow (igual ao original Python)
ANNUAL_SPREADSHEET_ID = "1AEKhDrda9xhxRQA_1ty3z3oPELzH6oANa6L0cysJSMk"
ANNUAL_SHEETS = ["AquecSki", "AquecBike", "AquecRow"]
SCOPES = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

CORES = {
    'verde': '#2ECC71', 'verde_escuro': '#1D8348',
    'azul': '#3498DB',  'azul_escuro':  '#2471A3',
    'laranja': '#F39C12', 'amarelo': '#F4D03F',
    'vermelho': '#E74C3C', 'vermelho_escuro': '#C0392B',
    'roxo': '#9B59B6', 'cinza': '#7F8C8D',
    'preto': '#2C3E50', 'branco': '#FFFFFF',
}
CORES_ATIV = {
    'Bike': '#E74C3C', 'Run': '#2ECC71', 'Row': '#3498DB',
    'Ski': '#9B59B6',  'WeightTraining': '#F39C12', 'Other': '#7F8C8D',
}

TYPE_MAP = {
    'VirtualSki': 'Ski', 'AlpineSki': 'Ski', 'Ski': 'Ski', 'NordicSki': 'Ski',
    'VirtualRow': 'Row', 'Rowing': 'Row', 'Row': 'Row',
    'VirtualRide': 'Bike', 'Cycling': 'Bike', 'Ride': 'Bike',
    'Bike': 'Bike', 'MountainBike': 'Bike', 'GravelRide': 'Bike',
    'VirtualRun': 'Run', 'Running': 'Run', 'Run': 'Run', 'TrailRun': 'Run', 'Treadmill': 'Run',
    'WeightTraining': 'WeightTraining',
}
VALID_TYPES = ['Bike', 'Row', 'Run', 'Ski', 'WeightTraining']
CICLICOS   = ['Bike', 'Row', 'Run', 'Ski']

MAPA_WELLNESS = {
    'hrv':          ['HRV', 'hrv', 'Heart Rate Variability'],
    'rhr':          ['HRR', 'RHR', 'rhr', 'RestingHR', 'Resting HR'],
    'sleep_hours':  ['Horas de Sono', 'Sleep', 'sleep', 'Hours Sleep'],
    'sleep_quality':['Sono Qualidade', 'Sono_Qualidade', 'Sleep Quality'],
    'stress':       ['Stress Do dia', 'Stress', 'stress'],
    'fatiga':       ['Cansaço/Vontade de Treinar', 'Fatiga', 'Fadiga'],
    'humor':        ['Humor', 'humor', 'Mood'],
    'soreness':     ['Cansaço Muscular Geral', 'Muscle Soreness', 'Soreness'],
    'peso':         ['Peso', 'Weight'],
    'fat':          ['FAT', 'Fat', 'Gordura'],
}

MAPA_TRAINING = {
    'date':             ['start_date_local', 'date', 'Date', 'data'],
    'start_date_local': ['start_date_local'],
    'moving_time':      ['moving_time', 'duration', 'Duration'],
    'distance':         ['distance', 'Distance'],
    'power_avg':        ['icu_average_watts', 'average_watts', 'AvgPower'],
    'power_max':        ['MaxPwr', 'max_power', 'Peak5m'],
    'hr_avg':           ['average_heartrate', 'avg_hr'],
    'hr_max':           ['max_heartrate', 'max_hr'],
    'rpe':              ['icu_rpe', 'RPE', 'rpe'],
    'elevation':        ['total_elevation_gain', 'elevation'],
    'type':             ['type', 'sport'],
    'name':             ['name', 'Name'],
    'xss':              ['SS', 'XSS', 'xss', 'strain_score'],
    'pmax':             ['Pmax', 'pmax', 'p_max_usage'],
    'p_max':            ['p_max', 'P_max'],
    'icu_pm_p_max':     ['icu_pm_p_max'],
    'icu_pm_cp':        ['icu_pm_cp', 'cp'],
    'icu_training_load':['icu_training_load'],
    'glycolytic':       ['Glycolytic', 'glycolytic', 'glycolytic_usage'],
    'aerobic':          ['Aerobic', 'aerobic', 'cp_usage'],
    'cadence_avg':      ['average_cadence', 'cadence'],
    'decoupling':       ['CardiacDrift', 'decoupling'],
    'icu_ftp':          ['icu_ftp', 'FTP', 'ftp'],
    'icu_eftp':         ['icu_eftp', 'eFTP', 'estimated_cp', 'est_cp', 'EFTP'],
    'z1_secs': ['z1_secs'], 'z2_secs': ['z2_secs'],
    'z3_secs': ['z3_secs'], 'z4_secs': ['z4_secs'],
    'z5_secs': ['z5_secs'], 'z6_secs': ['z6_secs'],
    'hr_z1_secs': ['hr_z1_secs'], 'hr_z2_secs': ['hr_z2_secs'],
    'hr_z3_secs': ['hr_z3_secs'], 'hr_z4_secs': ['hr_z4_secs'],
    'hr_z5_secs': ['hr_z5_secs'], 'hr_z6_secs': ['hr_z6_secs'],
    'hr_z7_secs': ['hr_z7_secs'],
    'icu_joules': ['icu_joules'], 'icu_weight': ['icu_weight'],
    'AllWorkFTP': ['AllWorkFTP'], 'WorkHourKgoverCP': ['WorkHourKgoverCP'],
    'WorkHour':   ['WorkHour'],
    # ── 3-zone power model (desde 2023) ─────────────────────────────────
    'z1_kj':  ['Z1KJ'],
    'z2_kj':  ['Z2KJ'],
    'z3_kj':  ['Z3KJ'],
    'z1_pwr': ['Z1Pw'],
    'z2_pwr': ['Z2pwr'],
    'z3_pwr': ['ZPwr'],
    'z1_sec': ['Z1sec'],
    'z2_sec': ['Z2sec'],
    'z3_sec': ['Z3sec'],
}



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: utils/helpers.py
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# utils/helpers.py — ATHELTICA Dashboard
# Funções auxiliares reutilizáveis por todas as tabs.
# Importar com: from utils.helpers import *
# ════════════════════════════════════════════════════════════════════════════════

# ── Parsing e conversão ────────────────────────────────────────────────────────

def detectar_col(df, lst):
    """Retorna o primeiro nome de coluna da lista que existe no df."""
    for n in lst:
        if n in df.columns: return n
    return None

def br_float(v):
    """Converte string BR (vírgula decimal) ou numérico para float."""
    if pd.isna(v) or v == '': return None
    if isinstance(v, (int, float)): return float(v)
    try: return float(str(v).replace(',', '.').strip())
    except: return None

def parse_date(v):
    """Parser robusto de datas em vários formatos."""
    if pd.isna(v): return None
    if isinstance(v, (pd.Timestamp, datetime)): return v
    s = str(v).strip()
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try: return datetime.strptime(s, fmt)
        except: pass
    return None

# ── Tipo de actividade ─────────────────────────────────────────────────────────

def norm_tipo(t):
    if not isinstance(t, str): return 'Other'
    return TYPE_MAP.get(t.strip(), 'Other')

def fmt_dur(horas):
    """Converte horas decimais para string legível: 55min, 1h05, 2h30"""
    if horas is None or (hasattr(horas, '__float__') and horas != horas): return '—'
    try:
        h = float(horas)
    except (TypeError, ValueError):
        return '—'
    if h <= 0: return '—'
    total_min = round(h * 60)
    hh = total_min // 60
    mm = total_min % 60
    if hh == 0:   return f'{mm}min'
    if mm == 0:   return f'{hh}h'
    return f'{hh}h{mm:02d}'

def fmt_dur_sec(segundos):
    """Converte segundos directamente para string legível."""
    if segundos is None: return '—'
    try:
        return fmt_dur(float(segundos) / 3600)
    except (TypeError, ValueError):
        return '—'

def get_cor(tipo):
    return CORES_ATIV.get(tipo, CORES_ATIV['Other'])

# ── Normalização e stats ───────────────────────────────────────────────────────

def norm_serie(s, lo=0, hi=100):
    mn, mx = s.min(), s.max()
    if mx == mn: return pd.Series([50] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * (hi - lo) + lo

def cvr(s, w=7):
    """CV% rolling."""
    m = s.rolling(w, min_periods=3).mean()
    sd = s.rolling(w, min_periods=3).std()
    return (sd / m) * 100

def conv_15(v):
    """Converte escala 1-5 para 0-100."""
    if pd.isna(v): return 50
    return max(0, min(100, (float(v) - 1) * 25))

def norm_range(base, cv):
    """Normal range HRV4Training: baseline ± 0.5×CV%×baseline."""
    if pd.isna(base) or pd.isna(cv) or base <= 0: return None, None
    m = 0.5 * (cv / 100) * base
    return base - m, base + m

def calcular_swc(base, cv):
    """SWC = 0.5 × CV% × Baseline / 100  (Hopkins et al. 2009)"""
    if pd.isna(base) or pd.isna(cv) or base <= 0 or cv <= 0: return None
    return 0.5 * (cv / 100) * base

# ── Limpeza de dados ──────────────────────────────────────────────────────────

def remove_zscore(s, thr=3.0):
    v = pd.to_numeric(s, errors='coerce')
    ok = v[v.notna()]
    if len(ok) < 4: return v
    z = np.abs(scipy_stats.zscore(ok))
    v.loc[ok.index[z > thr]] = np.nan
    return v

def remove_zeros(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns: df.loc[df[c] == 0, c] = np.nan
    return df

def fill_missing(df, col, w=7):
    if col not in df.columns: return df
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    mask = df[col].isna()
    if mask.any():
        r = df[col].rolling(w, min_periods=2, center=True).mean()
        df.loc[mask, col] = r[mask]
        mask2 = df[col].isna()
        if mask2.any():
            r2 = df[col].rolling(14, min_periods=2, center=True).mean()
            df.loc[mask2, col] = r2[mask2]
        mask3 = df[col].isna()
        if mask3.any(): df.loc[mask3, col] = df[col].mean()
    return df

# ── Actividades ───────────────────────────────────────────────────────────────

def filtrar_principais(df):
    """Remove WeightTraining em dias com outras modalidades."""
    if len(df) == 0: return df
    df = df.copy()
    df['_t'] = df['type'].apply(norm_tipo)
    df['_d'] = pd.to_datetime(df['Data']).dt.date
    por_dia = df.groupby('_d')['_t'].apply(list).to_dict()
    df['_ok'] = df.apply(
        lambda r: not (r['_t'] == 'WeightTraining' and len(por_dia.get(r['_d'], [])) > 1), axis=1)
    return df[df['_ok']].drop(columns=['_t', '_d', '_ok'])

def add_tempo(df):
    """Adiciona colunas mes, ano, trimestre."""
    if len(df) > 0 and 'Data' in df.columns:
        df = df.copy()
        df['mes']       = pd.to_datetime(df['Data']).dt.strftime('%Y-%m')
        df['ano']       = pd.to_datetime(df['Data']).dt.year
        df['trimestre'] = pd.to_datetime(df['Data']).dt.to_period('Q').astype(str)
    return df

def classificar_rpe(v):
    if pd.isna(v): return None
    v = float(v)
    if 1 <= v <= 4.9: return 'Leve'
    if 5 <= v <= 6.9: return 'Moderado'
    if 7 <= v <= 10:  return 'Pesado'
    return None

# ── CTL/ATL/FTLM ─────────────────────────────────────────────────────────────

def calcular_series_carga(df_act, ate_hoje=True):
    """
    Calcula série diária de CTL/ATL/TSB/FTLM a partir das atividades.

    Métrica: session_rpe = (moving_time_min) × RPE
    Esta é a escala usada no código original Python/SQLite (CTL ~350-450).

    Retorna DataFrame com colunas: Data, load_val, CTL, ATL, TSB, FTLM
    e o gamma óptimo do FTLM.
    """
    df = filtrar_principais(df_act).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    if 'moving_time' not in df.columns or 'rpe' not in df.columns:
        return pd.DataFrame(), 0.30
    df['rpe_fill'] = pd.to_numeric(df['rpe'], errors='coerce')
    df['rpe_fill'] = df['rpe_fill'].fillna(df['rpe_fill'].median())
    df['load_val'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * df['rpe_fill']
    df['load_val'] = df['load_val'].fillna(0)

    ld = df.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
    if ate_hoje:
        idx = pd.date_range(ld['Data'].min(), datetime.now().date())
    else:
        idx = pd.date_range(ld['Data'].min(), ld['Data'].max())
    ld = ld.set_index('Data').reindex(idx, fill_value=0).reset_index()
    ld.columns = ['Data', 'load_val']

    # CTL (42d EMA) / ATL (7d EMA) — igual ao Intervals.icu / original SQLite
    ld['CTL'] = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL'] = ld['load_val'].ewm(span=7,  adjust=False).mean()
    ld['TSB'] = ld['CTL'] - ld['ATL']

    # FTLM — gamma optimizado por correlação com a carga
    best_g, best_r = 0.30, -1.0
    for g in np.arange(0.25, 0.36, 0.01):
        ema = ld['load_val'].ewm(alpha=g, adjust=False).mean()
        if ema.std() > 0:
            r = abs(np.corrcoef(ld['load_val'].values, ema.values)[0, 1])
            if r > best_r: best_r, best_g = r, g
    ld['FTLM'] = ld['load_val'].ewm(alpha=best_g, adjust=False).mean()

    return ld, best_g

def calcular_bpe(dw, metrica='hrv', baseline_dias=60):
    """
    BPE Z-Score semanal (metodologia do artigo Hopkins 2009 / Plews 2013).
    Baseline FIXO = últimos N dias do período total.
    Z = (Média_semanal − Baseline) / SWC
    """
    if metrica not in dw.columns or len(dw) < 14: return pd.DataFrame()
    df = dw.copy().sort_values('Data')
    df['Data'] = pd.to_datetime(df['Data'])
    df[metrica] = pd.to_numeric(df[metrica], errors='coerce')
    df_clean = df.dropna(subset=[metrica])
    if len(df_clean) < 14: return pd.DataFrame()

    # Baseline fixo sobre os últimos N dias
    n_base  = min(baseline_dias, len(df_clean))
    bdata   = df_clean[metrica].tail(n_base)
    base    = bdata.mean()
    std_b   = bdata.std()
    if pd.isna(base) or base <= 0 or pd.isna(std_b) or std_b <= 0: return pd.DataFrame()
    cv   = (std_b / base) * 100
    swc  = calcular_swc(base, cv)
    if swc is None or swc <= 0: return pd.DataFrame()

    df_clean['semana'] = df_clean['Data'].dt.to_period('W')
    rows = []
    for sem in sorted(df_clean['semana'].unique()):
        df_sem = df_clean[df_clean['semana'] == sem]
        media_sem = df_sem[metrica].mean()
        if pd.isna(media_sem): continue
        rows.append({
            'ano_semana':    str(sem),
            'media_semanal': media_sem,
            'baseline':      base,
            'swc':           swc,
            'cv_percent':    cv,
            'zscore':        (media_sem - base) / swc,
            'n_dias':        len(df_sem),
        })
    return pd.DataFrame(rows)

def calcular_recovery(dw):
    """Recovery Score composto (0-100). Igual ao original Python."""
    if len(dw) == 0: return pd.DataFrame()
    df = dw.copy().sort_values('Data')
    df['hrv_baseline'] = df['hrv'].rolling(14, min_periods=7).mean()
    df['rhr_baseline'] = df['rhr'].rolling(14, min_periods=7).mean() if 'rhr' in df.columns else np.nan
    df['hrv_cv_7d']    = cvr(df['hrv'], 7)
    df['hrv_cv_30d']   = cvr(df['hrv'], 30)
    rows = []
    for _, row in df.iterrows():
        hv = row.get('hrv'); hb = row.get('hrv_baseline'); cv = row.get('hrv_cv_7d')
        rv = row.get('rhr'); rb = row.get('rhr_baseline')
        # HRV (30%)
        hs = 50
        if pd.notna(hv) and pd.notna(hb) and hb > 0 and pd.notna(cv):
            inf_, sup_ = norm_range(hb, cv)
            if inf_ and sup_:
                band = sup_ - inf_
                if hv >= sup_:   hs = min(100, 75 + (hv - sup_) / band * 25 if band > 0 else 75)
                elif hv <= inf_: hs = max(0,   40 - (inf_ - hv) / band * 40 if band > 0 else 40)
                else:            hs = 50 + ((hv - inf_) / band * 25 if band > 0 else 0)
        # RHR (15%)
        rs = 50
        if pd.notna(rv) and pd.notna(rb) and rb > 0:
            pct = (rv - rb) / rb * 100
            rs = 90 if pct < -10 else 75 if pct < -5 else 55 if pct < 5 else 35 if pct < 10 else 20
        sl = conv_15(row.get('sleep_quality'))
        fa = conv_15(row.get('fatiga'))
        st = conv_15(row.get('stress'))
        hu = conv_15(row.get('humor'))
        so = conv_15(row.get('soreness'))
        score = hs*0.30 + rs*0.15 + sl*0.20 + fa*0.10 + st*0.10 + hu*0.05 + so*0.05 + 50*0.05
        i_, s_ = (norm_range(hb, cv) if pd.notna(hb) and pd.notna(cv) else (None, None))
        rows.append({'Data': row['Data'], 'recovery_score': score,
                     'hrv': hv, 'hrv_baseline': hb, 'hrv_cv_7d': cv, 'hrv_cv_30d': row.get('hrv_cv_30d'),
                     'normal_range_inf': i_, 'normal_range_sup': s_,
                     'hrv_component': hs, 'rhr_component': rs,
                     'sleep_component': sl, 'fatiga_component': fa, 'stress_component': st})
    return pd.DataFrame(rows)

def calcular_polinomios_carga(df_act_full):
    """
    Polynomial fit CTL/ATL Overall e por modalidade.
    Retorna dict com resultados + '_ld' (série completa).
    """
    df = filtrar_principais(df_act_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    ld, _ = calcular_series_carga(df_act_full)
    if len(ld) == 0: return None
    ld['dias_num'] = (ld['Data'] - ld['Data'].min()).dt.days.values
    res = {'overall': {'CTL': {}, 'ATL': {}}, '_ld': ld}
    for met in ['CTL', 'ATL']:
        x, y = ld['dias_num'].values, ld[met].values
        for grau in [2, 3]:
            try:
                z = np.polyfit(x, y, grau); p = np.poly1d(z)
                r2 = np.corrcoef(y, p(x))[0, 1] ** 2
                res['overall'][met][f'grau{grau}'] = {'poly': p, 'r2': r2, 'x': x, 'y': y}
            except Exception: pass
    if 'moving_time' in df.columns and 'rpe' in df.columns:
        df['rpe_fill'] = pd.to_numeric(df['rpe'], errors='coerce').fillna(df['rpe'].median())
        df['load_val'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * df['rpe_fill']
        for tipo in ['Bike', 'Run', 'Row', 'Ski']:
            dt = df[df['type'] == tipo].copy()
            if len(dt) < 5: continue
            lt = dt.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
            idx_t = pd.date_range(lt['Data'].min(), ld['Data'].max())
            lt = lt.set_index('Data').reindex(idx_t, fill_value=0).reset_index(); lt.columns = ['Data', 'lv']
            lt['CTL'] = lt['lv'].ewm(span=42, adjust=False).mean()
            lt['ATL'] = lt['lv'].ewm(span=7,  adjust=False).mean()
            lt['dn']  = (lt['Data'] - lt['Data'].min()).dt.days.values
            res[f'tipo_{tipo}'] = {'CTL': {}, 'ATL': {}}
            for met in ['CTL', 'ATL']:
                x, y = lt['dn'].values, lt[met].values
                for grau in [2, 3]:
                    try:
                        z = np.polyfit(x, y, grau); p = np.poly1d(z)
                        r2 = np.corrcoef(y, p(x))[0, 1] ** 2
                        res[f'tipo_{tipo}'][met][f'grau{grau}'] = {'poly': p, 'r2': r2, 'x': x, 'y': y}
                    except Exception: pass
    return res

def analisar_falta_estimulo(df_act_full, janela_dias=14, baseline_dias=90):
    """
    Need Score v4 por modalidade.
    Load = duração_min × RPE

    Componentes base:
      A (25%) — Share FTLM deficit vs baseline 90d  [floor 10%]
      B (25%) — Quality deficit: carga_Z3/carga_total vs baseline
      C (20%) — Load deficit na janela vs load típico baseline
      D (20%) — FTLM slope_pct variação total no baseline [floor FTLM ini]
      E (10%) — Interacção A×B normalizada

    Scores derivados (coerentes com A/B/C/D):
      Need_volume    = A + C  (défice de carga total)
      Need_intensity = B + D×0.5  (défice de qualidade; D penaliza queda fitness)

    Overload por score acumulado (≥2 de 3 critérios):
      share_ratio > 1.4  → +1
      quality_ratio > 1.4 → +1
      slope_pct_janela > 0.15 → +1 (crescimento rápido na janela activa)
    Se overload: Need_final × 0.5, flag ⚠️, intensidade=LOW
    """
    MODS = ['Bike', 'Row', 'Ski', 'Run']
    df   = filtrar_principais(df_act_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    if 'moving_time' not in df.columns or 'rpe' not in df.columns:
        return None, None

    df['dur_min']  = pd.to_numeric(df['moving_time'], errors='coerce') / 60
    df['rpe_n']    = pd.to_numeric(df['rpe'], errors='coerce')
    df['load']     = df['dur_min'] * df['rpe_n']
    df['load_z3']  = np.where(df['rpe_n'] >= 7, df['load'], 0.0)

    hoje     = pd.Timestamp.now().normalize()
    ini_jan  = hoje - pd.Timedelta(days=janela_dias)
    ini_base = hoje - pd.Timedelta(days=baseline_dias)

    # ── FTLM por modalidade — histórico completo (não recalcular na janela curta)
    all_dates = pd.date_range(df['Data'].min(), hoje, freq='D')
    ftlm_mods = {}
    for mod in MODS:
        dm = df[df['type'] == mod].copy()
        ld = (dm.groupby('Data')['load'].sum()
                .reindex(all_dates, fill_value=0.0)
                .reset_index())
        ld.columns = ['Data', 'load']
        ld['FTLM'] = ld['load'].ewm(span=28, adjust=False).mean()
        ftlm_mods[mod] = ld.set_index('Data')

    ftlm_total = sum(ftlm_mods[m]['FTLM'] for m in MODS).replace(0, np.nan)

    results    = {}
    debug_rows = []

    for mod in MODS:
        fm = ftlm_mods[mod]

        # ── A — Share FTLM deficit ──────────────────────────────────────────
        share_jan  = float((fm.loc[ini_jan:hoje,  'FTLM'] /
                            ftlm_total.loc[ini_jan:hoje]).mean())
        share_base = float((fm.loc[ini_base:hoje, 'FTLM'] /
                            ftlm_total.loc[ini_base:hoje]).mean())
        share_jan  = 0.0 if np.isnan(share_jan)  else share_jan
        share_base = 0.0 if np.isnan(share_base) else share_base
        floor_peso = min(1.0, share_base / 0.10) if share_base < 0.10 else 1.0
        A_raw = max(0.0, (share_base - share_jan) / share_base) if share_base > 0 else 0.0
        A     = min(A_raw, 1.0) * floor_peso

        # ── B — Quality deficit (carga_Z3 / carga_total) ───────────────────
        dm_jan  = df[(df['type']==mod) & (df['Data'] >= ini_jan)]
        dm_base = df[(df['type']==mod) & (df['Data'] >= ini_base)]
        def _q(d_):
            ct = d_['load'].sum()
            return float(d_['load_z3'].sum() / ct) if ct > 0 else 0.0
        q_jan  = _q(dm_jan)
        q_base = _q(dm_base)
        B_raw  = max(0.0, (q_base - q_jan) / q_base) if q_base > 0 else 0.0
        B      = min(B_raw, 1.0)

        # ── C — Load deficit na janela vs load típico ───────────────────────
        load_jan    = dm_jan['load'].sum()
        load_base   = dm_base['load'].sum()
        n_dias_base = max(1, (hoje - ini_base).days)
        load_tipico = (load_base / n_dias_base) * janela_dias
        C_raw = max(0.0, (load_tipico - load_jan) / load_tipico) if load_tipico > 0 else 0.0
        C     = min(C_raw, 1.0)

        # ── D — FTLM slope_pct (variação % total no baseline) ──────────────
        ftlm_base_series = fm.loc[ini_base:hoje, 'FTLM'].dropna()
        slope_pct = 0.0
        D = 0.0
        if len(ftlm_base_series) >= 10:
            ftlm_ini = float(ftlm_base_series.iloc[0])
            ftlm_fim = float(ftlm_base_series.iloc[-1])
            threshold_ini = max(1.0, ftlm_fim * 0.05)
            if ftlm_ini >= threshold_ini and ftlm_ini > 0:
                slope_pct = (ftlm_fim - ftlm_ini) / ftlm_ini
                if slope_pct < 0:
                    D = min(1.0, abs(slope_pct) * 2)
                elif slope_pct > 0.30:
                    D = min(0.5, (slope_pct - 0.30) * 2)

        # ── E — Interacção A×B ──────────────────────────────────────────────
        if A_raw > 0 and B_raw > 0:    fator_e = 1.3
        elif A_raw > 0 or B_raw > 0:   fator_e = 0.7
        else:                           fator_e = 0.0
        E = float(np.clip((A_raw + B_raw) / 2 * fator_e, 0.0, 1.0))

        # ── Need Score base ─────────────────────────────────────────────────
        need = min(100.0,
                   A * 100 * 0.25 +
                   B * 100 * 0.25 +
                   C * 100 * 0.20 +
                   D * 100 * 0.20 +
                   E * 100 * 0.10)

        # ── Need_volume e Need_intensity — camada de prescrição (read-only) ──
        # NÃO alteram Need_score. Apenas interpretam A/B/C/D.
        # Pesos: A×0.6+C×0.4 para volume | B×0.7+D×0.3 para intensidade
        need_vol = min(100.0, A * 100 * 0.60 + C * 100 * 0.40)
        need_int = min(100.0, B * 100 * 0.65 + D * 100 * 0.35)

        prio_base = 'ALTA' if need >= 70 else 'MÉDIA' if need >= 40 else 'BAIXA'

        # ── Overload por score acumulado ────────────────────────────────────
        share_ratio   = share_jan / share_base if share_base > 0 else 0.0
        quality_ratio = q_jan / q_base         if q_base    > 0 else 0.0

        # slope_pct na janela activa (para overload — diferente do D que usa baseline)
        ftlm_jan_series = fm.loc[ini_jan:hoje, 'FTLM'].dropna()
        slope_pct_jan = 0.0
        if len(ftlm_jan_series) >= 3:
            fi = float(ftlm_jan_series.iloc[0])
            ff = float(ftlm_jan_series.iloc[-1])
            if fi > 0:
                slope_pct_jan = (ff - fi) / fi

        score_overload = 0
        if share_ratio   > 1.4: score_overload += 1
        if quality_ratio > 1.4: score_overload += 1
        if slope_pct_jan > 0.15: score_overload += 1
        overload = score_overload >= 2

        # Overload → reduzir intensidade (não zerar, não remover)
        need_int_prescr = min(100.0, need_int * 0.65) if overload else need_int

        # C_reforçado — debug, NÃO substitui C original
        datas_mod_dbg  = df[df['type']==mod]['Data'].sort_values()
        ultima_dbg     = datas_mod_dbg.max() if len(datas_mod_dbg) > 0 else pd.NaT
        dias_sem       = int((hoje - ultima_dbg).days) if pd.notna(ultima_dbg) else 999
        gap_tipico_dbg = float(max(1, datas_mod_dbg[datas_mod_dbg >= ini_base]
                                   .diff().dt.days.dropna().median())
                               if len(datas_mod_dbg[datas_mod_dbg >= ini_base]) >= 2
                               else 7.0)
        gap_score      = min(1.0, dias_sem / (gap_tipico_dbg * 2))
        c_reforcado    = max(C, gap_score)  # debug — observação apenas

        # ── Prescrição textual com contexto temporal ───────────────────────
        ALTO = 50
        # gap_ratio: quanto tempo passou vs padrão (cap 3.0 para evitar distorções)
        gap_ratio = min(dias_sem / gap_tipico_dbg, 3.0) if gap_tipico_dbg > 0 else 0.0

        if overload:
            prescricao = "🟢 Treino leve/moderado (overload — reduzir intensidade)"

        elif need_int_prescr >= ALTO and need_vol >= ALTO:
            # Ambos altos → sessão completa independentemente do gap
            # gap_ratio NÃO reduz prescrição quando ambos são críticos
            prescricao = "🔴 Sessão completa (volume + intensidade)"

        elif need_int_prescr >= ALTO and need_vol < ALTO:
            # Intensidade alta + volume baixo → modular pelo gap_ratio
            if gap_ratio <= 1.0:
                # Treinou recentemente → pode aguentar intensidade
                prescricao = "🟠 Sessão intensa/curta (fresco — défice qualidade)"
            elif gap_ratio <= 2.0:
                # Algum tempo sem treinar → reentrée gradual
                prescricao = "🟡 Sessão mista vol+int (reentrée gradual)"
            else:
                # Gap longo (>2× padrão, cap 3×) → base primeiro
                prescricao = "🔵 Sessão de base/reentrée (gap longo — não forçar intensidade)"

        elif need_vol >= ALTO and need_int_prescr < ALTO:
            prescricao = "🔵 Sessão de volume/base (défice de carga)"

        else:
            prescricao = "⚪ Manutenção ou descanso"

        results[mod] = dict(
            need_score=need, prioridade=prio_base, overload=overload,
            overload_score=score_overload,
            need_vol=round(need_vol, 1),
            need_int=round(need_int, 1),
            need_int_prescr=round(need_int_prescr, 1),
            prescricao=prescricao,
            dias_sem=dias_sem, gap_score=round(gap_score, 2),
            c_reforcado=round(c_reforcado, 3),
            share_actual=share_jan, share_hist=share_base,
            quality_actual=q_jan, quality_hist=q_base,
            load_jan=load_jan, load_tipico=load_tipico,
            ftlm_slope_pct=slope_pct, ftlm_slope_jan=slope_pct_jan,
            A=A, B=B, C=C, D=D, E=E, floor_peso=floor_peso)

        debug_rows.append({
            'Modalidade':           mod,
            'Need Score':           round(need, 1),
            'Need Volume':          round(need_vol, 1),
            'Need Intensity':       round(need_int, 1),
            'Need Int (prescrição)':round(need_int_prescr, 1),
            'Prescrição':           prescricao,
            'dias_sem_sessao':      dias_sem,
            'gap_score':            round(gap_score, 2),
            'C_reforçado':          round(c_reforcado, 3),
            'Prioridade base':      prio_base,
            'Overload score':       score_overload,
            'Overload':             '⚠️ SIM' if overload else 'não',
            'A Share actual%':      round(share_jan  * 100, 1),
            'A Share hist90d%':     round(share_base * 100, 1),
            'A Deficit%':           round(A_raw * 100, 1),
            'A Floor peso':         round(floor_peso, 2),
            'A contribuição':       round(A * 100 * 0.25, 1),
            'B Quality actual%':    round(q_jan  * 100, 1),
            'B Quality hist90d%':   round(q_base * 100, 1),
            'B Deficit%':           round(B_raw * 100, 1),
            'B contribuição':       round(B * 100 * 0.25, 1),
            'C Load janela':        round(load_jan, 1),
            'C Load típico':        round(load_tipico, 1),
            'C Deficit%':           round(C_raw * 100, 1),
            'C contribuição':       round(C * 100 * 0.20, 1),
            'D FTLM ini':           round(ftlm_ini if len(ftlm_base_series)>=10 else 0, 1),
            'D FTLM fim':           round(ftlm_fim if len(ftlm_base_series)>=10 else 0, 1),
            'D slope_pct%':         round(slope_pct * 100, 1),
            'D slope_jan%':         round(slope_pct_jan * 100, 1),
            'D contribuição':       round(D * 100 * 0.20, 1),
            'E Fator VQ':           fator_e,
            'E contribuição':       round(E * 100 * 0.10, 1),
        })

    results_sorted = dict(sorted(
        results.items(), key=lambda x: x[1]['need_score'], reverse=True))
    return results_sorted, pd.DataFrame(debug_rows)



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: data_loader.py
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# data_loader.py — ATHELTICA Dashboard
# Autenticação Google Sheets, carregamento e preprocessing.
# ════════════════════════════════════════════════════════════════════════════════

# ── Autenticação ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_gc():
    """Autentica Google Sheets com Service Account em st.secrets."""
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Erro autenticação Google: {e}")
        st.info("Configura as credenciais em Settings → Secrets do Streamlit Cloud.")
        return None

# ── Carregamento ──────────────────────────────────────────────────────────────

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

def _preencher_faltantes_lookback(df, coluna):
    """
    Preenche NaN com mediana dos 7 dias ANTERIORES (lookback, sem data leakage).
    Fallback: mediana 14 dias anteriores → média global.
    Igual ao método preencher_faltantes() do ATHELTICA v12 original.
    """
    if coluna not in df.columns: return df
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').reset_index(drop=True)
    for idx, row in df.iterrows():
        if pd.notna(row[coluna]): continue
        data_ref = row['Data']
        # Janela 7 dias anteriores
        w7 = df[(df['Data'] >= data_ref - timedelta(days=7)) &
                (df['Data'] <  data_ref) &
                (df[coluna].notna())][coluna]
        if len(w7) >= 2:
            df.at[idx, coluna] = w7.median(); continue
        # Janela 14 dias anteriores
        w14 = df[(df['Data'] >= data_ref - timedelta(days=14)) &
                 (df['Data'] <  data_ref) &
                 (df[coluna].notna())][coluna]
        if len(w14) >= 3:
            df.at[idx, coluna] = w14.median(); continue
        # Fallback: média global
        media = df[df[coluna].notna()][coluna].mean()
        if pd.notna(media):
            df.at[idx, coluna] = media
    return df


def preproc_wellness(df):
    """
    Limpeza wellness — igual ao ATHELTICA v12 DataPreprocessor.limpar_wellness():
    1. Ordenar por Data
    2. Remover duplicatas por Data (keep=first)
    3. Z-score threshold=3.0 → NaN (picos)
    4. Zeros inválidos → NaN (hrv, rhr, sleep_hours)
    5. Preencher faltantes: mediana 7d anteriores → 14d → média global
    """
    if len(df) == 0: return df
    df = df.copy().sort_values('Data')
    df = df.drop_duplicates(subset=['Data'], keep='first')
    # [1] Remover picos Z-score
    for c in [c for c in ['hrv', 'rhr', 'sleep_hours', 'sleep_quality',
                           'stress', 'fatiga', 'humor', 'soreness', 'peso', 'fat']
              if c in df.columns]:
        df[c] = remove_zscore(df[c], 3.0)
    # [2] Zeros inválidos → NaN
    df = remove_zeros(df, ['hrv', 'rhr', 'sleep_hours'])
    # [3] Preencher faltantes com lookback (sem data leakage)
    for c in [c for c in ['hrv', 'rhr', 'sleep_quality', 'fatiga',
                           'stress', 'humor', 'soreness']
              if c in df.columns]:
        df = _preencher_faltantes_lookback(df, c)
    return df.reset_index(drop=True)

def preproc_ativ(df):
    """
    Limpeza atividades — igual ao ATHELTICA v12 DataPreprocessor.limpar_training():
    1. Ordenar por Data
    2. Remover duplicatas por [Data, type, moving_time]
    3. Padronizar tipos via TYPE_MAP
    4. Remover tipos inválidos (não em VALID_TYPES)
    5. Z-score threshold=3.5 → NaN em icu_eftp E AllWorkFTP
    6. Zeros → NaN em moving_time, icu_eftp, AllWorkFTP
    7. Remover moving_time ≤ 60s
    """
    if len(df) == 0: return df
    df = df.copy().sort_values('Data')
    # [1] Deduplicar
    sub = [c for c in ['Data', 'type', 'moving_time'] if c in df.columns]
    if len(sub) >= 2: df = df.drop_duplicates(subset=sub, keep='first')
    # [2] Padronizar tipos
    if 'type' in df.columns: df['type'] = df['type'].apply(norm_tipo)
    # [3] Filtrar tipos válidos
    if 'type' in df.columns: df = df[df['type'].isin(VALID_TYPES)]
    # [4] Z-score picos icu_eftp e AllWorkFTP (threshold=3.5, igual ao original)
    if 'icu_eftp'    in df.columns: df['icu_eftp']    = remove_zscore(df['icu_eftp'],    3.5)
    if 'AllWorkFTP'  in df.columns: df['AllWorkFTP']  = remove_zscore(df['AllWorkFTP'],  3.5)
    # [5] Zeros → NaN (rpe=0 também é inválido — sem esforço não é sessão)
    df = remove_zeros(df, ['moving_time', 'icu_eftp', 'AllWorkFTP', 'rpe'])
    # [6] Remover duração ≤ 60s
    if 'moving_time' in df.columns:
        df = df[pd.to_numeric(df['moving_time'], errors='coerce') > 60]
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
    Carrega AquecSki, AquecBike, AquecRow via gviz CSV.
    Data cleaning idêntico ao código original Python:
    - Renomeia colunas Unnamed (AquecRow)
    - Remove colunas 100% vazias
    - Converte vazios/nan/null → NaN
    - Converte numéricos, zeros → NaN
    - Remove ruídos: HR fora 40-220, O2 fora 20-100, Drag fora 80-200, Pwr fora 0.3-3.0
    - Normaliza coluna DATA e remove linhas sem data
    """
    COLS_NAO_NUM = ['Mês', 'Mes', 'Fase', 'DATA', 'Data', 'Treino_antes', 'Atividade']
    AQUECROW_RENAME = {
        'Unnamed: 2': 'DATA',       'Treino antes': 'Treino_antes',
        'Unnamed: 4': 'HR_140W',    'Unnamed: 5':   'HR_160W',
        'Unnamed: 6': 'HR_180W',    'Unnamed: 7':   'HR_200W',
        'Unnamed: 8': 'HR_Pwr_140w','Unnamed: 9':   'HR_Pwr_160w',
        'Unnamed: 10':'HR_Pwr_180w','Unnamed: 11':  'O2_140W',
        'Unnamed: 12':'O2_160W',    'Unnamed: 13':  'O2_180W',
        'Drag Factor':'Drag_Factor',
    }
    dfs = {}
    for aba in ANNUAL_SHEETS:
        url = (f"https://docs.google.com/spreadsheets/d/{ANNUAL_SPREADSHEET_ID}"
               f"/gviz/tq?tqx=out:csv&sheet={aba}")
        try:
            df = pd.read_csv(url)
            df.columns = [str(c).strip() for c in df.columns]
            if aba == 'AquecRow':
                df = df.rename(columns={k:v for k,v in AQUECROW_RENAME.items() if k in df.columns})
            # Normalizar nomes genéricos
            rename_gen = {}
            for col in df.columns:
                if col in ['Mês','Mes']: rename_gen[col] = 'Mês'
                elif col in ['DATA','Data']: rename_gen[col] = 'DATA'
                elif 'Drag' in col and 'Factor' in col: rename_gen[col] = 'Drag_Factor'
                elif 'Treino' in col: rename_gen[col] = 'Treino_antes'
            if rename_gen: df = df.rename(columns=rename_gen)
            # Remove colunas 100% vazias
            df = df.dropna(axis=1, how='all')
            # Strings inválidas → NaN
            df = df.replace(['', ' ', 'nan', 'NaN', 'null', 'NULL', 'None'], np.nan)
            # Numéricos + zeros → NaN
            for col in df.columns:
                if col not in COLS_NAO_NUM:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].replace({0.0: np.nan, 0: np.nan})
            # Remover ruídos por tipo (igual ao original Python)
            for col in df.columns:
                if col in COLS_NAO_NUM: continue
                cu = col.upper()
                if 'HR' in cu and 'PWR' not in cu and 'DRAG' not in cu:
                    df[col] = df[col].where((df[col] >= 40) & (df[col] <= 220))
                elif 'O2' in cu:
                    df[col] = df[col].where((df[col] >= 20) & (df[col] <= 100))
                elif 'DRAG' in cu:
                    df[col] = df[col].where((df[col] >= 80) & (df[col] <= 200))
                elif 'PWR' in cu:
                    df[col] = df[col].where((df[col] >= 0.3) & (df[col] <= 3.0))
            # Normalizar DATA
            if 'DATA' in df.columns:
                df['DATA'] = pd.to_datetime(df['DATA'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['DATA']).sort_values('DATA')
            df['Atividade'] = aba
            dfs[aba] = df.reset_index(drop=True)
        except Exception:
            dfs[aba] = pd.DataFrame()
    validos = [d for d in dfs.values() if len(d) > 0]
    df_all = pd.concat(validos, ignore_index=True) if validos else pd.DataFrame()
    return dfs, df_all


@st.cache_data(ttl=3600, show_spinner="A carregar dados corporais...")
def carregar_corporal():
    """Carrega aba Consolidado_Comida. Lida com vírgula decimal (PT-BR)."""
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        ws  = gc.open_by_url(FOOD_URL).worksheet("Consolidado_Comida")
        df  = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]
        df  = df.dropna(how='all')
        df.columns = [c.strip() for c in df.columns]
        cd  = detectar_col(df, ['Data', 'data', 'Date'])
        if not cd: return pd.DataFrame()
        df['Data'] = df[cd].apply(parse_date)
        df = df.dropna(subset=['Data']).sort_values('Data')
        # Remover datas futuras — dados só até hoje
        df = df[df['Data'] <= pd.Timestamp.now().normalize()]
        for col in ['Peso','BF','Calorias','Carb','Fat','Ptn',
                    'Carb_perc','Fat_perc','Ptn_perc','Net']:
            if col in df.columns: df[col] = df[col].apply(br_float)
        # Remover linhas sem nenhum dado numérico (linhas futuras vazias)
        num_cols = [c for c in ['Peso','BF','Calorias','Carb','Fat','Ptn','Net']
                    if c in df.columns]
        if num_cols:
            df = df[df[num_cols].notna().any(axis=1)]
        # Ranges fisiológicos
        for col, lo, hi in [('Peso',30,200),('BF',3,50),('Calorias',500,6000),
                             ('Carb',0,800),('Fat',0,400),('Ptn',0,400),
                             ('Net',-2000,4000)]:
            if col in df.columns:
                df.loc[~df[col].between(lo, hi, inclusive='both'), col] = np.nan
        # Z-score 3.0 nos campos principais
        for col in ['Peso','BF','Calorias','Net']:
            if col in df.columns: df[col] = remove_zscore(df[col], 3.0)
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro ao carregar dados corporais: {e}")
        return pd.DataFrame()


def tab_visao_geral(dw, da, di, df_, da_full=None, wc_full=None, dc=None):
    st.header("📊 Visão Geral")

    # ── KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    horas = (da['moving_time'].sum() / 3600) if 'moving_time' in da.columns and len(da) > 0 else None
    hrv_m = dw['hrv'].dropna().tail(7).mean() if 'hrv' in dw.columns and len(dw) > 0 else None
    rhr_u = dw['rhr'].dropna().iloc[-1] if 'rhr' in dw.columns and len(dw) > 0 and dw['rhr'].notna().any() else None
    c1.metric("🏋️ Sessões",   f"{len(da)}")
    c2.metric("⏱️ Horas",     fmt_dur(horas) if horas else "—")
    c3.metric("💚 HRV (7d)", f"{hrv_m:.0f} ms" if hrv_m else "—")
    c4.metric("❤️ RHR",       f"{rhr_u:.0f} bpm" if rhr_u else "—")
    st.markdown("---")

    # ── Semana ACTUAL (seg→hoje) ──────────────────────────────────────────
    if da_full is not None and len(da_full) > 0:
        _df_sw = da_full.copy()
        _df_sw['Data'] = pd.to_datetime(_df_sw['Data'])
        _df_sw = _df_sw[_df_sw['type'].apply(norm_tipo) != 'WeightTraining']
        _df_sw = filtrar_principais(_df_sw)
        if 'icu_joules' in _df_sw.columns:
            _df_sw['_kj'] = pd.to_numeric(_df_sw['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in _df_sw.columns:
            _df_sw['_kj'] = (pd.to_numeric(_df_sw['power_avg'], errors='coerce') *
                              pd.to_numeric(_df_sw['moving_time'], errors='coerce') / 1000)
        else:
            _df_sw['_kj'] = np.nan
        _df_sw['_mt']  = pd.to_numeric(_df_sw['moving_time'], errors='coerce') / 3600
        _df_sw['_rpe'] = pd.to_numeric(_df_sw['rpe'], errors='coerce')                          if 'rpe' in _df_sw.columns else np.nan

        _hoje_sw  = pd.Timestamp.now().normalize()
        _dow_sw   = _hoje_sw.weekday()
        _sem_ini_sw = _hoje_sw - pd.Timedelta(days=_dow_sw)  # segunda desta semana
        _df_sw_cur = _df_sw[_df_sw['Data'] >= _sem_ini_sw].copy()

        if len(_df_sw_cur) > 0:
            _df_sw_cur['_dia'] = _df_sw_cur['Data'].dt.strftime('%a %d/%m')
            rows_sw = []
            for _, r in _df_sw_cur.sort_values('Data').iterrows():
                rows_sw.append({
                    'Dia':        r['_dia'],
                    'Modalidade': r['type'],
                    'RPE':        f"{r['_rpe']:.0f}" if pd.notna(r['_rpe']) else '—',
                    'KJ':         f"{r['_kj']:.0f}" if pd.notna(r['_kj']) and r['_kj']>0 else '—',
                    'Horas':      fmt_dur(r['_mt']) if pd.notna(r['_mt']) else '—',
                })
            st.subheader("📅 Semana actual")
            st.dataframe(pd.DataFrame(rows_sw), width="stretch", hide_index=True)
        else:
            st.subheader("📅 Semana actual")
            st.info("Sem actividades desde segunda-feira.")
        st.markdown("---")

    # ── HRV-Guided + Recovery Score + Peso/BF ────────────────────────────
    vg_r1, vg_r2, vg_r3, vg_r4, vg_r5 = st.columns(5)

    # ── HRV-Guided Training (LnrMSSD, baseline 14d, ±0.5 SD) ────────────
    hrv_hoje    = None
    hrv_class   = "Sem dados"
    hrv_emoji   = "⚪"
    rec_score   = None
    rec_trend   = ""

    if wc_full is not None and len(wc_full) > 0 and 'hrv' in wc_full.columns:
        _wc = wc_full.copy()
        _wc['Data'] = pd.to_datetime(_wc['Data'])
        _wc = _wc.sort_values('Data')
        _wc['LnrMSSD'] = np.where(_wc['hrv'] > 0, np.log(_wc['hrv']), np.nan)
        _wc = _wc.dropna(subset=['LnrMSSD'])
        if len(_wc) >= 7:
            _wc['bm']  = _wc['LnrMSSD'].rolling(14, min_periods=7).mean()
            _wc['bs']  = _wc['LnrMSSD'].rolling(14, min_periods=7).std()
            _wc['linf']= _wc['bm'] - 0.5 * _wc['bs']
            _wc['lsup']= _wc['bm'] + 0.5 * _wc['bs']
            last = _wc.dropna(subset=['bm']).iloc[-1]
            hrv_hoje = last['hrv'] if 'hrv' in last else None
            if pd.notna(last['bm']) and pd.notna(last['LnrMSSD']):
                if last['linf'] <= last['LnrMSSD'] <= last['lsup']:
                    hrv_class = "HIIT"; hrv_emoji = "🟢"
                else:
                    hrv_class = "Recuperação"; hrv_emoji = "🔴"

        # Recovery Score trend (7d)
        _rec = calcular_recovery(_wc.rename(columns={'Data':'Data'}))
        if len(_rec) >= 7:
            rec_vals = _rec['recovery_score'].dropna()
            if len(rec_vals) >= 7:
                rec_score = rec_vals.iloc[-1]
                rec_mean7 = rec_vals.tail(7).mean()
                rec_mean_prev = rec_vals.iloc[-14:-7].mean() if len(rec_vals) >= 14 else rec_vals.mean()
                if rec_mean7 > rec_mean_prev * 1.03:
                    rec_trend = "↗"
                elif rec_mean7 < rec_mean_prev * 0.97:
                    rec_trend = "↘"
                else:
                    rec_trend = "→"

    with vg_r1:
        st.metric("🧠 HRV-Guided",
                  f"{hrv_emoji} {hrv_class}",
                  f"HRV {hrv_hoje:.0f} ms" if hrv_hoje else None)
    with vg_r2:
        st.metric("🔋 Recovery Score",
                  f"{rec_score:.0f}/100" if rec_score else "—",
                  rec_trend if rec_score else None)

    # ── Peso e BF (rolling 7d vs média 30d, ignorar dias sem dados) ──────
    peso_7d = peso_trend = bf_7d = bf_trend = None
    if dc is not None and len(dc) > 0:
        _dc = dc.copy()
        _dc['Data'] = pd.to_datetime(_dc['Data'])
        _dc = _dc.sort_values('Data')
        hoje_dc = _dc['Data'].max()

        for col, var_7, var_t in [('Peso','peso_7d','peso_trend'),
                                   ('BF',  'bf_7d',  'bf_trend')]:
            if col not in _dc.columns: continue
            serie = _dc[['Data',col]].dropna(subset=[col]).copy()
            if len(serie) < 3: continue
            v7  = serie[serie['Data'] >= hoje_dc - pd.Timedelta(days=7)][col].mean()
            v30 = serie[serie['Data'] >= hoje_dc - pd.Timedelta(days=30)][col].mean()
            if pd.notna(v7) and pd.notna(v30) and v30 > 0:
                diff_pct = (v7 - v30) / v30 * 100
                if   diff_pct >  0.5: trend = f"↗ +{diff_pct:.1f}%"
                elif diff_pct < -0.5: trend = f"↘ {diff_pct:.1f}%"
                else:                 trend = "→ estável"
                if col == 'Peso': peso_7d = v7; peso_trend = trend
                else:             bf_7d   = v7; bf_trend   = trend

    with vg_r3:
        st.metric("⚖️ Peso (7d)",
                  f"{peso_7d:.1f} kg" if peso_7d else "—",
                  peso_trend)
    with vg_r4:
        st.metric("🫁 BF (7d)",
                  f"{bf_7d:.1f}%" if bf_7d else "—",
                  bf_trend)
    with vg_r5:
        pass  # coluna vazia para espaçamento

    st.markdown("---")

    # ── Prioridades + Need Score (visão rápida) ──────────────────────────
    st.subheader("🎯 Próxima Sessão — Prioridades e Necessidade")

    # Controlos de prioridade — partilhados com a aba Análises via keys únicas
    _mods_vg = ['Bike', 'Row', 'Ski', 'Run']
    # Defaults com session_state — lembra as últimas escolhas do utilizador
    _def_preset = st.session_state.get("vg_prio_preset", "Balanceado (K=10)")
    _def_p1     = st.session_state.get("vg_prio1", "Bike")
    _def_p2     = st.session_state.get("vg_prio2", "Row")
    _def_p3     = st.session_state.get("vg_prio3", "Ski")
    _def_p4     = st.session_state.get("vg_prio4", "Run")
    _presets    = ["Conservador (K=6)", "Balanceado (K=10)", "Agressivo (K=15)"]

    vg_c0, vg_c1, vg_c2, vg_c3, vg_c4 = st.columns([1, 1, 1, 1, 1])
    with vg_c0:
        vg_preset = st.selectbox("Preset K", _presets,
            index=_presets.index(_def_preset) if _def_preset in _presets else 1,
            key="vg_prio_preset")
        vg_K = {"Conservador (K=6)":6,"Balanceado (K=10)":10,"Agressivo (K=15)":15}[vg_preset]
    with vg_c1:
        vg_p1 = st.selectbox("🥇 P1 Foco", _mods_vg,
            index=_mods_vg.index(_def_p1) if _def_p1 in _mods_vg else 0,
            key="vg_prio1")
    with vg_c2:
        vg_p2 = st.selectbox("🥈 P2 Foco", _mods_vg,
            index=_mods_vg.index(_def_p2) if _def_p2 in _mods_vg else 1,
            key="vg_prio2")
    with vg_c3:
        vg_p3 = st.selectbox("🥉 P3 Manutenção", _mods_vg,
            index=_mods_vg.index(_def_p3) if _def_p3 in _mods_vg else 2,
            key="vg_prio3")
    with vg_c4:
        vg_p4 = st.selectbox("4️⃣  P4 Manutenção", _mods_vg,
            index=_mods_vg.index(_def_p4) if _def_p4 in _mods_vg else 3,
            key="vg_prio4")

    vg_prio_rank  = {vg_p1:1, vg_p2:2, vg_p3:3, vg_p4:4}
    vg_grupo_foco = {vg_p1, vg_p2}
    vg_grupo_man  = {vg_p3, vg_p4}

    if da_full is not None and len(da_full) > 0:
        vg_res, _ = analisar_falta_estimulo(da_full, janela_dias=14)
        if vg_res:
            rows_f, rows_m = [], []
            for mod, d in vg_res.items():
                rank   = vg_prio_rank.get(mod, 4)
                peso   = (4 + 1 - rank) / 4
                bonus  = peso * vg_K * (1 - d['need_score'] / 100)
                nf     = d['need_score'] + bonus
                ol_flag= d.get('overload', False)
                if ol_flag:   nf *= 0.5
                if mod in vg_grupo_man: nf = min(nf, 40)
                nf = max(nf, 10)
                pf = ('ALTA' if nf>=70 else 'MÉDIA' if nf>=40 else 'BAIXA')
                row_d = {
                    'Modalidade': f"{'🎯' if mod in vg_grupo_foco else '🔧'} {mod}"
                                  + (' ⚠️' if ol_flag else ''),
                    'Need':       f"{d['need_score']:.0f}",
                    'Final':      f"{nf:.0f}",
                    'Vol/Int':    f"{d.get('need_vol',0):.0f}/{d.get('need_int_prescr',0):.0f}",
                    'Prescrição': d.get('prescricao','—'),
                }
                if mod in vg_grupo_foco: rows_f.append((nf, row_d))
                else:                    rows_m.append((nf, row_d))

            rows_f.sort(key=lambda x: x[0], reverse=True)
            rows_m.sort(key=lambda x: x[0], reverse=True)

            col_f, col_m = st.columns(2)
            with col_f:
                st.markdown("**🎯 Foco**")
                if rows_f:
                    st.dataframe(pd.DataFrame([r for _,r in rows_f]),
                                 width="stretch", hide_index=True)
                    # Sugestão principal
                    top    = rows_f[0]
                    top_mod= top[1]['Modalidade'].replace('🎯 ','').replace('🔧 ','').replace(' ⚠️','')
                    top_d  = vg_res.get(top_mod, {})
                    if top_d.get('overload'):
                        st.warning(f"⚠️ **{top_mod}**: {top_d.get('prescricao','—')}")
                    else:
                        st.info(f"🎯 **{top_mod}** — {top_d.get('prescricao','—')}")
            with col_m:
                st.markdown("**🔧 Manutenção**")
                if rows_m:
                    st.dataframe(pd.DataFrame([r for _,r in rows_m]),
                                 width="stretch", hide_index=True)
        else:
            st.info("Dados insuficientes para análise de necessidade.")
    else:
        st.info("Dados de actividade completos não disponíveis.")

    # ── Semana anterior + Comparação mensal ─────────────────────────────
    if da_full is not None and len(da_full) > 0:
        _df_all = da_full.copy()
        _df_all['Data'] = pd.to_datetime(_df_all['Data'])
        _df_all = _df_all[_df_all['type'].apply(norm_tipo) != 'WeightTraining']

        # KJ
        if 'icu_joules' in _df_all.columns:
            _df_all['_kj'] = pd.to_numeric(_df_all['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in _df_all.columns and 'moving_time' in _df_all.columns:
            _df_all['_kj'] = (pd.to_numeric(_df_all['power_avg'], errors='coerce') *
                              pd.to_numeric(_df_all['moving_time'], errors='coerce') / 1000)
        else:
            _df_all['_kj'] = np.nan
        _df_all['_km']  = pd.to_numeric(_df_all['distance'], errors='coerce') / 1000                           if 'distance' in _df_all.columns else np.nan
        _df_all['_mt']  = pd.to_numeric(_df_all['moving_time'], errors='coerce') / 3600
        _df_all['_rpe'] = pd.to_numeric(_df_all['rpe'], errors='coerce')                           if 'rpe' in _df_all.columns else np.nan

        hoje      = pd.Timestamp.now().normalize()
        dow       = hoje.weekday()
        sem_fim   = hoje - pd.Timedelta(days=dow + 1)
        sem_ini   = sem_fim - pd.Timedelta(days=6)
        mes_c_ini = hoje.replace(day=1)
        mes_p_fim = mes_c_ini - pd.Timedelta(days=1)
        mes_p_ini = mes_p_fim.replace(day=1)

        def _vg_agg(df, d_ini, d_fim, semana=False):
            sub = df[(df['Data'] >= d_ini) & (df['Data'] <= d_fim)]
            if len(sub) == 0: return {}
            res = {}
            for mod in sorted(sub['type'].apply(norm_tipo).unique()):
                s = sub[sub['type'].apply(norm_tipo) == mod]
                d = {'kj': s['_kj'].sum() if s['_kj'].notna().any() else 0,
                     'km': s['_km'].sum() if '_km' in s and s['_km'].notna().any() else 0,
                     'horas': s['_mt'].sum() if s['_mt'].notna().any() else 0}
                if semana:
                    d['sessoes']   = len(s)
                    d['rpe_altas'] = int((s['_rpe'] >= 7).sum())                                      if '_rpe' in s.columns else 0
                res[mod] = d
            return res

        sem_data = _vg_agg(_df_all, sem_ini, sem_fim, semana=True)
        mes_p    = _vg_agg(_df_all, mes_p_ini, mes_p_fim)
        mes_c    = _vg_agg(_df_all, mes_c_ini, hoje)

        # ── Tabela semana anterior ─────────────────────────────────────────
        st.subheader(f"📅 Semana anterior "
                     f"({sem_ini.strftime('%d/%m')} → {sem_fim.strftime('%d/%m')})")
        if sem_data:
            rows_sem = []
            for mod, d in sem_data.items():
                rows_sem.append({
                    'Modalidade':     mod,
                    'Sessões':        d['sessoes'],
                    'Horas':          fmt_dur(d['horas']),
                    'KM':             f"{d['km']:.0f}" if d['km'] > 0 else '—',
                    'KJ':             f"{d['kj']:.0f}" if d['kj'] > 0 else '—',
                    'Sessões RPE≥7':  d['rpe_altas'],
                })
            st.dataframe(pd.DataFrame(rows_sem), width="stretch", hide_index=True)
        else:
            st.info(f"Sem actividades entre {sem_ini.strftime('%d/%m')} e {sem_fim.strftime('%d/%m')}.")

        st.markdown("---")

        # ── Gráfico barras horizontais KJ ────────────────────────────────
        all_mods_m = sorted(set(list(mes_p.keys()) + list(mes_c.keys())))
        if all_mods_m:
            st.subheader(f"⚡ KJ por Modalidade — "
                         f"{mes_p_ini.strftime('%b %Y')} vs {mes_c_ini.strftime('%b %Y')}")
            kj_p = [mes_p.get(m,{}).get('kj',0) for m in all_mods_m]
            kj_c = [mes_c.get(m,{}).get('kj',0) for m in all_mods_m]
            fig_kj = go.Figure()
            fig_kj.add_trace(go.Bar(
                y=all_mods_m, x=kj_p, name=mes_p_ini.strftime('%b %Y'),
                orientation='h', marker_color='#95a5a6',
                text=[f"{v:.0f}" if v>0 else '' for v in kj_p],
                textposition='outside',
                hovertemplate='%{y}: <b>%{x:.0f} kJ</b><extra></extra>'))
            fig_kj.add_trace(go.Bar(
                y=all_mods_m, x=kj_c,
                name=f"{mes_c_ini.strftime('%b %Y')} (corrente)",
                orientation='h', marker_color='#2ecc71',
                text=[f"{v:.0f}" if v>0 else '' for v in kj_c],
                textposition='outside',
                hovertemplate='%{y}: <b>%{x:.0f} kJ</b><extra></extra>'))
            fig_kj.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                barmode='group', height=max(200, len(all_mods_m)*60+80),
                font=dict(color='#222222'),
                margin=dict(l=60, r=80, t=30, b=20),
                xaxis=dict(title='kJ', showgrid=True, gridcolor='#eeeeee',
                           tickfont=dict(color='#333333')),
                yaxis=dict(tickfont=dict(color='#333333')),
                legend=dict(orientation='h', y=1.08,
                            font=dict(color='#111111', size=11)))
            st.plotly_chart(fig_kj, use_container_width=True)

        # ── Tabela comparativa mensal ─────────────────────────────────────
        if all_mods_m:
            st.subheader("📊 Comparação Mensal por Modalidade")
            st.caption(f"Mês passado: {mes_p_ini.strftime('%b %Y')} | "
                       f"Corrente: {mes_c_ini.strftime('%b %Y')} (até hoje)")

            def _fmt_diff(vc, vp, unid='', fmt='.0f'):
                if vp == 0: return '(sem dados mês ant.)'
                diff = vc - vp; pct = diff / vp * 100
                s = '↗' if diff > 0 else ('↘' if diff < 0 else '→')
                return f"{s} {diff:+{fmt}}{unid} ({pct:+.0f}%)"

            rows_mes = []
            for mod in all_mods_m:
                dp = mes_p.get(mod, {})
                rows_mes.append({
                    'Mês':        mes_p_ini.strftime('%b %Y'),
                    'Modalidade': mod,
                    'KJ':         f"{dp.get('kj',0):.0f}"    if dp.get('kj',0)>0    else '—',
                    'KM':         f"{dp.get('km',0):.0f}"    if dp.get('km',0)>0    else '—',
                    'Horas':      fmt_dur(dp.get('horas',0)) if dp.get('horas',0)>0 else '—',
                    'Δ vs mês ant.': '—',
                })
            for mod in all_mods_m:
                dc_ = mes_c.get(mod, {}); dp_ = mes_p.get(mod, {})
                rows_mes.append({
                    'Mês':        f"{mes_c_ini.strftime('%b %Y')} ▶",
                    'Modalidade': mod,
                    'KJ':         f"{dc_.get('kj',0):.0f}"    if dc_.get('kj',0)>0    else '—',
                    'KM':         f"{dc_.get('km',0):.0f}"    if dc_.get('km',0)>0    else '—',
                    'Horas':      fmt_dur(dc_.get('horas',0)) if dc_.get('horas',0)>0 else '—',
                    'Δ vs mês ant.': (
                        f"KJ {_fmt_diff(dc_.get('kj',0),dp_.get('kj',0),'kJ')}  "
                        f"KM {_fmt_diff(dc_.get('km',0),dp_.get('km',0),'km')}  "
                        f"H {_fmt_diff(dc_.get('horas',0),dp_.get('horas',0),'h','.1f')}"
                    ) if dp_ else '— (sem dados mês ant.)',
                })
            st.dataframe(pd.DataFrame(rows_mes), width="stretch", hide_index=True)


    # ── Camada de Progressão de Carga ────────────────────────────────────
    if da_full is not None and len(da_full) > 0:
        st.subheader("📈 Progressão de Carga Semanal")
        st.caption(
            "Camada independente do Need Score — controla QUANTO treinar. "
            "Fator modulado pelo Need (leitura apenas). Cap +12% vs ano anterior.")

        _pf = da_full.copy()
        _pf['Data'] = pd.to_datetime(_pf['Data'])
        _pf = _pf[_pf['type'].apply(norm_tipo) != 'WeightTraining']

        # KJ
        if 'icu_joules' in _pf.columns:
            _pf['_kj'] = pd.to_numeric(_pf['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in _pf.columns and 'moving_time' in _pf.columns:
            _pf['_kj'] = (pd.to_numeric(_pf['power_avg'], errors='coerce') *
                          pd.to_numeric(_pf['moving_time'], errors='coerce') / 1000)
        else:
            _pf['_kj'] = np.nan
        _pf['_km']  = pd.to_numeric(_pf['distance'], errors='coerce') / 1000                       if 'distance' in _pf.columns else np.nan
        _pf['_mt']  = pd.to_numeric(_pf['moving_time'], errors='coerce') / 3600
        _pf['_rpe_n']   = pd.to_numeric(_pf['rpe'], errors='coerce') if 'rpe' in _pf.columns else np.nan
        _pf['_dur_min'] = _pf['_mt'] * 60

        hoje_pf   = pd.Timestamp.now().normalize()
        ano_atual = hoje_pf.year
        ano_ant   = ano_atual - 1
        dia_ano   = hoje_pf.timetuple().tm_yday
        # Semana actual: segunda até hoje
        dow_pf    = hoje_pf.weekday()
        sem_ini_pf= hoje_pf - pd.Timedelta(days=dow_pf)

        # Buscar Need scores e overload do modelo (se disponível)
        _need_cache = {}
        _ol_cache   = {}
        _ni_cache   = {}
        try:
            _res_prog, _ = analisar_falta_estimulo(_pf, janela_dias=14)
            if _res_prog:
                for _m, _d in _res_prog.items():
                    _need_cache[_m] = _d.get('need_score', 40)
                    _ol_cache[_m]   = _d.get('overload', False)
                    _ni_cache[_m]   = _d.get('need_int_prescr', 50)
        except Exception:
            pass

        # eFTP por modalidade (último valor disponível)
        _eftp = {}
        # icu_ftp = FTP testado (estável, para zonas) — preferido
        # icu_eftp = estimado pelo modelo (para acompanhamento de forma)
        _ftp_col = 'icu_ftp' if 'icu_ftp' in _pf.columns else 'icu_eftp'
        if _ftp_col in _pf.columns:
            for _m in ['Bike','Row','Ski','Run']:
                _s = _pf[_pf['type'].apply(norm_tipo)==_m][_ftp_col]
                _s = pd.to_numeric(_s, errors='coerce').dropna()
                if len(_s) > 0: _eftp[_m] = float(_s.iloc[-1])

        def _med_semanas(df, col, n_sem=8):
            """Mediana das últimas n semanas com dados (col > 0)."""
            df = df.copy()
            df['_sem'] = df['Data'].dt.to_period('W')
            agg = df.groupby('_sem')[col].sum().reset_index()
            agg = agg.sort_values('_sem').tail(n_sem)
            vals = agg[col][agg[col] > 0]
            return float(vals.median()) if len(vals) > 0 else 0.0

        def _sugestao_sessao(kj_rest, h_rest, km_rest, mod, eftp, ni, ol, df_hist=None):
            """
            Retorna (df_opcoes, ref_line, ol_warn).
            5 tipos de estimulo com mesmo KJ_target. Principal por Need_intensity.
            """
            _TIPOS = [
                {"key":"anaerobio",  "label":"⚫ Anaeróbio",  "pct":(1.15,1.30), "rpe_lbl":"9–10"},
                {"key":"vo2",        "label":"🔴 VO2max",     "pct":(0.95,1.10), "rpe_lbl":"8–9"},
                {"key":"threshold",  "label":"🟡 Threshold",  "pct":(0.83,0.90), "rpe_lbl":"6–7"},
                {"key":"sweetspot",  "label":"🟢 Sweet Spot", "pct":(0.78,0.83), "rpe_lbl":"5–6"},
                {"key":"leve",       "label":"🔵 Leve",       "pct":(0.55,0.68), "rpe_lbl":"3–4"},
            ]

            def _estrutura(tipo_key, dur_work_min):
                d = max(1.0, float(dur_work_min))
                structs = []
                if tipo_key == "anaerobio":
                    for reps, on_s, rest_s in [(10,20,120),(8,25,150),(6,30,180),(5,30,180),(4,40,240)]:
                        if d > 0 and abs(reps*on_s/60 - d)/d < 0.30:
                            structs.append(f"{reps}x{on_s}s rest {rest_s//60}:{rest_s%60:02d}")
                    if not structs:
                        structs = [f"{max(4,round(d*60/25))}x25s rest 2:30"]
                elif tipo_key == "vo2":
                    for reps, on_s in [(4,240),(5,180),(6,180),(5,240),(4,300),(6,240),(3,300),(8,150),(10,120)]:
                        if d > 0 and abs(reps*on_s/60 - d)/d < 0.20:
                            ms = on_s//60; ss_r = on_s%60
                            structs.append(f"{reps}x{ms}:{ss_r:02d} rest {ms}:{ss_r:02d}")
                    reps_30 = round(d*2)
                    if 6 <= reps_30 <= 20:
                        n_ser = max(1, reps_30//8); rpp = reps_30//n_ser
                        structs.append(f"{n_ser}x{rpp}x30/30")
                    structs = structs[:3]
                    if not structs:
                        structs = [f"{max(3,round(d/4))}x4:00 rest 4:00"]
                elif tipo_key == "threshold":
                    for reps, on_m in [(3,10),(4,8),(2,15),(4,10),(3,12),(5,8),(2,20)]:
                        if d > 0 and abs(reps*on_m - d)/d < 0.25:
                            structs.append(f"{reps}x{on_m}min rest 3–4min")
                    structs = structs[:2]
                    if not structs:
                        n = max(2, round(d/10))
                        structs = [f"{n}x{round(d/n)}min rest 3min"]
                elif tipo_key == "sweetspot":
                    for reps, on_m in [(1,40),(2,20),(3,15),(1,30),(2,25),(1,50)]:
                        if d > 0 and abs(reps*on_m - d)/d < 0.25:
                            structs.append(f"1x{on_m}min contínuo" if reps==1 else f"{reps}x{on_m}min rest 4–5min")
                    structs = structs[:2]
                    if not structs:
                        structs = [f"1x{round(d)}min contínuo"]
                else:
                    structs = [f"Contínuo {round(d)}min"]
                return "  /  ".join(structs)

            # ── Histórico ─────────────────────────────────────────────────
            _ref_kj = _ref_dur = _ref_pwr = None
            _eff_delta = 0.0
            _kjh_ref = _kjh_baseline = None
            kj_target_A_adj = 1.0
            _PROG_CAP = {"Z3":1.07, "Z2":1.10, "Z1":1.12}
            _TEMPO_CAP = 1.10
            pwr_inc_base = 0.01 if ni >= 70 else 0.02

            if df_hist is not None and len(df_hist) >= 2:
                _df_mod = df_hist[df_hist["type"].apply(norm_tipo) == mod].copy()
                if len(_df_mod) > 0:
                    _has_zona = "z3_kj" in _df_mod.columns and "z2_kj" in _df_mod.columns
                    if _has_zona:
                        _z3n = pd.to_numeric(_df_mod["z3_kj"], errors="coerce").fillna(0)
                        _z2n = pd.to_numeric(_df_mod["z2_kj"], errors="coerce").fillna(0)
                        _tot = pd.to_numeric(_df_mod["_kj"], errors="coerce").replace(0,np.nan).fillna(1)
                        _df_mod["_zona_dom"] = "Z1"
                        _df_mod.loc[_z2n/_tot > 0.30, "_zona_dom"] = "Z2"
                        _df_mod.loc[_z3n/_tot > 0.20, "_zona_dom"] = "Z3"
                        _df_mod.loc[(_z3n==0)&(_z2n==0), "_zona_dom"] = None
                    else:
                        _df_mod["_zona_dom"] = None
                    _df_mod["_rpe_n_num"] = pd.to_numeric(_df_mod["_rpe_n"], errors="coerce")
                    _df_mod["_match_rpe"]  = _df_mod["_rpe_n_num"].between(5,10).astype(float)
                    _df_mod["_match_zona"] = (_df_mod["_zona_dom"]=="Z3").astype(float)*0.5
                    _nt = len(_df_mod)
                    _df_mod["_recency"] = (np.arange(_nt)/max(_nt-1,1))*0.2
                    _df_mod["_score"]   = _df_mod["_match_rpe"]+_df_mod["_match_zona"]+_df_mod["_recency"]
                    _pool = _df_mod[_df_mod["_match_rpe"] >= 1.0].copy()
                    _dh   = _pool.sort_values("_score", ascending=False).head(5)
                    if len(_dh) >= 2:
                        _ref_kj  = float(_dh["_kj"].median())
                        _ref_dur = float(_dh["_dur_min"].median())
                        if "z3_pwr" in _dh.columns:
                            _zp = pd.to_numeric(_dh["z3_pwr"], errors="coerce").replace(0,np.nan)
                            _ref_pwr = float(_zp.median()) if _zp.notna().any() else None
                        _pool2 = _pool.copy()
                        _kj_w  = pd.to_numeric(_pool2["_kj"], errors="coerce").replace(0,np.nan)
                        _trimp = (pd.to_numeric(_pool2["_dur_min"], errors="coerce") *
                                  pd.to_numeric(_pool2["_rpe_n"],   errors="coerce"))
                        _pool2["_eff"] = _trimp / _kj_w
                        _eff_bl  = float(_pool2["_eff"].median()) if _pool2["_eff"].notna().any() else None
                        _cut8w   = pd.Timestamp.now() - pd.Timedelta(weeks=8)
                        _prec    = _pool2[_pool2["Data"] >= _cut8w]
                        _eff_rec = float(_prec["_eff"].median()) if (len(_prec)>=2 and _prec["_eff"].notna().any()) else None
                        if _eff_bl and _eff_rec and _eff_bl > 0:
                            _eff_delta = (_eff_rec/_eff_bl) - 1.0
                        _pool2["_kjh"] = (pd.to_numeric(_pool2["_kj"], errors="coerce") /
                                          (pd.to_numeric(_pool2["_dur_min"], errors="coerce")/60))
                        _kjh_baseline = float(_pool2["_kjh"].median()) if _pool2["_kjh"].notna().any() else None
                        _kjh_ref = (_ref_kj/(_ref_dur/60)) if (_ref_kj and _ref_dur and _ref_dur>0) else None

            # ── pwr_inc ────────────────────────────────────────────────────
            pwr_inc = pwr_inc_base
            if   _eff_delta < -0.05: pwr_inc *= 1.2
            elif _eff_delta <  0.05: pwr_inc *= 1.0
            elif _eff_delta <  0.12: pwr_inc *= 0.9
            else:                    pwr_inc *= 0.7
            if _kjh_baseline and _kjh_ref:
                _kjh_ratio = _kjh_ref/_kjh_baseline
                if   _kjh_ratio >= 1.0:  pwr_inc *= 1.1
                elif _kjh_ratio >= 0.90: kj_target_A_adj = 0.95
                else:                    kj_target_A_adj = 0.90

            # ── zona e KJ_target ───────────────────────────────────────────
            _prog_cap = _PROG_CAP["Z3" if ni>=70 else "Z2" if ni>=40 else "Z1"]
            if ol:
                kj_target = (_ref_kj*0.65) if _ref_kj else max(kj_rest*0.65, 60)
            elif _ref_kj and _ref_kj > 0:
                kj_target = min(kj_rest if kj_rest>0 else _ref_kj*1.03,
                                _ref_kj*_prog_cap) * kj_target_A_adj
            else:
                kj_target = max(kj_rest, 80)

            tempo_max = (_ref_dur*_TEMPO_CAP) if _ref_dur else 90
            watts_ftp = eftp if eftp else 200

            # ── Semana anterior — Z1/Z2/Z3 por zona ───────────────────────
            _sa_z3_kj = _sa_z3_dur = _sa_z3_pwr = None
            _sa_z2_kj = _sa_z2_dur = _sa_z2_pwr = None
            _sa_z1_kj = _sa_z1_dur = _sa_z1_rpe_med = None
            hoje_sa    = pd.Timestamp.now().normalize()
            sem_ini_sa = hoje_sa - pd.Timedelta(days=hoje_sa.weekday())
            sa_ini     = sem_ini_sa - pd.Timedelta(weeks=1)
            sa_fim     = sem_ini_sa - pd.Timedelta(days=1)

            if df_hist is not None and len(df_hist) > 0 and "_rpe_n" in df_hist.columns:
                _df_sa_all = df_hist[
                    (df_hist["type"].apply(norm_tipo) == mod) &
                    (df_hist["Data"] >= sa_ini) &
                    (df_hist["Data"] <= sa_fim)
                ].copy()
                _rpe_sa = pd.to_numeric(_df_sa_all.get("_rpe_n", pd.Series(dtype=float)), errors="coerce")
                _kj_sa  = pd.to_numeric(_df_sa_all.get("_kj",    pd.Series(dtype=float)), errors="coerce").replace(0,np.nan)

                # Z3 — RPE >= 7 AND z3_kj/total_kj > 20%
                if "z3_kj" in _df_sa_all.columns:
                    _z3_sa  = pd.to_numeric(_df_sa_all["z3_kj"], errors="coerce").replace(0,np.nan)
                    _z3_dom = (_z3_sa / _kj_sa.fillna(1) > 0.20) & (_rpe_sa >= 7)
                    _df_z3  = _df_sa_all[_z3_dom].copy()
                    if len(_df_z3) > 0:
                        _sa_z3_kj  = float(pd.to_numeric(_df_z3["z3_kj"],  errors="coerce").replace(0,np.nan).sum())   if "z3_kj"  in _df_z3.columns else None
                        _sa_z3_dur = float(pd.to_numeric(_df_z3["z3_sec"], errors="coerce").replace(0,np.nan).sum()/60) if "z3_sec" in _df_z3.columns else None
                        _sa_z3_pwr = float(pd.to_numeric(_df_z3["z3_pwr"], errors="coerce").replace(0,np.nan).mean())  if "z3_pwr" in _df_z3.columns else None

                # Z2 — RPE 5-7 AND z2_kj/total_kj > 30%
                if "z2_kj" in _df_sa_all.columns:
                    _z2_sa  = pd.to_numeric(_df_sa_all["z2_kj"], errors="coerce").replace(0,np.nan)
                    _z2_dom = (_z2_sa / _kj_sa.fillna(1) > 0.30) & (_rpe_sa.between(5,7))
                    _df_z2  = _df_sa_all[_z2_dom].copy()
                    if len(_df_z2) > 0:
                        _sa_z2_kj  = float(pd.to_numeric(_df_z2["z2_kj"],  errors="coerce").replace(0,np.nan).sum())   if "z2_kj"  in _df_z2.columns else None
                        _sa_z2_dur = float(pd.to_numeric(_df_z2["z2_sec"], errors="coerce").replace(0,np.nan).sum()/60) if "z2_sec" in _df_z2.columns else None
                        _sa_z2_pwr = float(pd.to_numeric(_df_z2["z2_pwr"], errors="coerce").replace(0,np.nan).mean())  if "z2_pwr" in _df_z2.columns else None

                # Z1 — RPE <= 4 (qualquer sessão leve)
                _df_z1 = _df_sa_all[_rpe_sa <= 4].copy()
                if len(_df_z1) > 0:
                    _sa_z1_kj      = float(_kj_sa[_rpe_sa<=4].sum())         if _kj_sa[_rpe_sa<=4].notna().any() else None
                    _sa_z1_dur     = float(pd.to_numeric(_df_z1["_dur_min"], errors="coerce").sum()) if "_dur_min" in _df_z1.columns else None
                    _sa_z1_rpe_med = float(_rpe_sa[_rpe_sa<=4].median())

            # ── Principal ──────────────────────────────────────────────────
            if ol:
                _pk = "leve"
            elif ni >= 90:
                _pk = "anaerobio"
            elif ni >= 70:
                _pk = "vo2"
            elif ni >= 60:
                _pk = "threshold"
            elif ni >= 40:
                _pk = "sweetspot"
            else:
                _pk = "leve"

            # ── Gerar linhas ───────────────────────────────────────────────
            _rows = []
            for _t in _TIPOS:
                _key  = _t["key"]
                _pct  = (_t["pct"][0]+_t["pct"][1])/2
                _pwr_z = watts_ftp * _pct
                if _key in ("vo2","anaerobio") and _ref_pwr:
                    _pwr_z = _ref_pwr
                _inc = pwr_inc if _key==_pk else (0.0 if _key=="anaerobio" else pwr_inc*0.5)
                _pwr_f = _pwr_z*(1+_inc)*(0.95 if ol else 1.0)
                _kj_z  = (kj_target*0.35 if _key=="anaerobio" else
                           kj_target*1.10 if _key=="leve" else kj_target)
                _dw = min((_kj_z*1000/(_pwr_f*60)) if _pwr_f>0 else 40, tempo_max)
                _rr = {"anaerobio":8.0,"vo2":1.0,"threshold":0.35,"sweetspot":0.20}.get(_key,0.0)
                _dt = _dw*(1+_rr)
                _kj_r = _pwr_f*_dw*60/1000
                _kjh  = _kj_r/(_dt/60) if _dt>0 else 0
                _struct = _estrutura(_key, _dw)
                # ── vs semana anterior por zona ──────────────────────────
                _vs = ""
                if _key in ("vo2","anaerobio"):
                    # Z3: comparar kJ, duração e power em Z3
                    if _sa_z3_kj and _sa_z3_kj > 5:
                        _dk = _kj_r - _sa_z3_kj
                        _vs = f"{_dk:+.0f}kJ Z3"
                        if _sa_z3_pwr and _sa_z3_pwr > 0:
                            _dp = _pwr_f - _sa_z3_pwr
                            _vs += f" | {_dp:+.0f}W"
                    elif _sa_z3_dur and _sa_z3_dur > 2:
                        _vs = f"{_dw-_sa_z3_dur:+.1f}min Z3"

                elif _key in ("threshold","sweetspot"):
                    # Z2: comparar kJ e duração em Z2
                    if _sa_z2_kj and _sa_z2_kj > 10:
                        _dk2 = _kj_r - _sa_z2_kj
                        _vs = f"{_dk2:+.0f}kJ Z2"
                        if _sa_z2_pwr and _sa_z2_pwr > 0:
                            _dp2 = _pwr_f - _sa_z2_pwr
                            _vs += f" | {_dp2:+.0f}W"
                    elif _sa_z2_dur and _sa_z2_dur > 5:
                        _vs = f"{_dw-_sa_z2_dur:+.1f}min Z2"
                    elif not _sa_z2_kj and not _sa_z2_dur:
                        # Sem Z2 na semana anterior — novo estímulo
                        _vs = "novo estímulo Z2"

                elif _key == "leve":
                    # Z1: opção B — se RPE médio < 4 e dur > 45min, sugerir subir para Z2
                    if _sa_z1_kj and _sa_z1_dur:
                        _dk1 = _kj_r - _sa_z1_kj
                        _vs = f"{_dk1:+.0f}kJ Z1"
                        # Sugerir upgrade se leve bem tolerado
                        if (_sa_z1_rpe_med is not None and _sa_z1_rpe_med < 3.5
                                and _sa_z1_dur > 45):
                            _vs += " → considera Sweet Spot"
                    elif _sa_z1_dur and _sa_z1_dur > 0:
                        _dd1 = _dw - _sa_z1_dur
                        _vs = f"{_dd1:+.0f}min Z1"
                        if (_sa_z1_rpe_med is not None and _sa_z1_rpe_med < 3.5
                                and _sa_z1_dur > 45):
                            _vs += " → considera Sweet Spot"
                _kjh_str = f"{_kjh:.0f}"
                if _kjh_ref and _kjh_ref>0:
                    _kjh_str += f" ({(_kjh-_kjh_ref)/_kjh_ref*100:+.0f}%)"
                _rows.append({
                    "Tipo":       ("★ " if _key==_pk else "  ")+_t["label"],
                    "Estrutura":  _struct,
                    "Watts":      f"{round(_pwr_f)}W",
                    "Work":       f"{_dw:.0f}min",
                    "Total":      f"{_dt:.0f}min",
                    "KJ":         f"{_kj_r:.0f}",
                    "KJ/h":       _kjh_str,
                    "RPE":        _t["rpe_lbl"],
                    "vs sem.ant": _vs,
                })

            if not _rows:
                return None, None, None

            _df_out = pd.DataFrame(_rows)
            _ref_line = ""
            if _ref_kj and _ref_dur:
                _kjh_r = _ref_kj/(_ref_dur/60)
                _ref_line = f"Ref: {_ref_kj:.0f} kJ | {_ref_dur:.0f} min | {_kjh_r:.0f} kJ/h"
                if _sa_z3_kj:
                    _ref_line += f"  |  Sem.ant Z3: {_sa_z3_kj:.0f} kJ"
                    if _sa_z3_dur: _ref_line += f" / {_sa_z3_dur:.0f} min"
            _ol_warn = "⚠️ EM OVERLOAD — power reduzido 5%" if ol else ""
            return _df_out, _ref_line, _ol_warn

        rows_prog = []
        for mod in ['Bike','Row','Ski','Run']:
            _sub = _pf[_pf['type'].apply(norm_tipo)==mod].copy()
            if len(_sub) == 0: continue

            need  = _need_cache.get(mod, 40)
            ol    = _ol_cache.get(mod, False)
            ni    = _ni_cache.get(mod, 50)
            eftp  = _eftp.get(mod)

            # Baseline mediana 8 semanas (só semanas com dados)
            kj_base  = _med_semanas(_sub, '_kj')
            h_base   = _med_semanas(_sub, '_mt')
            km_base  = _med_semanas(_sub, '_km') if '_km' in _sub.columns else 0.0
            has_kj   = kj_base > 0
            has_km   = km_base > 0

            # Fator progressão
            if ol:                fator = 0.98
            elif need > 60:       fator = 1.04
            elif need < 30:       fator = 1.01
            else:                 fator = 1.02

            # Sem anterior (última semana completa)
            sem_ant_ini = sem_ini_pf - pd.Timedelta(weeks=1)
            sem_ant_fim = sem_ini_pf - pd.Timedelta(days=1)
            _sem_ant = _sub[(_sub['Data']>=sem_ant_ini)&(_sub['Data']<=sem_ant_fim)]
            kj_sem_ant = float(_sem_ant['_kj'].sum()) if has_kj else 0.0
            h_sem_ant  = float(_sem_ant['_mt'].sum())

            # Meta semana (com suavização anti-salto)
            kj_meta = min(kj_base * fator, max(kj_sem_ant, kj_base) * 1.08) if has_kj else 0.0
            h_meta  = min(h_base  * fator, max(h_sem_ant,  h_base)  * 1.08)
            km_meta = km_base * fator if has_km else 0.0

            # Informativo: comparação com ano anterior (sem cap — só leitura)
            _ano_ant = _sub[_sub['Data'].dt.year == ano_ant]
            _ano_cur = _sub[_sub['Data'].dt.year == ano_atual]
            h_2025   = float(_ano_ant['_mt'].sum())
            kj_2025  = float(_ano_ant['_kj'].sum()) if has_kj else 0.0
            km_2025  = float(_ano_ant['_km'].sum()) if has_km else 0.0
            h_acum   = float(_ano_cur['_mt'].sum())
            kj_acum  = float(_ano_cur['_kj'].sum()) if has_kj else 0.0
            km_acum  = float(_ano_cur['_km'].sum()) if has_km else 0.0
            h_proj   = h_acum  / dia_ano * 365 if h_acum  > 0 else 0
            kj_proj  = kj_acum / dia_ano * 365 if has_kj and kj_acum > 0 else 0
            cap_atingido = False  # cap removido — só informativo

            # Status horas vs ano anterior (informativo)
            if h_2025 > 0 and h_proj > 0:
                h_delta_pct = (h_proj - h_2025) / h_2025 * 100
                if h_delta_pct < 0:
                    status_ano = f"⚠️ Abaixo ({h_delta_pct:+.0f}% vs {ano_ant})"
                elif h_delta_pct <= 3:
                    status_ano = f"→ Manutenção ({h_delta_pct:+.0f}% vs {ano_ant})"
                elif h_delta_pct <= 12:
                    status_ano = f"✅ No range ({h_delta_pct:+.0f}% vs {ano_ant})"
                else:
                    status_ano = f"📈 Acima ({h_delta_pct:+.0f}% vs {ano_ant})"
            elif h_2025 == 0:
                status_ano = "— (sem ano anterior)"
            else:
                status_ano = "— (sem proj.)"

            # Semana actual (Seg → hoje)
            _sem_cur = _sub[_sub['Data'] >= sem_ini_pf]
            kj_feito = float(_sem_cur['_kj'].sum()) if has_kj else 0.0
            h_feito  = float(_sem_cur['_mt'].sum())
            km_feito = float(_sem_cur['_km'].sum()) if has_km else 0.0

            kj_rest  = max(0.0, kj_meta - kj_feito) if has_kj else 0.0
            h_rest   = max(0.0, h_meta  - h_feito)
            km_rest  = max(0.0, km_meta - km_feito) if has_km else 0.0

            # Overload: nao adicionar carga — sugestao apenas de recuperacao
            if ol:
                kj_rest = 0.0; h_rest = 0.0; km_rest = 0.0

            # Sugestão — retorna (df, ref_line, ol_warn)
            _sug_df, _sug_ref, _sug_ol = _sugestao_sessao(
                kj_rest, h_rest, km_rest, mod, eftp, ni, ol, df_hist=_pf)

            # Fator label
            if ol:       fl = "↓ 0.98 overload"
            elif cap_atingido: fl = "→ 1.00 cap"
            elif need>60: fl = "↑ 1.04"
            elif need<30: fl = "↑ 1.01"
            else:         fl = "↑ 1.02"

            row = {
                'Modalidade':    mod,
                'Métrica':       'KJ' if has_kj else ('KM' if has_km else 'Horas'),
                'Base (med.8s)': f"{kj_base:.0f} kJ" if has_kj else
                                  (f"{km_base:.0f} km | {fmt_dur(h_base)}" if has_km
                                   else fmt_dur(h_base)),
                'Fator':         fl,
                'Meta semana':   f"{kj_meta:.0f} kJ" if has_kj else
                                  (f"{km_meta:.0f} km | {fmt_dur(h_meta)}" if has_km
                                   else fmt_dur(h_meta)),
                'Feito':         f"{kj_feito:.0f} kJ" if has_kj else
                                  (f"{km_feito:.0f} km | {fmt_dur(h_feito)}" if has_km
                                   else fmt_dur(h_feito)),
                'Restante':      (f"✅ 0" if kj_rest==0 and has_kj else
                                   f"{kj_rest:.0f} kJ") if has_kj else
                                  (f"✅ 0" if km_rest==0 and h_rest==0 else
                                   f"{km_rest:.0f} km | {fmt_dur(h_rest)}" if has_km
                                   else fmt_dur(h_rest)),
                'Proj. Horas 2026': fmt_dur(h_proj) if h_proj>0 else "—",
                'Range Horas (+3–12%)': (
                    f"{fmt_dur(h_2025)} → {fmt_dur(h_2025*1.03)}–{fmt_dur(h_2025*1.12)}"
                    if h_2025 > 0 else "—"),
                'Status Horas':  status_ano,
                '_sug_df': _sug_df, '_sug_ref': _sug_ref, '_sug_ol': _sug_ol,
            }
            rows_prog.append(row)

        if rows_prog:
            # Remover colunas internas antes de mostrar tabela
            _rows_prog_display = [{k:v for k,v in r.items()
                                   if k not in ("_sug_df","_sug_ref","_sug_ol")}
                                  for r in rows_prog]
            df_prog = pd.DataFrame(_rows_prog_display)

            # ── Deload / Taper detector ───────────────────────────────────
            # baseline = mediana últimas 3 semanas com dados (KJ ou Horas)
            _sem_ini_det  = hoje_pf - pd.Timedelta(weeks=1)
            _sem3_ini_det = hoje_pf - pd.Timedelta(weeks=4)
            _kj_total_cur = 0.0
            _kj_total_b3  = []
            for _m3 in ['Bike','Row','Ski','Run']:
                _s3 = _pf[_pf['type'].apply(norm_tipo)==_m3].copy()
                if len(_s3) == 0: continue
                _s3['_ksem'] = _s3['Data'].dt.to_period('W')
                _agg3 = _s3.groupby('_ksem')['_kj'].sum()
                # Últimas 3 semanas completas (antes da actual)
                _prev3 = _agg3[_agg3.index < _s3['Data'].max().to_period('W')].tail(3)
                _kj_total_b3.extend(_prev3.values)
                # Semana actual desta modalidade
                _kj_total_cur += float(_s3[_s3['Data'] >= sem_ini_pf]['_kj'].sum())

            _kj_b3_mean = float(np.mean(_kj_total_b3)) if _kj_total_b3 else 0.0
            _load_ratio  = (_kj_total_cur / _kj_b3_mean) if _kj_b3_mean > 0 else 1.0

            # Limiares
            _kj_normal_min = _kj_b3_mean * 0.80
            _kj_deload_min = _kj_b3_mean * 0.60
            _kj_taper_min  = _kj_b3_mean * 0.30
            _kj_taper_max  = _kj_b3_mean * 0.60
            _kj_deload_max = _kj_b3_mean * 0.80

            if _kj_b3_mean > 0:
                st.markdown("**📊 Estado de carga — semana actual**")
                _c_det1, _c_det2, _c_det3 = st.columns(3)
                with _c_det1:
                    st.metric("Carga actual (semana)", f"{_kj_total_cur:.0f} kJ",
                              f"{(_load_ratio-1)*100:+.0f}% vs baseline")
                with _c_det2:
                    st.metric("Baseline (média 3 sem)", f"{_kj_b3_mean:.0f} kJ")
                with _c_det3:
                    if _load_ratio >= 0.80:
                        _fase = "✅ Normal"
                    elif _load_ratio >= 0.60:
                        _fase = "📉 Deload"
                    elif _load_ratio >= 0.30:
                        _fase = "🔵 Taper"
                    else:
                        _fase = "⚠️ Muito baixo"
                    st.metric("Fase detectada", _fase, f"ratio={_load_ratio:.2f}")

                # Ranges informativos
                st.caption(
                    f"**Ranges de referência** (baseline = {_kj_b3_mean:.0f} kJ/sem) — "
                    f"Normal: >{_kj_normal_min:.0f} kJ | "
                    f"Deload: {_kj_deload_min:.0f}–{_kj_deload_max:.0f} kJ "
                    f"(60–80% = redução 20–40%) | "
                    f"Taper: {_kj_taper_min:.0f}–{_kj_taper_max:.0f} kJ "
                    f"(30–60% = redução 40–70%)")

                if _load_ratio < 0.80:
                    if _load_ratio >= 0.60:
                        st.info("📉 **Deload detectado** — volume 20–40% abaixo do baseline. "
                                "Manter intensidade. Duração típica: 3–7 dias.")
                    elif _load_ratio >= 0.30:
                        st.info("🔵 **Taper detectado** — volume 40–70% abaixo do baseline. "
                                "Manter ou aumentar intensidade. Duração típica: 7–14 dias.")
                    else:
                        st.warning("⚠️ Carga muito baixa (>70% de redução). "
                                   "Confirma se é intencional.")

            st.markdown("---")

            # ── ΔCTL estimado — usa icu_training_load (mesma escala do PMC) ──
            # CTL/ATL calculados com icu_training_load → consistente com Tab PMC
            # ΔCTL/sessão = load_sessão / 42  (definição EMA span=42)
            # Threshold: +1 a +5 CTL por semana (válido na escala TSS-like)
            _pf2 = _pf.copy()

            # Usar icu_training_load se disponível, fallback session_rpe
            _tl_col = None
            if 'icu_training_load' in _pf2.columns:
                _pf2['_load_icu'] = pd.to_numeric(_pf2['icu_training_load'], errors='coerce').fillna(0)
                _tl_col = 'icu_training_load'
            else:
                _pf2['_load_icu'] = (_pf2['_mt'] * 60 *
                    pd.to_numeric(_pf2.get('rpe', pd.Series(dtype=float)), errors='coerce').fillna(5))
                _tl_col = 'session_rpe (fallback)'

            _dates_ctl = pd.date_range(_pf2['Data'].min(), hoje_pf, freq='D')
            _load_ctl  = _pf2.groupby('Data')['_load_icu'].sum().reindex(_dates_ctl, fill_value=0)
            _ctl_hoje  = float(_load_ctl.ewm(span=42, adjust=False).mean().iloc[-1])
            _atl_hoje  = float(_load_ctl.ewm(span=7,  adjust=False).mean().iloc[-1])
            _tsb_hoje  = _ctl_hoje - _atl_hoje

            # ΔCTL por modalidade:
            # Estimar load da próxima sessão = mediana icu_training_load das sessões comparáveis
            # ΔCTL_sessão = load_estimada / 42
            _delta_ctl_total = 0.0
            _delta_rows = []
            for r_p in rows_prog:
                _mod_p = r_p['Modalidade']
                _sug_df_p = r_p.get('_sug_df')

                # Load estimada: mediana das sessões comparáveis desta modalidade
                _load_est = 0.0
                _load_src = "—"
                if 'icu_training_load' in _pf2.columns:
                    _df_mod_p = _pf2[_pf2['type'].apply(norm_tipo) == _mod_p]
                    _ni_p     = _ni_cache.get(_mod_p, 50)
                    # Filtrar por RPE compatível com Need_intensity
                    _rpe_min  = 7 if _ni_p >= 75 else 5 if _ni_p >= 40 else 1
                    _rpe_max  = 10 if _ni_p >= 60 else 7 if _ni_p >= 30 else 5
                    _rpe_f    = pd.to_numeric(_df_mod_p.get('rpe', pd.Series(dtype=float)), errors='coerce')
                    _df_comp  = _df_mod_p[_rpe_f.between(_rpe_min, _rpe_max)]
                    if len(_df_comp) >= 3:
                        _tl_vals  = pd.to_numeric(_df_comp['icu_training_load'], errors='coerce').dropna()
                        if len(_tl_vals) >= 2:
                            _load_est = float(_tl_vals.tail(10).median())
                            _load_src = f"mediana {min(len(_tl_vals),10)} sessões RPE {_rpe_min}–{_rpe_max}"
                    if _load_est == 0:
                        # Fallback: mediana geral desta modalidade
                        _tl_all = pd.to_numeric(_df_mod_p['icu_training_load'], errors='coerce').dropna()
                        if len(_tl_all) >= 2:
                            _load_est = float(_tl_all.tail(10).median())
                            _load_src = "mediana geral (fallback)"

                # ΔCTL = load / 42
                _delta_p = _load_est / 42.0 if _load_est > 0 else 0.0
                _delta_ctl_total += _delta_p
                _delta_rows.append({
                    'Modalidade':    _mod_p,
                    'Load estimada': f"{_load_est:.0f}" if _load_est > 0 else "—",
                    'Fonte':         _load_src,
                    'ΔCTL est.':     f"{_delta_p:+.2f}",
                })

            # CTL/TSB projectados
            _ctl_proj = _ctl_hoje + _delta_ctl_total
            _tsb_proj = _tsb_hoje - _delta_ctl_total

            # Threshold +1 a +5 CTL/semana (escala icu_training_load)
            _dentro_range = 1.0 <= _delta_ctl_total <= 5.0

            st.markdown("**⚡ Impacto CTL estimado — semana actual**")
            st.caption(f"Métrica: **{_tl_col}** — consistente com Tab PMC")
            _cc1, _cc2, _cc3, _cc4 = st.columns(4)
            with _cc1: st.metric("CTL actual",     f"{_ctl_hoje:.1f}")
            with _cc2: st.metric("ΔCTL estimado",  f"{_delta_ctl_total:+.2f}",
                                 "✅ no range 1–5" if _dentro_range else
                                 ("⚠️ abaixo de 1" if _delta_ctl_total < 1 else "⚠️ acima de 5"))
            with _cc3: st.metric("CTL projectado", f"{_ctl_proj:.1f}")
            with _cc4: st.metric("TSB projectado", f"{_tsb_proj:.1f}")

            if _delta_rows:
                with st.expander("🔍 Detalhe ΔCTL por modalidade"):
                    st.dataframe(pd.DataFrame(_delta_rows), width="stretch", hide_index=True)
                    st.caption(
                        "Load estimada = mediana icu_training_load das sessões comparáveis (RPE range). "
                        "ΔCTL = load / 42. Threshold: +1 a +5 CTL/semana.")

            st.markdown("---")
            st.dataframe(df_prog,
                         width="stretch", hide_index=True)
            st.markdown("**💡 Sugestões de sessão (semana actual)**")

            # Mini-tabs por modalidade
            _mods_sug = [r for r in rows_prog if r.get("_sug_df") is not None]
            if _mods_sug:
                _emj_map = {"Bike":"🚴 Bike","Row":"🚣 Row","Ski":"🎿 Ski","Run":"🏃 Run"}
                _tab_labels = [_emj_map.get(r["Modalidade"], r["Modalidade"])
                               for r in _mods_sug]
                _sug_tabs = st.tabs(_tab_labels)
                for _stab, r in zip(_sug_tabs, _mods_sug):
                    with _stab:
                        _df_s  = r["_sug_df"]
                        _ref_s = r.get("_sug_ref","")
                        _ol_s  = r.get("_sug_ol","")
                        # Linha de referência + overload
                        if _ref_s: st.caption(_ref_s)
                        if _ol_s:  st.warning(_ol_s)
                        # Tabela de opções — principal marcada com ★ na coluna Tipo
                        # Não usar style (causa texto branco no tema escuro do Streamlit)
                        st.dataframe(
                            _df_s,
                            hide_index=True,
                            use_container_width=True)
                        # KJ restante e meta abaixo da tabela
                        _kj_r_val = r.get("Restante","")
                        _meta_val = r.get("Meta semana","")
                        _feito_val = r.get("Feito","")
                        if _kj_r_val or _meta_val:
                            st.caption(
                                f"Meta semana: **{_meta_val}** | "
                                f"Feito: **{_feito_val}** | "
                                f"Restante: **{_kj_r_val}**")
            st.caption(
                "⚠️ Quantidade de carga: esta camada. Tipo de treino: Need Score acima. "
                "Cap horas +12% vs " + str(ano_ant) + ".")

    st.markdown("---")

    st.markdown("---")

    st.markdown("---")

    # ── Tabela % KM por modalidade ──
    st.subheader("📏 Distribuição de KM por modalidade")
    if 'distance' in da.columns and da['distance'].notna().any():
        _agr_col, _ = st.columns([1, 3])
        _agr = _agr_col.selectbox("Agrupar por", ["Semana", "Mês", "Ano"],
                                   key="vg_agrup_km")
        _code = {"Semana": "W", "Mês": "M", "Ano": "Y"}[_agr]

        df_km = da.copy()
        df_km['_t'] = df_km['type'].apply(norm_tipo)
        df_km = df_km[df_km['_t'] != 'WeightTraining']
        df_km['km'] = pd.to_numeric(df_km['distance'], errors='coerce') / 1000
        df_km = df_km[df_km['km'].notna() & (df_km['km'] > 0)]
        df_km['Data'] = pd.to_datetime(df_km['Data'])
        df_km['_p'] = df_km['Data'].dt.to_period(_code)

        tipos_km = [t for t in ['Bike', 'Row', 'Ski', 'Run']
                    if t in df_km['_t'].unique()]

        if len(df_km) > 0 and tipos_km:
            piv = (df_km.groupby(['_p', '_t'])['km'].sum()
                   .unstack(fill_value=0)
                   .reindex(columns=tipos_km, fill_value=0))
            piv['Total'] = piv[tipos_km].sum(axis=1)

            rows_km = []
            for p, r in piv.sort_index(ascending=False).iterrows():
                if _agr == "Ano":    lbl = str(p.year)
                elif _agr == "Semana": lbl = p.start_time.strftime('%d/%m/%y')
                else: lbl = pd.to_datetime(str(p)).strftime('%B %Y').title()
                row = {'Período': lbl}
                tot = r['Total']
                for t in tipos_km:
                    v = r[t]
                    pct = (v / tot * 100) if tot > 0 else 0
                    row[t] = f"{v:.0f} km ({pct:.0f}%)" if v > 0 else '—'
                row['Total'] = f"{tot:.0f} km"
                rows_km.append(row)

            if rows_km:
                st.dataframe(pd.DataFrame(rows_km),
                             width="stretch", hide_index=True)

    st.markdown("---")

    # ── Atividades Recentes ──
    st.subheader("📋 Atividades Recentes")
    df_tab = filtrar_principais(da).sort_values('Data', ascending=False).head(10)
    if len(df_tab) > 0:
        cs = [c for c in ['Data', 'type', 'name', 'moving_time',
                           'rpe', 'power_avg', 'icu_eftp'] if c in df_tab.columns]
        ds = df_tab[cs].copy()
        if 'moving_time' in ds.columns:
            ds['moving_time'] = ds['moving_time'].apply(
                lambda x: f"{int(x/3600)}h{int((x%3600)/60):02d}m"
                if pd.notna(x) else '—')
        ds.columns = [c.replace('_', ' ').title() for c in ds.columns]
        st.dataframe(ds, width="stretch", hide_index=True)

    st.markdown("---")

    # ── Resumo Semanal ──
    st.subheader("📋 Resumo Semanal")
    col1, col2, col3 = st.columns(3)
    if len(da) > 0:
        dw7 = da[pd.to_datetime(da['Data']).dt.date >=
                 (datetime.now().date() - timedelta(days=7))]
        col1.metric("Sessões (7d)", len(dw7))
        if 'moving_time' in dw7.columns:
            col2.metric("Horas (7d)", fmt_dur_sec(dw7['moving_time'].sum()))
        if 'rpe' in dw7.columns and dw7['rpe'].notna().any():
            col3.metric("RPE médio (7d)", f"{dw7['rpe'].mean():.1f}")

    # ── Top 10 por Potência ──
    df_rank = filtrar_principais(da).copy()
    if 'power_avg' in df_rank.columns and df_rank['power_avg'].notna().any():
        st.subheader("🏆 Top 10 por Potência")
        top = df_rank.nlargest(10, 'power_avg')[
            ['Data', 'type', 'name', 'power_avg', 'rpe']].copy()
        top['Data'] = pd.to_datetime(top['Data']).dt.strftime('%Y-%m-%d')
        top.columns = ['Data', 'Tipo', 'Nome', 'Power (W)', 'RPE']
        st.dataframe(top, width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PMC + FTLM
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_pmc.py
# ════════════════════════════════════════════════════════════════════════════

def tab_pmc(da):
    """
    PMC — icu_training_load para CTL/ATL/FTLM.
    Barras de Load = TRIMP (session_rpe).
    Tabelas mensais: eFTP por modalidade + KM/KJ por modalidade (como Intervals.icu).
    Gráfico KM semanal stacked com linha de média por modalidade.
    """
    st.header("📈 PMC — Performance Management Chart")
    if len(da) == 0: st.warning("Sem dados de atividades."); return

    da_full = st.session_state.get('da_full', da)
    _mods = st.session_state.get('mods_sel', None)
    if _mods and 'type' in da_full.columns:
        da_full = da_full[da_full['type'].isin(_mods + ['WeightTraining'])]
    df = filtrar_principais(da_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    # ── CTL/ATL: icu_training_load primeiro, session_rpe fallback ──
    if 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        df['icu_tl'] = pd.to_numeric(df['icu_training_load'], errors='coerce').fillna(0)
        _metrica_ctl = "icu_training_load (Intervals.icu)"
    elif 'moving_time' in df.columns and 'rpe' in df.columns and df['rpe'].notna().sum() > 10:
        _rpe = pd.to_numeric(df['rpe'], errors='coerce')
        df['icu_tl'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * _rpe.fillna(_rpe.median())
        _metrica_ctl = "session_rpe (fallback)"
    else:
        st.warning("Sem icu_training_load nem RPE para calcular CTL/ATL."); return

    # ── Load bars: TRIMP = session_rpe ──
    _rpe2 = pd.to_numeric(df['rpe'], errors='coerce') if 'rpe' in df.columns else pd.Series(dtype=float)
    _mt   = pd.to_numeric(df['moving_time'], errors='coerce') if 'moving_time' in df.columns else pd.Series(dtype=float)
    if _rpe2.notna().sum() > 5:
        df['trimp_val'] = (_mt / 60) * _rpe2.fillna(_rpe2.median())
    else:
        df['trimp_val'] = df['icu_tl']

    # ── Série diária CTL/ATL ──
    ld = df.groupby('Data')['icu_tl'].sum().reset_index().sort_values('Data')
    idx_full = pd.date_range(ld['Data'].min(), datetime.now().date())
    ld = ld.set_index('Data').reindex(idx_full, fill_value=0).reset_index()
    ld.columns = ['Data', 'load_val']
    ld['CTL']  = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL']  = ld['load_val'].ewm(span=7,  adjust=False).mean()
    ld['TSB']  = ld['CTL'] - ld['ATL']

    best_g, best_r = 0.30, -1.0
    for g in np.arange(0.25, 0.36, 0.01):
        ema = ld['load_val'].ewm(alpha=g, adjust=False).mean()
        if ema.std() > 0:
            r = abs(np.corrcoef(ld['load_val'].values, ema.values)[0, 1])
            if r > best_r: best_r, best_g = r, g
    ld['FTLM'] = ld['load_val'].ewm(alpha=best_g, adjust=False).mean()
    u = ld.iloc[-1]

    st.caption(f"CTL/ATL/FTLM: **{_metrica_ctl}** | "
               f"Barras Load: **TRIMP (session_rpe)** | Histórico: {len(ld)} dias")

    # ── Controlos ──
    col1, col2, col3 = st.columns(3)
    dias_opts = {"30 dias": 30, "60 dias": 60, "90 dias": 90,
                 "180 dias": 180, "1 ano": 365, "Todo histórico": len(ld)}
    dias_exib = dias_opts[col1.selectbox("Período exibido", list(dias_opts.keys()), index=2)]
    ld_plot   = ld.tail(dias_exib).copy()
    smooth    = col2.checkbox("Suavizar CTL/ATL (3d)", value=False)
    show_ftlm = col3.checkbox("Mostrar FTLM", value=True)
    if smooth:
        ld_plot['CTL'] = ld_plot['CTL'].rolling(3, min_periods=1).mean()
        ld_plot['ATL'] = ld_plot['ATL'].rolling(3, min_periods=1).mean()

    # ── GRÁFICO 1: PMC + Load (TRIMP) ──
    fig, (ax_pmc, ax_load) = plt.subplots(
        2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ax_pmc.plot(ld_plot['Data'], ld_plot['CTL'], label='CTL (Fitness)',
                color=CORES['azul'], linewidth=2.5)
    ax_pmc.plot(ld_plot['Data'], ld_plot['ATL'], label='ATL (Fadiga)',
                color=CORES['vermelho'], linewidth=2.5)
    ax_pmc.fill_between(ld_plot['Data'], 0, ld_plot['TSB'],
                        where=(ld_plot['TSB'] >= 0),
                        color=CORES['verde'], alpha=0.25, label='TSB+ (Forma)')
    ax_pmc.fill_between(ld_plot['Data'], 0, ld_plot['TSB'],
                        where=(ld_plot['TSB'] < 0),
                        color=CORES['vermelho'], alpha=0.20, label='TSB- (Fadiga)')
    ax_pmc.axhline(0, color=CORES['cinza'], linestyle='--', linewidth=0.8)
    ax_pmc.set_ylabel('CTL / ATL / TSB', fontweight='bold')
    ax_pmc.grid(True, alpha=0.3)
    if show_ftlm:
        ax2 = ax_pmc.twinx()
        ax2.plot(ld_plot['Data'], ld_plot['FTLM'],
                 label=f'FTLM (gamma={best_g:.2f})',
                 color=CORES['laranja'], linewidth=2, linestyle='--', alpha=0.85)
        ax2.set_ylabel('FTLM', color=CORES['laranja'], fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=CORES['laranja'])
        l1, lb1 = ax_pmc.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax_pmc.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=9)
    else:
        ax_pmc.legend(loc='upper left', fontsize=9)
    ax_pmc.text(0.99, 0.97,
                f"CTL: {u['CTL']:.1f}  |  ATL: {u['ATL']:.1f}  |  TSB: {u['TSB']:+.1f}",
                transform=ax_pmc.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax_pmc.set_title('PMC — CTL / ATL / TSB / FTLM', fontsize=14, fontweight='bold')

    trimp_d = df.groupby(['Data', 'type'])['trimp_val'].sum().reset_index()
    trimp_d['Data'] = pd.to_datetime(trimp_d['Data'])
    tipos_ord = [t for t in ['Bike', 'Row', 'Ski', 'Run', 'WeightTraining']
                 if t in trimp_d['type'].unique()]
    tipos_ord += [t for t in trimp_d['type'].unique() if t not in tipos_ord]
    bot = np.zeros(len(ld_plot))
    for tipo in tipos_ord:
        dt = trimp_d[trimp_d['type'] == tipo][['Data', 'trimp_val']]
        merged = ld_plot[['Data']].merge(dt, on='Data', how='left').fillna(0)
        ax_load.bar(ld_plot['Data'], merged['trimp_val'].values, bottom=bot,
                    color=get_cor(tipo), alpha=0.85, width=0.8, label=tipo,
                    edgecolor='white', linewidth=0.3)
        bot += merged['trimp_val'].values
    ax_load.legend(loc='upper left', fontsize=8, ncol=min(5, len(tipos_ord)))
    ax_load.set_ylabel('Load\n(TRIMP)', fontweight='bold', fontsize=9)
    ax_load.grid(True, alpha=0.2, axis='y')
    ax_load.tick_params(axis='x', rotation=45)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── RESUMO PMC ──
    st.subheader("📊 Resumo PMC")
    tsb_v = u['TSB']
    if   tsb_v >  25: tsb_i = "🟢 Forma — pronto para competir/esforço máximo"
    elif tsb_v >   5: tsb_i = "🟡 Fresco — bom equilíbrio treino/recuperação"
    elif tsb_v > -10: tsb_i = "🟠 Neutro — zona de manutenção"
    elif tsb_v > -25: tsb_i = "🔴 Fatigado — carga elevada, monitorar recuperação"
    else:              tsb_i = "⛔ Sobrecarregado — reduzir carga imediatamente"
    resumo = pd.DataFrame([
        {'Métrica': 'CTL (Fitness atual)',  'Valor': f"{u['CTL']:.1f}",
         'Interpretação': 'Capacidade aeróbica crónica (42d). Maior = melhor condição.'},
        {'Métrica': 'ATL (Fadiga atual)',   'Valor': f"{u['ATL']:.1f}",
         'Interpretação': 'Fadiga aguda (7d). Maior = mais fatigado recentemente.'},
        {'Métrica': 'TSB (Forma atual)',    'Valor': f"{u['TSB']:+.1f}",
         'Interpretação': tsb_i},
        {'Métrica': 'CTL max histórico',    'Valor': f"{ld['CTL'].max():.1f}",
         'Interpretação': 'Pico de fitness no período carregado.'},
        {'Métrica': 'ATL max histórico',    'Valor': f"{ld['ATL'].max():.1f}",
         'Interpretação': 'Pico de fadiga no período carregado.'},
    ])
    st.dataframe(resumo, width="stretch", hide_index=True)

    # ── FTLM — explicação + resultado atual ──
    st.subheader("🔁 FTLM — Fast Training Load Monitor")
    ftlm_v = u['FTLM']
    ctl_v  = u['CTL']
    pct    = (ftlm_v / ctl_v * 100) if ctl_v > 0 else 0
    if   pct > 110: ftlm_i = "⚠️ Carga muito acima do crónico — risco de overreaching"
    elif pct > 100: ftlm_i = "🔴 Carga acima do CTL — fase de acumulação/sobrecarga"
    elif pct >  90: ftlm_i = "🟡 Ligeiramente abaixo do CTL — manutenção/tapering leve"
    elif pct >  75: ftlm_i = "🟢 Tapering activo — carga a baixar, forma a subir"
    else:            ftlm_i = "⬇️ Destreino — carga muito abaixo do nível crónico"

    with st.expander("📖 O que é o FTLM e como interpretar", expanded=True):
        st.markdown(f"""
**FTLM (Fast Training Load Monitor)** é uma média exponencial da carga diária com
um factor gamma (γ) **optimizado automaticamente** por correlação com os teus dados.

| Parâmetro | Valor actual |
|---|---|
| Gamma (γ) | `{best_g:.3f}` |
| FTLM actual | `{ftlm_v:.1f}` |
| CTL actual | `{ctl_v:.1f}` |
| FTLM / CTL | `{pct:.0f}%` |
| Interpretação | {ftlm_i} |

**Como interpretar:**
- **FTLM > CTL (>100%)** — Carga recente **acima** da capacidade crónica. Bloco de carga intenso. Monitorar recuperação.
- **FTLM ≈ CTL (90–110%)** — Carga estável. Manutenção do nível actual.
- **FTLM < CTL (<90%)** — Carga recente **abaixo** do crónico. Tapering intencional ou destreino.

**Diferença para o ATL:**
O ATL usa sempre span=7 (fixo). O FTLM usa γ=`{best_g:.3f}`
(equivalente a span≈{int(round(2/best_g - 1))}), **optimizado para os teus dados**,
tornando-o mais sensível ao teu padrão específico de treino.
        """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — VOLUME & CARGA
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_volume.py
# ════════════════════════════════════════════════════════════════════════════

def tab_volume(da, dw):
    st.header("📦 Volume & Carga")
    if len(da) == 0: st.warning("Sem dados de atividades."); return
    df = filtrar_principais(da).copy()
    df = add_tempo(df); df['Data'] = pd.to_datetime(df['Data'])
    df['horas'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 3600).fillna(0)
    ciclicos = ['Bike', 'Run', 'Row', 'Ski']
    CORES_MOD = {'Bike': CORES['vermelho'], 'Run': CORES['verde'], 'Row': CORES['azul'], 'Ski': CORES['roxo'], 'WeightTraining': CORES['laranja']}

    st.subheader("🚴 Volume Mensal — Atividades Cíclicas (horas)")
    df_cic = df[df['type'].isin(ciclicos)].copy()
    if len(df_cic) > 0:
        pivot = df_cic.pivot_table(index='mes', columns='type', values='horas', aggfunc='sum', fill_value=0).sort_index()
        fig, ax = plt.subplots(figsize=(14, 6))
        bottom = np.zeros(len(pivot))
        for tipo in [t for t in ciclicos if t in pivot.columns]:
            vals = pivot[tipo].values
            ax.bar(range(len(pivot)), vals, bottom=bottom, label=tipo, color=CORES_MOD.get(tipo, 'gray'), alpha=0.85, edgecolor='white')
            bottom += vals
        totais = pivot.sum(axis=1).values
        for i, t in enumerate(totais):
            if t > 0: ax.text(i, t + 0.1, f'{t:.1f}h', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(pivot))); ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        media = totais.mean(); ax.axhline(media, color='black', linestyle='--', alpha=0.5, label=f'Média: {media:.1f}h')
        ax.set_ylabel('Horas', fontweight='bold'); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        c1, c2 = st.columns(2)
        c1.metric("Total horas cíclicos", f"{pivot.values.sum():.1f}h")
        c2.metric("Média mensal", f"{media:.1f}h")

    st.subheader("🏋️ Volume Mensal — WeightTraining (horas)")
    df_wt = da[da['type'] == 'WeightTraining'].copy()
    if len(df_wt) > 0:
        df_wt = add_tempo(df_wt); df_wt['horas'] = (pd.to_numeric(df_wt['moving_time'], errors='coerce') / 3600).fillna(0)
        mensal = df_wt.groupby('mes').agg(horas=('horas', 'sum'), sessoes=('Data', 'count')).reset_index().sort_values('mes')
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(mensal)), mensal['horas'], color=CORES['laranja'], alpha=0.8, edgecolor='white')
        for i, (h, s) in enumerate(zip(mensal['horas'], mensal['sessoes'])):
            if h > 0: ax.text(i, h + 0.05, f'{h:.1f}h\n({s}x)', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(mensal))); ax.set_xticklabels(mensal['mes'], rotation=45, ha='right')
        media_wt = mensal['horas'].mean()
        ax.axhline(media_wt, color='red', linestyle='--', alpha=0.7, label=f'Média: {media_wt:.1f}h')
        ax.set_ylabel('Horas', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Sem sessões de WeightTraining no período.")

    st.subheader("💥 Strain Score (XSS)")
    xss_col = next((c for c in ['xss', 'SS', 'XSS'] if c in df.columns and df[c].notna().any()), None)
    if xss_col:
        df_xss = df[df['type'].isin(ciclicos)].dropna(subset=[xss_col]).copy()
        if len(df_xss) > 3:
            df_xss = df_xss.sort_values('Data'); df_xss['xss_s'] = pd.to_numeric(df_xss[xss_col], errors='coerce').rolling(7, min_periods=1).mean()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(df_xss['Data'], pd.to_numeric(df_xss[xss_col], errors='coerce'), alpha=0.4, label='XSS')
            ax1.plot(df_xss['Data'], df_xss['xss_s'], linewidth=2.5, label='XSS 7d')
            ax1.set_title('Evolução XSS', fontweight='bold'); ax1.legend(); ax1.tick_params(axis='x', rotation=45)
            comp = [c for c in ['glycolytic', 'aerobic', 'pmax'] if c in df_xss.columns]
            if comp:
                med = df_xss.groupby('type')[comp].mean().fillna(0)
                med.plot(kind='bar', stacked=True, ax=ax2, color=[CORES['vermelho'], CORES['verde'], CORES['laranja']][:len(comp)])
                ax2.set_title('Componentes por Tipo', fontweight='bold'); ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📊 Volume de Horas por Intensidade (Trimestral)")
    if 'rpe' in df.columns and 'moving_time' in df.columns:
        df_rpe = df[df['type'].isin(ciclicos)].copy()
        df_rpe['rpe_cat'] = df_rpe['rpe'].apply(classificar_rpe); df_rpe = df_rpe.dropna(subset=['rpe_cat'])
        if len(df_rpe) > 0:
            piv = df_rpe.pivot_table(index='trimestre', columns='rpe_cat', values='horas', aggfunc='sum', fill_value=0).sort_index()
            CORES_RPE = {'Leve': CORES['verde'], 'Moderado': CORES['laranja'], 'Pesado': CORES['vermelho']}
            fig, ax = plt.subplots(figsize=(13, 5))
            bottom = np.zeros(len(piv))
            for cat in ['Leve', 'Moderado', 'Pesado']:
                if cat in piv.columns:
                    vals = piv[cat].values
                    ax.bar(range(len(piv)), vals, bottom=bottom, label=cat, color=CORES_RPE.get(cat, 'gray'), alpha=0.85, edgecolor='white')
                    for i, (v, b) in enumerate(zip(vals, bottom)):
                        if v > 0.5: ax.text(i, b + v / 2, f'{v:.1f}h', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
                    bottom += vals
            ax.set_xticks(range(len(piv))); ax.set_xticklabels(piv.index, rotation=45, ha='right')
            ax.set_ylabel('Horas', fontweight='bold'); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3, axis='y')
            ax.set_title('Volume de Horas por Intensidade RPE (Trimestral)', fontsize=12, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — eFTP
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_eftp.py
# ════════════════════════════════════════════════════════════════════════════

def tab_eftp(da, mods_sel, da_full=None):
    st.header("⚡ Evolução do eFTP por Modalidade")
    ecol = next((c for c in ['icu_eftp', 'eFTP', 'eftp', 'EFTP'] if c in da.columns), None)
    if ecol is None: st.warning("Coluna eFTP não encontrada."); return
    df = filtrar_principais(da).copy(); df['Data'] = pd.to_datetime(df['Data']); df['ano'] = df['Data'].dt.year
    df[ecol] = pd.to_numeric(df[ecol], errors='coerce'); df = df[df['type'].isin(mods_sel)].dropna(subset=[ecol]); df = df[df[ecol] > 50]
    if len(df) == 0: st.warning("Sem dados de eFTP."); return
    anos = sorted(df['ano'].unique()); CANO = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']
    mapa_cor = {a: CANO[i % len(CANO)] for i, a in enumerate(anos)}
    anos_sel = st.multiselect("Filtrar anos", anos, default=list(anos)); df = df[df['ano'].isin(anos_sel)]
    mods = [m for m in mods_sel if m in df['type'].values]
    if not mods: st.info("Nenhuma modalidade com eFTP."); return
    fig, axes = plt.subplots(1, len(mods), figsize=(7 * len(mods), 6))
    if len(mods) == 1: axes = [axes]
    for ax, mod in zip(axes, mods):
        dm = df[df['type'] == mod].sort_values('Data'); cm = get_cor(mod)
        for ano in anos_sel:
            da_ = dm[dm['ano'] == ano]
            if len(da_) == 0: continue
            ax.scatter(da_['Data'], da_[ecol], color=mapa_cor[ano], alpha=0.65, s=35, label=str(ano))
            if len(da_) >= 3:
                xn = (da_['Data'] - da_['Data'].min()).dt.days.values; coef = np.polyfit(xn, da_[ecol].values, 1)
                xp = np.array([xn.min(), xn.max()]); yp = np.poly1d(coef)(xp)
                dp = [da_['Data'].min() + pd.Timedelta(days=int(x)) for x in xp]
                ax.plot(dp, yp, color=mapa_cor[ano], linewidth=2, linestyle='--', alpha=0.9)
                sm = coef[0] * 30
                ax.annotate(f'{sm:+.1f}W/mês', xy=(dp[1], yp[1]), xytext=(5, 2), textcoords='offset points', fontsize=7.5, color=mapa_cor[ano], fontweight='bold')
        if len(dm) >= 5:
            roll = dm.set_index('Data')[ecol].resample('7D').mean().interpolate()
            ax.plot(roll.index, roll.values, color=cm, linewidth=2, alpha=0.4, label='Média 7d')
        if len(dm) > 0:
            mx = dm[ecol].max()
            ax.axhline(mx, color=cm, linestyle=':', linewidth=1.2, alpha=0.6)
            ax.annotate(f'Máx: {mx:.0f}W', xy=(dm.loc[dm[ecol].idxmax(), 'Data'], mx), xytext=(0, 6), textcoords='offset points', fontsize=8, color=cm, fontweight='bold', ha='center')
        ax.set_title(f'eFTP — {mod}', fontsize=13, fontweight='bold', color=cm)
        ax.set_xlabel('Data'); ax.set_ylabel('eFTP (W)'); ax.tick_params(axis='x', rotation=45); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    plt.suptitle('Evolução eFTP por Modalidade', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📦 RPE por Modalidade")
    if 'rpe' in da.columns:
        df_r = filtrar_principais(da).copy(); df_r = add_tempo(df_r); df_r = df_r[df_r['type'].isin(mods_sel)].dropna(subset=['rpe'])
        if len(df_r) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            tipos = [t for t in mods_sel if t in df_r['type'].values]
            sns.boxplot(data=df_r, x='type', y='rpe', order=tipos, palette={t: get_cor(t) for t in tipos}, ax=ax1)
            ax1.set_title('RPE por Modalidade', fontweight='bold'); ax1.tick_params(axis='x', rotation=45)
            if 'mes' in df_r.columns:
                meses = sorted(df_r['mes'].unique())[-12:]; df_rm = df_r[df_r['mes'].isin(meses)]
                sns.violinplot(data=df_rm, x='mes', y='rpe', palette='Set2', ax=ax2)
                ax2.set_title('RPE por Mês', fontweight='bold'); ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    # ── TABELAS eFTP + KM/KJ — filtro próprio + agrupamento ─────────────────────
    st.markdown("---")
    st.subheader("📅 Tabelas históricas por modalidade")

    # Tabelas usam histórico COMPLETO (da_full) independente do filtro do sidebar
    # Assim o filtro próprio das tabelas funciona sobre todos os dados disponíveis
    # da_full param (passed from main) or session_state fallback or da
    _da_full_tab = da_full if da_full is not None and len(da_full) > 0 else st.session_state.get('da_full', da)
    _df_full_tab = filtrar_principais(_da_full_tab).copy()
    _df_full_tab['Data'] = pd.to_datetime(_df_full_tab['Data'])

    # ── Filtro de período (só para tabelas — gráficos usam filtro sidebar) ──
    _c1, _c2, _c3 = st.columns([2, 1, 1])
    _tab_opts = {
        "Últimos 3 meses": 90,  "Últimos 6 meses": 180,
        "Último ano": 365,      "Últimos 2 anos": 730,
        "Últimos 3 anos": 1095, "Todo histórico": 9999,
        "Datas manuais": -1,
    }
    _tab_sel  = _c1.selectbox("Período das tabelas",
                               list(_tab_opts.keys()), index=3,
                               key="pmc_tab_periodo")
    _tab_dias = _tab_opts[_tab_sel]
    if _tab_dias == -1:
        _tab_di  = _c2.date_input("Início", datetime(2017, 1, 1).date(), key="pmc_tab_di")
        _tab_df_ = _c3.date_input("Fim",    datetime.now().date(),       key="pmc_tab_df")
    else:
        _tab_df_ = datetime.now().date()
        _tab_di  = (_tab_df_ - timedelta(days=_tab_dias)
                    if _tab_dias < 9999 else datetime(2017, 1, 1).date())
        _c2.caption(f"De {_tab_di.strftime('%d/%m/%Y')}")
        _c3.caption(f"Até {_tab_df_.strftime('%d/%m/%Y')}")

    df_tab = _df_full_tab[
        (_df_full_tab['Data'] >= pd.Timestamp(_tab_di)) &
        (_df_full_tab['Data'] <= pd.Timestamp(_tab_df_))
    ].copy()
    df_tab = df_tab[df_tab['type'] != 'WeightTraining']
    st.caption(f"📊 {len(df_tab)} actividades "
               f"({_tab_di.strftime('%d/%m/%Y')} → {_tab_df_.strftime('%d/%m/%Y')})")

    # ── Função auxiliar: agregar período ─────────────────────────────────────
    def _agrupar(df_in, agrup):
        """Agrupa df_in por período (Ano / Mês / Semana) e devolve coluna label."""
        df_in = df_in.copy()
        if agrup == "Ano":
            df_in['_periodo'] = df_in['Data'].dt.to_period('Y')
            fmt = lambda p: str(p.year)
        elif agrup == "Semana":
            df_in['_periodo'] = df_in['Data'].dt.to_period('W')
            fmt = lambda p: f"Sem {p.start_time.strftime('%d/%m/%y')}"
        else:  # Mês (default)
            df_in['_periodo'] = df_in['Data'].dt.to_period('M')
            fmt = lambda p: pd.to_datetime(str(p)).strftime('%B %Y').title()
        return df_in, fmt

    # ── Função auxiliar: linha de tendência ──────────────────────────────────
    def _tendencia(series_num):
        """Calcula slope da regressão linear sobre a série numérica. Retorna sinal."""
        s = series_num.dropna().reset_index(drop=True)
        if len(s) < 3: return None
        x = np.arange(len(s), dtype=float)
        from scipy.stats import linregress
        sl, _, _, pv, _ = linregress(x, s.values)
        pct = (sl / s.mean() * 100) if s.mean() != 0 else 0
        sig = pv < 0.10  # p<10% como threshold de tendência
        if not sig or abs(pct) < 2: return "→ Estável"
        return f"↗ +{abs(pct):.1f}%/período" if sl > 0 else f"↘ -{abs(pct):.1f}%/período"

    # ════════════════════════════════════════════════════════════════
    # TABELA eFTP — com agrupamento ao lado do título
    # ════════════════════════════════════════════════════════════════
    _hdr1, _agr1 = st.columns([3, 1])
    _hdr1.markdown("**eFTP por modalidade**")
    _agrup_eftp = _agr1.selectbox("Agrupar por", ["Mês", "Ano", "Semana"],
                                   key="pmc_agrup_eftp")

    if 'icu_eftp' in df_tab.columns and df_tab['icu_eftp'].notna().any():
        tipos_eftp = [t for t in ['Bike', 'Run', 'Ski', 'Row']
                      if t in df_tab['type'].unique()]

        # Agregar por período escolhido
        df_tab_e, fmt_e = _agrupar(df_tab, _agrup_eftp)
        eftp_pivot = {}
        for tipo in tipos_eftp:
            df_t = (df_tab_e[df_tab_e['type'] == tipo][['_periodo', 'icu_eftp']]
                    .dropna())
            if len(df_t) == 0: continue
            eftp_pivot[tipo] = (df_t.groupby('_periodo')['icu_eftp']
                                .max().reset_index()
                                .rename(columns={'icu_eftp': tipo}))

        if eftp_pivot:
            df_e = None
            for tipo, dft in eftp_pivot.items():
                df_e = dft if df_e is None else df_e.merge(dft, on='_periodo', how='outer')
            df_e = df_e.sort_values('_periodo', ascending=False)

            num_e = [t for t in tipos_eftp if t in df_e.columns]
            rows_e = []
            for _, r in df_e.iterrows():
                row = {'Período': fmt_e(r['_periodo'])}
                for t in num_e:
                    row[f'{t} eFTP'] = f"{r[t]:.0f}w" if pd.notna(r.get(t)) else '—'
                rows_e.append(row)

            # Avg
            avg_e = {'Período': 'Avg'}
            for t in num_e:
                v = df_e[t].dropna()
                avg_e[f'{t} eFTP'] = f"{v.mean():.0f}w" if len(v) > 0 else '—'
            rows_e.append(avg_e)

            # Tendência
            tend_e = {'Período': 'Tendência'}
            for t in num_e:
                tr = _tendencia(df_e[t])
                tend_e[f'{t} eFTP'] = tr if tr else '—'
            rows_e.append(tend_e)

            st.dataframe(pd.DataFrame(rows_e),
                         width="stretch", hide_index=True)
    else:
        st.caption("Sem dados de eFTP disponíveis.")

    # ════════════════════════════════════════════════════════════════
    # TABELAS KM / Moving Time / kJ / Sessions — uma por modalidade
    # ════════════════════════════════════════════════════════════════
    tipos_vol = [t for t in ['Bike', 'Ski', 'Row', 'Run']
                 if t in df_tab['type'].unique()]

    for tipo in tipos_vol:
        df_t = df_tab[df_tab['type'] == tipo].copy()
        if len(df_t) == 0: continue

        _hdr2, _agr2 = st.columns([3, 1])
        _hdr2.markdown(f"**{tipo} — Distância, Tempo, kJ e Sessões**")
        _agrup_vol = _agr2.selectbox("Agrupar por", ["Mês", "Ano", "Semana"],
                                      key=f"pmc_agrup_{tipo}")

        df_t_a, fmt_v = _agrupar(df_t, _agrup_vol)

        # kJ: icu_joules (J → kJ dividindo por 1000)
        if 'icu_joules' in df_t_a.columns and df_t_a['icu_joules'].notna().any():
            df_t_a['_kj'] = pd.to_numeric(df_t_a['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in df_t_a.columns and df_t_a['power_avg'].notna().any():
            df_t_a['_kj'] = (pd.to_numeric(df_t_a['power_avg'], errors='coerce') *
                              pd.to_numeric(df_t_a['moving_time'], errors='coerce') / 1000)
        else:
            df_t_a['_kj'] = np.nan

        if 'distance' in df_t_a.columns:
            df_t_a['_km'] = pd.to_numeric(df_t_a['distance'], errors='coerce') / 1000
        else:
            df_t_a['_km'] = np.nan

        df_t_a['_mt'] = pd.to_numeric(df_t_a['moving_time'], errors='coerce').fillna(0)

        agg = df_t_a.groupby('_periodo').agg(
            _km_s=('_km',  'sum'),
            _mt_s=('_mt',  'sum'),
            _kj_s=('_kj',  'sum'),
            _ses=('Data',  'count'),
        ).reset_index().sort_values('_periodo', ascending=False)

        rows_v = []
        for _, r in agg.iterrows():
            mt_h = int(r['_mt_s'] // 3600); mt_m = int((r['_mt_s'] % 3600) // 60)
            rows_v.append({
                'Período':       fmt_v(r['_periodo']),
                'Distance':      f"{r['_km_s']:.0f} km" if pd.notna(r['_km_s']) and r['_km_s'] > 0 else '—',
                'Moving Time':   f"{mt_h}h{mt_m:02d}m",
                'kJ':            f"{r['_kj_s']:.0f}" if pd.notna(r['_kj_s']) and r['_kj_s'] > 0 else '—',
                'Sessions':      str(int(r['_ses'])),
            })

        if not rows_v: continue

        # Avg
        avg_km  = agg['_km_s'][agg['_km_s'] > 0].mean()   if (agg['_km_s'] > 0).any() else None
        avg_mt  = agg['_mt_s'].mean()
        avg_kj  = agg['_kj_s'][agg['_kj_s'] > 0].mean()   if (agg['_kj_s'] > 0).any() else None
        avg_ses = agg['_ses'].mean()
        avg_h   = int(avg_mt // 3600); avg_m = int((avg_mt % 3600) // 60)
        rows_v.append({
            'Período':     'Avg',
            'Distance':    f"{avg_km:.0f} km" if avg_km else '—',
            'Moving Time': f"{avg_h}h{avg_m:02d}m",
            'kJ':          f"{avg_kj:.0f}" if avg_kj else '—',
            'Sessions':    f"{avg_ses:.0f}",
        })

        # Tendência — regressão linear sobre série cronológica (ascending)
        agg_asc = agg.sort_values('_periodo', ascending=True)
        tend_km  = _tendencia(agg_asc['_km_s'])
        tend_mt  = _tendencia(agg_asc['_mt_s'])
        tend_kj  = _tendencia(agg_asc['_kj_s'])
        tend_ses = _tendencia(agg_asc['_ses'].astype(float))
        rows_v.append({
            'Período':     'Tendência',
            'Distance':    tend_km  if tend_km  else '—',
            'Moving Time': tend_mt  if tend_mt  else '—',
            'kJ':          tend_kj  if tend_kj  else '—',
            'Sessions':    tend_ses if tend_ses else '—',
        })

        st.dataframe(pd.DataFrame(rows_v),
                     width="stretch", hide_index=True)


    st.markdown("---")
    # ── CORRELAÇÕES: variáveis de carga vs eFTP ─────────────────────────────────
    st.subheader("🔗 O que está correlacionado com o eFTP?")
    st.caption("Correlação semanal, mensal e anual entre variáveis de carga e eFTP. "
               "Apenas correlações moderadas/fortes e estatisticamente significativas são mostradas.")

    # Correlações usam histórico completo
    _df_corr = _df_full_tab.copy() if '_df_full_tab' in locals() else df.copy()
    if 'icu_eftp' in _df_corr.columns and _df_corr['icu_eftp'].notna().any():
        from scipy.stats import spearmanr

        def _cv_ok(series, max_cv=50):
            """Retorna True se CV% da série for aceitável (não muito disperso)."""
            s = series.dropna()
            if len(s) < 3 or s.mean() == 0: return False
            return (s.std() / s.mean() * 100) < max_cv

        def _mdc_ok(eftp_series, icc=0.9):
            """Verifica se variação de eFTP excede MDC — confirma mudança real."""
            s = eftp_series.dropna()
            if len(s) < 3: return False
            std = s.std(ddof=1)
            sem = std * np.sqrt(1 - icc)
            mdc = sem * 1.96 * np.sqrt(2)
            return (s.max() - s.min()) > mdc

        def _forca(r):
            ar = abs(r)
            if ar >= 0.60: return "★★★ Forte"
            if ar >= 0.40: return "★★ Moderada"
            return None  # fraca — não mostrar

        def _corr_periodo(df_mod, periodo_label, periodo_code):
            """
            Agrega por período, filtra qualidade, calcula correlação Spearman
            entre variáveis de carga e eFTP.
            Retorna lista de resultados significativos.
            """
            results = []
            d = df_mod.copy()
            d['_p'] = d['Data'].dt.to_period(periodo_code)

            # eFTP: máximo do período
            eftp_agg = d.groupby('_p')['icu_eftp'].max()
            if eftp_agg.notna().sum() < 5: return results
            if not _mdc_ok(eftp_agg.dropna()): return results

            # Variáveis a testar
            vars_test = {}

            if 'icu_joules' in d.columns and d['icu_joules'].notna().any():
                kj = d.groupby('_p')['icu_joules'].sum() / 1000
                vars_test['KJ'] = kj

            if 'moving_time' in d.columns:
                hrs = d.groupby('_p')['moving_time'].sum() / 3600
                vars_test['Horas'] = hrs

            if 'distance' in d.columns and d['distance'].notna().any():
                km = d.groupby('_p')['distance'].sum() / 1000
                vars_test['KM'] = km

            sess = d.groupby('_p')['Data'].count()
            vars_test['Sessões'] = sess

            for var_name, var_series in vars_test.items():
                # Alinhar índices
                combined = pd.DataFrame({'eftp': eftp_agg, 'var': var_series}).dropna()
                if len(combined) < 5: continue
                # Filtro CV
                if not _cv_ok(combined['var']): continue
                # Correlação Spearman
                r, pv = spearmanr(combined['var'].values, combined['eftp'].values)
                if pv >= 0.10: continue  # não significativo
                forca = _forca(r)
                if forca is None: continue  # fraca — não mostrar
                results.append({
                    'Período': periodo_label,
                    'Variável': var_name,
                    'r (Spearman)': f"{r:+.2f}",
                    'p-value': f"{pv:.3f}",
                    'Força': forca,
                    'Correlação': (f"↗ {var_name} ↑ → eFTP ↑" if r > 0
                                   else f"↘ {var_name} ↑ → eFTP ↓"),
                })
            return results

        tipos_corr = [t for t in ['Bike', 'Run', 'Ski', 'Row']
                      if t in _df_corr['type'].unique()]

        for tipo in tipos_corr:
            df_mod = _df_corr[_df_corr['type'] == tipo].copy()
            if len(df_mod) < 10: continue

            all_results = []
            for label, code in [("Semanal", "W"), ("Mensal", "M"), ("Anual", "Y")]:
                all_results.extend(_corr_periodo(df_mod, label, code))

            if not all_results:
                continue  # sem correlações relevantes — não mostrar nada

            st.markdown(f"**{tipo}**")
            df_res = pd.DataFrame(all_results)
            # Remover duplicados (mesma variável em múltiplos períodos — mostrar o mais forte)
            df_res['_ar'] = df_res['r (Spearman)'].str.replace('+','',regex=False).astype(float).abs()
            df_res = (df_res.sort_values('_ar', ascending=False)
                            .drop_duplicates(subset=['Variável'], keep='first')
                            .drop(columns=['_ar'])
                            .sort_values('Força', ascending=True))
            st.dataframe(df_res, width="stretch", hide_index=True)

        if not any(
            len(df[df['type']==t]) >= 10 and
            'icu_eftp' in df.columns
            for t in tipos_corr
        ):
            st.info("Dados insuficientes para análise de correlação.")
    else:
        st.info("Coluna icu_eftp não disponível para análise de correlação.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — HR ZONES + RPE ZONES + CORRELAÇÃO
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_zones.py
# ════════════════════════════════════════════════════════════════════════════

def _zonas_bp(ax_bar, ax_pie, por_tipo, total_seg, cores_zona, tit_bar, tit_pie):
    zonas = list(cores_zona.keys()); bottom = np.zeros(len(por_tipo))
    for zona in zonas:
        if zona not in por_tipo.columns: continue
        vals = por_tipo[zona].values
        ax_bar.bar(por_tipo.index, vals, bottom=bottom, color=cores_zona[zona], label=zona, edgecolor='white', linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5: ax_bar.text(i, b + v / 2, f'{v:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        bottom += vals
    ax_bar.set_ylim(0, 100); ax_bar.set_ylabel('% sessões', fontweight='bold')
    ax_bar.set_title(tit_bar, fontweight='bold'); ax_bar.legend(loc='upper right', fontsize=8); ax_bar.grid(True, alpha=0.2, axis='y')
    lp = [l for l, v in total_seg.items() if v > 0]; sp = [v for v in total_seg.values() if v > 0]
    if sum(sp) > 0:
        _, _, ats = ax_pie.pie(sp, labels=lp, autopct='%1.1f%%', colors=[cores_zona[l] for l in lp], startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2), pctdistance=0.75)
        for at in ats: at.set_fontweight('bold'); at.set_fontsize(10)
    ax_pie.set_title(tit_pie, fontweight='bold')

def tab_zones(da, mods_sel):
    st.header("❤️ HR Zones & RPE Zones")
    df = filtrar_principais(da).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    df['ano']  = df['Data'].dt.year
    df = df[df['type'].isin(mods_sel)]
    if len(df) == 0:
        st.warning("Sem dados.")
        return

    zonas_hr = [c for c in df.columns if c.lower().startswith('hr_z') and c.lower().endswith('_secs')]
    rpe_col  = next((c for c in ['rpe','RPE','icu_rpe'] if c in df.columns), None)

    def gzn(col):
        m = re.search(r'hr_z(\d+)_secs', col.lower())
        return int(m.group(1)) if m else 0

    CORES_HR  = {'Baixa (Z1+Z2)':'#2ECC71', 'Moderada (Z3+Z4)':'#F39C12', 'Alta (Z5+Z6+Z7)':'#E74C3C'}
    CORES_RPE = {'Leve (1–4)':'#3498DB',    'Moderado (5–7)':'#F39C12',    'Forte (7–10)':'#E74C3C'}

    def _prep_hr(df_in):
        dh = df_in.copy()
        bc = [c for c in zonas_hr if gzn(c) in (1,2)]
        mc = [c for c in zonas_hr if gzn(c) in (3,4)]
        ac = [c for c in zonas_hr if gzn(c) in (5,6,7)]
        for cols in [bc, mc, ac]:
            for c in cols:
                dh[c] = pd.to_numeric(dh[c], errors='coerce').fillna(0)
        dh['zb'] = dh[bc].sum(axis=1) if bc else 0
        dh['zm'] = dh[mc].sum(axis=1) if mc else 0
        dh['za'] = dh[ac].sum(axis=1) if ac else 0
        dh['zt'] = dh['zb'] + dh['zm'] + dh['za']
        dh = dh[dh['zt'] > 0].copy()
        for z, p in [('zb','pb'),('zm','pm'),('za','pa')]:
            dh[p] = dh[z] / dh['zt'] * 100
        return dh

    def _prep_rpe(df_in):
        if not rpe_col: return pd.DataFrame()
        dr = df_in.dropna(subset=[rpe_col]).copy()
        dr[rpe_col] = pd.to_numeric(dr[rpe_col], errors='coerce')
        dr = dr.dropna(subset=[rpe_col])
        dr['rz'] = pd.cut(dr[rpe_col], bins=[0,4.9,6.9,10],
                          labels=list(CORES_RPE.keys()), right=True)
        return dr.dropna(subset=['rz'])

    LEGEND = dict(orientation='h', y=-0.22, font=dict(color='#111111', size=11),
                  bgcolor='rgba(255,255,255,0.9)', bordercolor='#ccc', borderwidth=1)
    PW = dict(paper_bgcolor='white', plot_bgcolor='white',
              font=dict(color='#222222', family='Arial'))

    # ── Dropdown ano (só para gráficos) ──────────────────────────────────
    anos = sorted(df['ano'].unique())
    ano_sel = st.selectbox("📅 Ano (gráficos)", anos, index=len(anos)-1)
    df_ano  = df[df['ano'] == ano_sel].copy()

    # ════════════════════════════════════════════════════════════════════
    # GRÁFICOS — ano seleccionado
    # ════════════════════════════════════════════════════════════════════
    dh_ano = _prep_hr(df_ano)  if zonas_hr else pd.DataFrame()
    dr_ano = _prep_rpe(df_ano) if rpe_col   else pd.DataFrame()

    col1, col2 = st.columns(2)

    # HR por modalidade — barras stacked com % dentro
    if len(dh_ano) > 0:
        pt_hr = dh_ano.groupby('type')[['pb','pm','pa']].mean().reset_index()
        pt_hr.columns = ['type'] + list(CORES_HR.keys())

        fig_hr_mod = go.Figure()
        for zona, cor in CORES_HR.items():
            vals = pt_hr[zona]
            fig_hr_mod.add_trace(go.Bar(
                x=pt_hr['type'], y=vals, name=zona,
                marker_color=cor,
                text=[f"{v:.0f}%" for v in vals],
                textposition='inside',
                textfont=dict(color='white', size=11, family='Arial'),
                hovertemplate=f'{zona}: <b>%{{y:.1f}}%</b><extra></extra>'))
        fig_hr_mod.update_layout(**PW,
            barmode='stack',
            title=dict(text=f'❤️ HR Zones por Modalidade — {ano_sel}',
                       font=dict(color='#222', size=13)),
            height=380, legend=LEGEND,
            xaxis=dict(title='', tickfont=dict(color='#333333'), showgrid=False,
                       linecolor='#ccc', showline=True),
            yaxis=dict(title='%', tickfont=dict(color='#333333'),
                       showgrid=True, gridcolor='#eeeeee', range=[0,108]))
        with col1:
            st.plotly_chart(fig_hr_mod, use_container_width=True)

        # HR Geral pizza
        ts_hr = {z: dh_ano[c].sum()
                 for z, c in zip(CORES_HR.keys(), ['zb','zm','za'])}
        fig_hr_gen = go.Figure(go.Pie(
            labels=list(ts_hr.keys()), values=list(ts_hr.values()),
            marker=dict(colors=list(CORES_HR.values())),
            textinfo='label+percent',
            textfont=dict(color='white', size=11),
            hovertemplate='%{label}: <b>%{percent}</b><extra></extra>'))
        fig_hr_gen.update_layout(**PW,
            title=dict(text=f'❤️ HR Geral — {ano_sel}', font=dict(color='#222', size=13)),
            height=380, showlegend=True,
            legend=dict(font=dict(color='#111111', size=11)))
        with col2:
            st.plotly_chart(fig_hr_gen, use_container_width=True)

    # RPE por modalidade
    col3, col4 = st.columns(2)
    if len(dr_ano) > 0:
        piv_r = (dr_ano.groupby(['type','rz'], observed=True)
                 .size().unstack(fill_value=0))
        for z in CORES_RPE.keys():
            if z not in piv_r.columns: piv_r[z] = 0
        piv_r   = piv_r[list(CORES_RPE.keys())]
        pct_r   = piv_r.div(piv_r.sum(axis=1), axis=0) * 100
        pct_r   = pct_r.reset_index()

        fig_rpe_mod = go.Figure()
        for zona, cor in CORES_RPE.items():
            vals = pct_r[zona]
            fig_rpe_mod.add_trace(go.Bar(
                x=pct_r['type'], y=vals, name=zona,
                marker_color=cor,
                text=[f"{v:.0f}%" for v in vals],
                textposition='inside',
                textfont=dict(color='white', size=11, family='Arial'),
                hovertemplate=f'{zona}: <b>%{{y:.1f}}%</b><extra></extra>'))
        fig_rpe_mod.update_layout(**PW,
            barmode='stack',
            title=dict(text=f'🎯 RPE Zones por Modalidade — {ano_sel}',
                       font=dict(color='#222', size=13)),
            height=380, legend=LEGEND,
            xaxis=dict(title='', tickfont=dict(color='#333333'), showgrid=False,
                       linecolor='#ccc', showline=True),
            yaxis=dict(title='%', tickfont=dict(color='#333333'),
                       showgrid=True, gridcolor='#eeeeee', range=[0,108]))
        with col3:
            st.plotly_chart(fig_rpe_mod, use_container_width=True)

        ts_rpe = {z: piv_r[z].sum() for z in CORES_RPE.keys()}
        fig_rpe_gen = go.Figure(go.Pie(
            labels=list(ts_rpe.keys()), values=list(ts_rpe.values()),
            marker=dict(colors=list(CORES_RPE.values())),
            textinfo='label+percent',
            textfont=dict(color='white', size=11),
            hovertemplate='%{label}: <b>%{percent}</b><extra></extra>'))
        fig_rpe_gen.update_layout(**PW,
            title=dict(text=f'🎯 RPE Geral — {ano_sel}', font=dict(color='#222', size=13)),
            height=380, showlegend=True,
            legend=dict(font=dict(color='#111111', size=11)))
        with col4:
            st.plotly_chart(fig_rpe_gen, use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # TABELA ANUAL — separada por ano, linha Geral em cinza
    # ════════════════════════════════════════════════════════════════════
    st.subheader("📊 HR Zones & RPE Zones — todos os anos")
    st.caption("% média por modalidade cíclica. Linha 'Geral' em cinza claro.")

    rows_anual = []
    for ano in sorted(df['ano'].unique(), reverse=True):
        df_a  = df[df['ano'] == ano]
        dh_a  = _prep_hr(df_a)  if zonas_hr else pd.DataFrame()
        rpe_a = _prep_rpe(df_a) if rpe_col   else pd.DataFrame()

        mods_ano = sorted(set(
            (dh_a['type'].unique().tolist() if len(dh_a) > 0 else []) +
            (rpe_a['type'].unique().tolist() if len(rpe_a) > 0 else [])))

        for mod in mods_ano + ['Geral']:
            is_geral = (mod == 'Geral')
            row = {'Ano': str(ano) if not is_geral else '', 'Modalidade': mod}

            sub_h = dh_a  if is_geral else (dh_a[dh_a['type']==mod]   if len(dh_a)  > 0 else pd.DataFrame())
            sub_r = rpe_a if is_geral else (rpe_a[rpe_a['type']==mod]  if len(rpe_a) > 0 else pd.DataFrame())

            if len(sub_h) > 0:
                row['HR Z1+Z2%'] = f"{sub_h['pb'].mean():.0f}%"
                row['HR Z3+Z4%'] = f"{sub_h['pm'].mean():.0f}%"
                row['HR Z5+%']   = f"{sub_h['pa'].mean():.0f}%"
            else:
                row['HR Z1+Z2%'] = row['HR Z3+Z4%'] = row['HR Z5+%'] = '—'

            if len(sub_r) > 0:
                t = len(sub_r)
                for z, lbl in [('Leve (1–4)','RPE Z1%'),
                                ('Moderado (5–7)','RPE Z2%'),
                                ('Forte (7–10)','RPE Z3%')]:
                    row[lbl] = f"{(sub_r['rz']==z).sum()/t*100:.0f}%"
            else:
                row['RPE Z1%'] = row['RPE Z2%'] = row['RPE Z3%'] = '—'

            row['_geral'] = is_geral
            rows_anual.append(row)

    if rows_anual:
        # 1. Combinar HR|RPE ANTES de criar DataFrame
        for row in rows_anual:
            for z_hr, z_rpe, z_lbl in [
                ('HR Z1+Z2%','RPE Z1%','Z1 (HR|RPE)'),
                ('HR Z3+Z4%','RPE Z2%','Z2 (HR|RPE)'),
                ('HR Z5+%',  'RPE Z3%','Z3 (HR|RPE)'),
            ]:
                row[z_lbl] = f"{row.get(z_hr,'—')} | {row.get(z_rpe,'—')}"

        # 2. Criar DataFrame depois da combinação
        df_anual     = pd.DataFrame(rows_anual)
        display_cols = ['Ano','Modalidade',
                        'Z1 (HR|RPE)','Z2 (HR|RPE)','Z3 (HR|RPE)']

        # 3. Render HTML com linha Geral em cinza
        html  = ('<table style="border-collapse:collapse;width:100%;'
                 'font-size:12px;background:#fff;color:#222">')
        html += '<tr style="background:#e0e0e0">'
        for c in display_cols:
            html += (f'<th style="border:1px solid #ccc;padding:6px 10px;'
                     f'text-align:center;color:#111;font-weight:bold">{c}</th>')
        html += '</tr>'
        for _, row in df_anual.iterrows():
            bg = '#f0f0f0' if row.get('_geral', False) else '#ffffff'
            fw = 'bold'    if row.get('_geral', False) else 'normal'
            html += f'<tr style="background:{bg}">'
            for c in display_cols:
                html += (f'<td style="border:1px solid #ddd;padding:5px 10px;'
                         f'text-align:center;color:#222;font-weight:{fw}">'
                         f'{row.get(c,"—")}</td>')
            html += '</tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # CORRELAÇÃO HR Zone × RPE Zone — tabela simplificada
    # ════════════════════════════════════════════════════════════════════
    st.subheader("🔗 Correlação HR Zone × RPE Zone")
    st.caption(
        "Pearson r entre % de tempo em cada HR Zone e RPE numérico. "
        "**Força Forte/Muito Forte = HR Zone e RPE altamente alinhados** "
        "(quando RPE sobe, tempo nessa HR Zone sobe ou desce consistentemente).")

    if zonas_hr and rpe_col:
        dh_ok = _prep_hr(df.copy())
        if len(dh_ok) > 0:
            dh_ok[rpe_col] = pd.to_numeric(dh_ok[rpe_col], errors='coerce')
            dh_ok = dh_ok.dropna(subset=[rpe_col,'pb','pm','pa'])
            dh_ok = dh_ok[dh_ok['zt'] > 0]

            HR_VARS   = [('pb','Baixa (Z1+Z2)', 'Leve (1–4)'),
                         ('pm','Moderada (Z3+Z4)', 'Moderado (5–7)'),
                         ('pa','Alta (Z5+Z6+Z7)', 'Forte (7–10)')]
            forca_fn  = lambda r: ('Muito Forte' if abs(r) >= 0.7 else
                                   'Forte'       if abs(r) >= 0.5 else
                                   'Moderada'    if abs(r) >= 0.3 else 'Fraca')

            mods_corr = sorted(dh_ok['type'].unique().tolist())  # todos os tipos com dados
            rows_corr = []
            for mod in mods_corr:
                dm = dh_ok[dh_ok['type'] == mod]
                if len(dm) < 3: continue
                for hv, hr_lbl, rpe_lbl in HR_VARS:
                    x = dm[rpe_col].values
                    y = dm[hv].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() < 5 or np.std(y[mask]) == 0:
                        continue
                    r, p = pearsonr(x[mask], y[mask])
                    sig  = ('***' if p<0.001 else '**' if p<0.01
                            else '*' if p<0.05 else 'ns')
                    # Ranges observados
                    hr_v  = dm[hv].dropna()
                    rpe_v = dm[rpe_col].dropna()
                    hr_rng  = (f"{hr_v.min():.0f}–{hr_v.max():.0f}%"
                               if len(hr_v) > 0 else '—')
                    rpe_rng = (f"{rpe_v.min():.1f}–{rpe_v.max():.1f}"
                               if len(rpe_v) > 0 else '—')
                    rows_corr.append({
                        'Modalidade': mod,
                        'HR Zone':    f"{hr_lbl} [{hr_rng}]",
                        'RPE Zone':   f"{rpe_lbl} [{rpe_rng}]",
                        'r':          f"{r:+.3f}",
                        'p':          f"{p:.4f}",
                        'Sig.':       sig,
                        'n':          int(mask.sum()),
                        'Força':      forca_fn(r),
                    })

            if rows_corr:
                st.dataframe(pd.DataFrame(rows_corr), width="stretch", hide_index=True)


def tab_correlacoes(da, dw):
    st.header("🧠 Correlações & Impacto")
    if len(da) == 0 or len(dw) == 0: st.warning("Sem dados suficientes."); return

    rpe_col = next((c for c in ['rpe', 'RPE', 'icu_rpe'] if c in da.columns), None)
    CICLICOS_T = ['Bike', 'Row', 'Run', 'Ski']

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _prep_merged(da_in, excluir_wt=True):
        """Cruza RPE diário com HRV/RHR do dia seguinte."""
        da2 = filtrar_principais(da_in).copy()
        if excluir_wt:
            da2 = da2[da2['type'].apply(norm_tipo) != 'WeightTraining']
        da2['Data'] = pd.to_datetime(da2['Data']).dt.normalize()
        if rpe_col:
            rpe_d = da2.groupby('Data')[rpe_col].mean().reset_index()
            rpe_d.columns = ['Data', 'rpe_avg']
            rpe_d['rpe_cat'] = rpe_d['rpe_avg'].apply(classificar_rpe)
        else:
            return pd.DataFrame()
        dw2 = dw.copy()
        dw2['Data'] = pd.to_datetime(dw2['Data']).dt.normalize()
        cols_dw = ['Data', 'hrv'] + (['rhr'] if 'rhr' in dw2.columns else [])
        dw2_shift = dw2[cols_dw].copy()
        dw2_shift['Data'] = dw2_shift['Data'] - pd.Timedelta(days=1)
        merged = rpe_d.merge(dw2_shift, on='Data', how='inner')
        return merged.dropna(subset=['hrv', 'rpe_cat'])

    def _tabela_delta(merged, grupo_col, baseline_col, grupos, label_col='Categoria'):
        """Gera tabela com Δ HRV% e Δ RHR para cada grupo."""
        rows = []
        base_hrv = merged['hrv'].mean()
        base_rhr = merged['rhr'].mean() if 'rhr' in merged.columns else None
        for g in grupos:
            sub = merged[merged[grupo_col] == g]
            if len(sub) < 3: continue
            d_hrv = (sub['hrv'].mean() - base_hrv) / base_hrv * 100
            row = {
                label_col: g,
                'N':        len(sub),
                'HRV médio': f"{sub['hrv'].mean():.0f} ms",
                'Δ HRV%':   f"{d_hrv:+.1f}%",
                'Δ HRV interp.': ('↗ recuperação' if d_hrv > 3 else
                                   '↘ stress' if d_hrv < -3 else '→ neutro'),
            }
            if base_rhr and 'rhr' in merged.columns:
                d_rhr = sub['rhr'].mean() - base_rhr
                row['RHR médio'] = f"{sub['rhr'].mean():.0f} bpm"
                row['Δ RHR']     = f"{d_rhr:+.1f} bpm"
                row['Δ RHR interp.'] = ('↘ recuperação' if d_rhr < -2 else
                                          '↗ stress' if d_rhr > 2 else '→ neutro')
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 1 — Impacto RPE → HRV/RHR (sem WeightTraining)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("💚 Impacto do RPE no HRV/RHR (dia seguinte)")
    st.caption("Só actividades cíclicas (excluindo WeightTraining). "
               "RPE agrupado em Leve/Moderado/Pesado. HRV/RHR do dia seguinte.")

    merged_rpe = _prep_merged(da, excluir_wt=True)

    if len(merged_rpe) >= 5:
        cats = ['Leve', 'Moderado', 'Pesado']
        base_hrv = merged_rpe['hrv'].mean()

        # Gráfico Plotly — mobile-friendly (barras verticais simples)
        cat_data = {}
        for c in cats:
            sub = merged_rpe[merged_rpe['rpe_cat'] == c]
            if len(sub) >= 2:
                cat_data[c] = {
                    'hrv': sub['hrv'].mean(),
                    'rhr': sub['rhr'].mean() if 'rhr' in sub.columns else None,
                    'n':   len(sub),
                }

        fig_rpe = go.Figure()
        CORES_CAT = {'Leve': '#2ecc71', 'Moderado': '#f39c12', 'Pesado': '#e74c3c'}
        for c in cats:
            if c not in cat_data: continue
            delta = (cat_data[c]['hrv'] - base_hrv) / base_hrv * 100
            fig_rpe.add_trace(go.Bar(
                x=[c], y=[round(delta, 1)],
                name=c,
                marker_color=CORES_CAT[c],
                text=[f"{delta:+.1f}%"],
                textposition='outside',
                width=0.5,
                hovertemplate=f'{c}<br>Δ HRV: <b>{delta:+.1f}%</b><br>N={cat_data[c]["n"]}<extra></extra>'))
        fig_rpe.add_hline(y=0, line_dash='dash', line_color='#888', line_width=1)
        fig_rpe.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            title=dict(text='Δ HRV% no dia seguinte por RPE', font=dict(size=13)),
            height=320, showlegend=False,
            xaxis=dict(title='Categoria RPE', tickfont=dict(size=14)),
            yaxis=dict(title='Δ HRV (%)', showgrid=True, gridcolor='#eee', zeroline=True),
            margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig_rpe, use_container_width=True)

        # Tabela
        df_tab_rpe = _tabela_delta(merged_rpe, 'rpe_cat', 'hrv', cats, 'Categoria RPE')
        if len(df_tab_rpe) > 0:
            st.dataframe(df_tab_rpe, hide_index=True, use_container_width=True)
    else:
        st.info("Dados insuficientes para análise de impacto RPE.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 2 — Scatter RPE→HRV | HRV→RHR
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🔍 Relação: RPE → HRV | HRV → RHR")

    col1, col2 = st.columns(2)

    with col1:
        if len(merged_rpe) >= 5 and rpe_col:
            m_cl = merged_rpe[['rpe_avg','hrv']].dropna()
            if len(m_cl) >= 5:
                r_val, _ = pearsonr(m_cl['rpe_avg'], m_cl['hrv'])
                fig_sc1 = go.Figure()
                fig_sc1.add_trace(go.Scatter(
                    x=m_cl['rpe_avg'], y=m_cl['hrv'],
                    mode='markers',
                    marker=dict(color='#3498db', size=7, opacity=0.5),
                    hovertemplate='RPE: %{x:.1f}<br>HRV: <b>%{y:.0f} ms</b><extra></extra>'))
                xr = np.linspace(m_cl['rpe_avg'].min(), m_cl['rpe_avg'].max(), 50)
                z = np.polyfit(m_cl['rpe_avg'], m_cl['hrv'], 1)
                fig_sc1.add_trace(go.Scatter(
                    x=xr, y=np.poly1d(z)(xr),
                    mode='lines', line=dict(color='#e74c3c', width=2),
                    hoverinfo='skip'))
                fig_sc1.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    title=dict(text=f'RPE → HRV (r={r_val:.2f})', font=dict(size=12)),
                    height=280,
                    xaxis=dict(title='RPE', showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='HRV (ms) dia+1', showgrid=True, gridcolor='#eee'),
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=False)
                st.plotly_chart(fig_sc1, use_container_width=True)

    with col2:
        if 'hrv' in dw.columns and 'rhr' in dw.columns:
            dw3 = dw[['hrv','rhr']].dropna()
            if len(dw3) >= 5:
                r2 = dw3['hrv'].corr(dw3['rhr'])
                fig_sc2 = go.Figure()
                fig_sc2.add_trace(go.Scatter(
                    x=dw3['hrv'], y=dw3['rhr'],
                    mode='markers',
                    marker=dict(color='#9b59b6', size=7, opacity=0.5),
                    hovertemplate='HRV: %{x:.0f}<br>RHR: <b>%{y:.0f} bpm</b><extra></extra>'))
                xr2 = np.linspace(dw3['hrv'].min(), dw3['hrv'].max(), 50)
                z2  = np.polyfit(dw3['hrv'], dw3['rhr'], 1)
                fig_sc2.add_trace(go.Scatter(
                    x=xr2, y=np.poly1d(z2)(xr2),
                    mode='lines', line=dict(color='#e74c3c', width=2),
                    hoverinfo='skip'))
                fig_sc2.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    title=dict(text=f'HRV vs RHR (r={r2:.2f})', font=dict(size=12)),
                    height=280,
                    xaxis=dict(title='HRV (ms)', showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='RHR (bpm)', showgrid=True, gridcolor='#eee'),
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=False)
                st.plotly_chart(fig_sc2, use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 3 — Correlações Wellness (heatmap + tabela)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📊 Correlações entre Métricas Wellness")

    mets_num = [c for c in ['hrv','rhr','sleep_quality','fatiga','stress','humor','soreness']
                if c in dw.columns and dw[c].notna().any()]

    if len(mets_num) >= 3:
        corr_mat = dw[mets_num].corr(method='pearson')

        # Heatmap (mobile-friendly: menor, quadrado)
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig_hm, ax_hm = plt.subplots(figsize=(min(8, len(mets_num)*1.2),
                                               min(7, len(mets_num)*1.0)))
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
        sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f',
                    cmap='RdYlGn', center=0, ax=ax_hm, square=True,
                    linewidths=0.5, vmin=-1, vmax=1,
                    cbar_kws={'shrink':0.8},
                    annot_kws={'size': 9})
        ax_hm.set_title('Correlações Wellness (Pearson)', fontsize=12, fontweight='bold')
        ax_hm.tick_params(axis='both', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_hm)
        plt.close()

        # Tabela de interpretação
        def _forca_str(r):
            a = abs(r)
            if a >= 0.70: return "★★★ Forte"
            if a >= 0.50: return "★★ Moderada"
            if a >= 0.30: return "★ Fraca"
            return "— Nenhuma"

        rows_corr = []
        for i in range(len(mets_num)):
            for j in range(i+1, len(mets_num)):
                r = corr_mat.iloc[i,j]
                if abs(r) < 0.20: continue  # ignorar muito fracas
                rows_corr.append({
                    'Variável A':  mets_num[i],
                    'Variável B':  mets_num[j],
                    'r':           f"{r:+.2f}",
                    'Força':       _forca_str(r),
                    'Direcção':    ('↗ positiva' if r > 0 else '↘ negativa'),
                })
        if rows_corr:
            df_corr_tab = (pd.DataFrame(rows_corr)
                           .sort_values('r', key=lambda s: s.abs(), ascending=False))
            st.dataframe(df_corr_tab, hide_index=True, use_container_width=True)
            st.caption("★★★ Forte |r|≥0.70 | ★★ Moderada |r|≥0.50 | ★ Fraca |r|≥0.30")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECÇÃO 4 — Impacto por Tipo de Actividade → HRV/RHR (dia+1)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🏃 Impacto por Tipo de Actividade → HRV/RHR (dia seguinte)")
    st.caption(
        "Cíclicas: todas as sessões. "
        "WeightTraining: só dias sem outra modalidade cíclica.")

    da3 = filtrar_principais(da).copy()
    da3['Data'] = pd.to_datetime(da3['Data']).dt.normalize()
    da3['_tipo'] = da3['type'].apply(norm_tipo)

    # WeightTraining só se sozinho no dia
    dias_ciclicos = set(da3[da3['_tipo'].isin(CICLICOS_T)]['Data'].dt.normalize())
    da3_filter = da3[
        (da3['_tipo'].isin(CICLICOS_T)) |
        ((da3['_tipo'] == 'WeightTraining') & (~da3['Data'].isin(dias_ciclicos)))
    ].copy()

    tipo_daily = (da3_filter.groupby('Data')['_tipo']
                  .agg(lambda x: x.mode()[0] if len(x) > 0 else None)
                  .reset_index())

    dw2b = dw.copy()
    dw2b['Data'] = pd.to_datetime(dw2b['Data']).dt.normalize()
    cols_dw2 = ['Data','hrv'] + (['rhr'] if 'rhr' in dw2b.columns else [])
    dw2b_s = dw2b[cols_dw2].copy()
    dw2b_s['Data'] = dw2b_s['Data'] - pd.Timedelta(days=1)
    merged2 = tipo_daily.merge(
        dw2b_s.rename(columns={'Data':'Data'}), on='Data', how='inner')
    merged2 = merged2.dropna(subset=['hrv','_tipo'])

    if len(merged2) >= 5:
        tipos_disp = [t for t in CICLICOS_T + ['WeightTraining']
                      if t in merged2['_tipo'].values]
        base_hrv2  = merged2['hrv'].mean()
        base_rhr2  = merged2['rhr'].mean() if 'rhr' in merged2.columns else None

        CORES_T = {'Bike':'#e74c3c','Row':'#3498db','Ski':'#9b59b6',
                   'Run':'#2ecc71','WeightTraining':'#f39c12'}

        # Gráfico barras (mobile-friendly)
        fig_tipo = go.Figure()
        for t in tipos_disp:
            sub = merged2[merged2['_tipo']==t]
            if len(sub) < 3: continue
            delta = (sub['hrv'].mean() - base_hrv2) / base_hrv2 * 100
            fig_tipo.add_trace(go.Bar(
                x=[t], y=[round(delta,1)],
                name=t,
                marker_color=CORES_T.get(t,'#888'),
                text=[f"{delta:+.1f}%"],
                textposition='outside',
                width=0.5,
                hovertemplate=f'{t}<br>Δ HRV: <b>{delta:+.1f}%</b><br>N={len(sub)}<extra></extra>'))
        fig_tipo.add_hline(y=0, line_dash='dash', line_color='#888', line_width=1)
        fig_tipo.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            title=dict(text='Δ HRV% no dia seguinte por Modalidade', font=dict(size=13)),
            height=320, showlegend=False,
            xaxis=dict(title='Modalidade', tickfont=dict(size=13)),
            yaxis=dict(title='Δ HRV (%)', showgrid=True, gridcolor='#eee', zeroline=True),
            margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig_tipo, use_container_width=True)

        # Tabela igual à do RPE
        df_tab_tipo = _tabela_delta(merged2, '_tipo', 'hrv', tipos_disp, 'Modalidade')
        if len(df_tab_tipo) > 0:
            st.dataframe(df_tab_tipo, hide_index=True, use_container_width=True)
    else:
        st.info("Dados insuficientes para análise por tipo de actividade.")


def tab_recovery(dw):
    st.header("🔋 Recovery Score & HRV Analysis")
    if len(dw) == 0 or 'hrv' not in dw.columns: st.warning("Sem dados de wellness/HRV."); return
    rec = calcular_recovery(dw)
    if len(rec) == 0: return
    u = rec.iloc[-1]; score = u['recovery_score']
    cat = ('🟢 Excelente' if score >= 80 else '🟡 Bom' if score >= 60 else '🟠 Moderado' if score >= 40 else '🔴 Baixo')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recovery Score", f"{score:.0f}/100", delta=cat)
    c2.metric("HRV atual", f"{u['hrv']:.0f} ms" if pd.notna(u['hrv']) else "—")
    c3.metric("Baseline HRV", f"{u['hrv_baseline']:.0f} ms" if pd.notna(u['hrv_baseline']) else "—")
    c4.metric("CV% 7d", f"{u['hrv_cv7']:.1f}%" if pd.notna(u['hrv_cv7']) else "—")
    st.markdown("---")
    n_dias = st.slider("Dias a mostrar", 14, min(len(rec), 365), min(90, len(rec)))
    df_tl = rec.tail(n_dias).copy()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhspan(80, 100, alpha=0.15, color=CORES['verde'], label='Excelente (80–100)')
    ax.axhspan(60, 80, alpha=0.15, color=CORES['amarelo'], label='Bom (60–79)')
    ax.axhspan(40, 60, alpha=0.15, color=CORES['laranja'], label='Moderado (40–59)')
    ax.axhspan(0, 40, alpha=0.15, color=CORES['vermelho'], label='Baixo (0–39)')
    x = range(len(df_tl)); sc = df_tl['recovery_score'].values
    cpts = [CORES['verde'] if s >= 80 else CORES['amarelo'] if s >= 60 else CORES['laranja'] if s >= 40 else CORES['vermelho'] for s in sc]
    ax.plot(x, sc, color=CORES['azul_escuro'], linewidth=2, alpha=0.7)
    ax.scatter(x, sc, c=cpts, s=70, edgecolors='white', linewidths=2, zorder=5)
    if len(df_tl) >= 7: ax.plot(x, pd.Series(sc).rolling(7, min_periods=3).mean(), color=CORES['roxo'], linewidth=2.5, linestyle='--', label='Média 7d', alpha=0.8)
    datas = df_tl['Data'].dt.strftime('%d/%m'); step = max(1, len(x) // 10)
    ax.set_xticks(list(x)[::step]); ax.set_xticklabels([datas.iloc[i] for i in range(0, len(datas), step)], rotation=45)
    ax.set_ylim(0, 105); ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('Recovery Score — Timeline', fontsize=14, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📊 HRV com Normal Range")
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1.2])
    xr = range(len(df_tl)); dr = df_tl['Data'].dt.strftime('%d/%m'); sr = max(1, len(xr) // 10)
    if df_tl['normal_range_inf'].notna().any():
        ax1.fill_between(xr, df_tl['normal_range_inf'], df_tl['normal_range_sup'], alpha=0.25, color=CORES['azul'], label='Normal Range HRV')
    if df_tl['hrv_baseline'].notna().any():
        ax1.plot(xr, df_tl['hrv_baseline'], color=CORES['roxo'], linestyle='--', linewidth=2, label='Baseline 14d')
    hv = df_tl['hrv'].values
    chr_list = [CORES['verde'] if (not pd.isna(df_tl['normal_range_sup'].iloc[i]) and not pd.isna(hv[i]) and hv[i] > df_tl['normal_range_sup'].iloc[i])
                else CORES['vermelho'] if (not pd.isna(df_tl['normal_range_inf'].iloc[i]) and not pd.isna(hv[i]) and hv[i] < df_tl['normal_range_inf'].iloc[i])
                else CORES['azul'] for i in range(len(df_tl))]
    ax1.plot(xr, hv, color=CORES['preto'], linewidth=2, alpha=0.6)
    ax1.scatter(xr, hv, c=chr_list, s=70, edgecolors='white', linewidths=2, zorder=5)
    ax1.legend(loc='upper right', fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_ylabel('HRV (ms)', fontweight='bold')
    ax1.set_title('HRV com Normal Range (HRV4Training)', fontsize=13, fontweight='bold')
    cv_s = df_tl['hrv_cv7'].copy()
    if cv_s.notna().sum() >= 5:
        cv_b = cv_s.rolling(14, min_periods=5).mean(); cv_sd = cv_s.rolling(14, min_periods=5).std()
        cv_inf = cv_b - 0.5 * cv_sd; cv_sup = cv_b + 0.5 * cv_sd
        ax2.fill_between(xr, cv_inf, cv_sup, alpha=0.25, color=CORES['laranja'], label='Normal Range CV%')
        ax2.plot(xr, cv_b, color=CORES['roxo'], linestyle='--', linewidth=1.8, alpha=0.8)
        cv_v = cv_s.values
        ccv = [CORES['verde'] if (not pd.isna(cv_sup.iloc[i]) and not pd.isna(cv_v[i]) and cv_v[i] > cv_sup.iloc[i])
               else CORES['vermelho'] if (not pd.isna(cv_inf.iloc[i]) and not pd.isna(cv_v[i]) and cv_v[i] < cv_inf.iloc[i])
               else CORES['azul'] for i in range(len(df_tl))]
        ax2.plot(xr, cv_v, color=CORES['preto'], linewidth=1.8, alpha=0.6)
        ax2.scatter(xr, cv_v, c=ccv, s=55, edgecolors='white', linewidths=1.5, zorder=5)
        ax2.axhline(3, color=CORES['cinza'], linestyle=':', alpha=0.5); ax2.axhline(10, color=CORES['cinza'], linestyle=':', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=8)
    for axr in [ax1, ax2]:
        axr.set_xticks(list(xr)[::sr]); axr.set_xticklabels([dr.iloc[i] for i in range(0, len(dr), sr)], rotation=45); axr.grid(True, alpha=0.3)
    ax2.set_ylabel('CV% HRV', fontweight='bold'); ax2.set_title('CV% com Normal Range', fontsize=12, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.subheader("📊 BPE — Z-Score Semanal (Método SWC)")
    mets_bpe = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress'] if m in dw.columns and dw[m].notna().any()]
    n_semanas_disp = max(1, len(dw) // 7)
    _skip_bpe = n_semanas_disp < 4
    if _skip_bpe:
        st.info(f"Dados insuficientes para BPE (min 4 semanas, disponivel: {n_semanas_disp}).")
        n_sem = n_semanas_disp
    else:
        _slider_max = min(52, n_semanas_disp)
        _slider_val = min(16, _slider_max)
        if _slider_max > 4:
            n_sem = st.slider("Semanas (BPE)", 4, _slider_max, _slider_val)
        else:
            n_sem = _slider_max
            st.caption(f"BPE: {n_sem} semanas disponíveis")
    dados_bpe = {}
    if not _skip_bpe:
        for met in mets_bpe:
            s = calcular_bpe(dw, met, 60)
            if len(s) > 0: dados_bpe[met] = s.tail(n_sem)
    if dados_bpe:
        semanas = list(dados_bpe[list(dados_bpe.keys())[0]]['ano_semana'])
        nm = len(dados_bpe); mat = np.zeros((nm, len(semanas)))
        for i, met in enumerate(dados_bpe.keys()):
            z = dados_bpe[met]['zscore'].values; mat[i, :len(z)] = (-z if met == 'rhr' else z)[:len(semanas)]
        cmap = LinearSegmentedColormap.from_list('bpe', [CORES['vermelho'], CORES['amarelo'], CORES['verde']], N=100)
        fig, ax = plt.subplots(figsize=(max(14, len(semanas) * 0.9), max(5, nm * 1.1)))
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=-2, vmax=2)
        nomes = {'hrv': 'HRV', 'rhr': 'RHR (inv)', 'sleep_quality': 'Sono', 'fatiga': 'Energia', 'stress': 'Relaxamento'}
        ax.set_yticks(range(nm)); ax.set_yticklabels([nomes.get(m, m) for m in dados_bpe.keys()], fontsize=11)
        slbls = [s.split('-W')[1] if '-W' in s else s for s in semanas]
        ax.set_xticks(range(len(semanas))); ax.set_xticklabels([f'S{s}' for s in slbls], rotation=45, fontsize=9)
        for i in range(nm):
            for j in range(len(semanas)):
                v = mat[i, j]; tc = 'white' if abs(v) > 1 else 'black'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('Z-Score BPE (múltiplos de SWC)')
        cbar.set_ticks([-2, -1, 0, 1, 2]); cbar.set_ticklabels(['🔴 -2', '🟠 -1', '0', '🟡 +1', '🟢 +2'])
        ax.set_title('BPE — Blocos de Padrão Específico (Z-Score com SWC)', fontsize=13, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("🏋️ HRV-Guided Training (LnrMSSD)")
    if len(dw) >= 14 and 'hrv' in dw.columns and dw['hrv'].notna().sum() >= 14:
        df_hg = dw.copy().sort_values('Data'); df_hg['Data'] = pd.to_datetime(df_hg['Data'])
        df_hg['LnrMSSD'] = np.where(df_hg['hrv'] > 0, np.log(df_hg['hrv']), np.nan); df_hg = df_hg.dropna(subset=['LnrMSSD'])
        dias_fam = st.slider("Dias baseline rolling", 7, 28, 14)
        df_hg['bm'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).mean()
        df_hg['bs'] = df_hg['LnrMSSD'].rolling(dias_fam, min_periods=dias_fam).std()
        df_hg['linf'] = df_hg['bm'] - 0.5 * df_hg['bs']; df_hg['lsup'] = df_hg['bm'] + 0.5 * df_hg['bs']
        df_hg['desvio'] = (df_hg['LnrMSSD'] - df_hg['bm']) / df_hg['bs']
        df_hg['intens'] = df_hg.apply(lambda r: 'HIIT' if pd.notna(r['bm']) and r['linf'] <= r['LnrMSSD'] <= r['lsup'] else ('Recuperação' if pd.notna(r['bm']) else 'Sem dados'), axis=1)
        n_hg = st.slider("Dias HRV-Guided", 14, min(len(df_hg), 180), min(60, len(df_hg)))
        df_p = df_hg.tail(n_hg).copy(); xh = range(len(df_p)); dh = df_p['Data'].dt.strftime('%d/%m'); sh = max(1, len(xh) // 15)
        ch = [CORES['verde'] if i == 'HIIT' else CORES['laranja'] if i == 'Recuperação' else CORES['cinza'] for i in df_p['intens']]
        fig3, (ah1, ah2) = plt.subplots(2, 1, figsize=(15, 10))
        ah1.plot(xh, df_p['LnrMSSD'], '-', alpha=0.6, color=CORES['azul'], linewidth=2, label='LnrMSSD')
        ah1.scatter(xh, df_p['LnrMSSD'], c=ch, s=80, edgecolors='white', linewidths=2, zorder=5)
        ah1.plot(xh, df_p['bm'], color=CORES['verde_escuro'], linestyle='--', linewidth=2.5, label=f'Baseline ({dias_fam}d)', alpha=0.8)
        ah1.plot(xh, df_p['lsup'], color=CORES['laranja'], linestyle=':', linewidth=2, label='±0.5 DP', alpha=0.7)
        ah1.plot(xh, df_p['linf'], color=CORES['laranja'], linestyle=':', linewidth=2, alpha=0.7)
        ah1.fill_between(xh, df_p['linf'], df_p['lsup'], alpha=0.15, color=CORES['verde'], label='Zona HIIT')
        ah1.legend(loc='best', fontsize=9); ah1.grid(True, alpha=0.3); ah1.set_ylabel('LnrMSSD', fontweight='bold')
        ah1.set_title(f'HRV-Guided Training — Baseline Rolling ({dias_fam}d)', fontsize=13, fontweight='bold')
        ah1.set_xticks(list(xh)[::sh]); ah1.set_xticklabels([dh.iloc[i] for i in range(0, len(dh), sh)], rotation=45)
        ah2.axhline(0, color=CORES['verde_escuro'], linestyle='-', linewidth=2, alpha=0.8)
        ah2.axhline(0.5, color=CORES['laranja'], linestyle=':', linewidth=2); ah2.axhline(-0.5, color=CORES['laranja'], linestyle=':', linewidth=2)
        ah2.fill_between(xh, -0.5, 0.5, alpha=0.2, color=CORES['verde'], label='Zona HIIT')
        ah2.scatter(xh, df_p['desvio'], c=ch, alpha=0.7, s=60, edgecolors='white', linewidths=1)
        ah2.plot(xh, df_p['desvio'], color=CORES['cinza'], alpha=0.4, linewidth=1)
        ah2.set_ylim(-3, 3); ah2.legend(loc='best', fontsize=9); ah2.grid(True, alpha=0.3)
        ah2.set_ylabel('Desvio (DP)', fontweight='bold'); ah2.set_title('Desvio LnrMSSD do Baseline', fontsize=12, fontweight='bold')
        ah2.set_xticks(list(xh)[::sh]); ah2.set_xticklabels([dh.iloc[i] for i in range(0, len(dh), sh)], rotation=45)
        plt.tight_layout(); st.pyplot(fig3); plt.close()
        df_val = df_hg[df_hg['bm'].notna()]
        if len(df_val) > 0:
            hiit_n = (df_val['intens'] == 'HIIT').sum(); rec_n = (df_val['intens'] == 'Recuperação').sum(); total_n = len(df_val)
            c1, c2, c3 = st.columns(3)
            c1.metric("Dias HIIT", f"{hiit_n} ({hiit_n/total_n*100:.0f}%)")
            c2.metric("Dias Recuperação", f"{rec_n} ({rec_n/total_n*100:.0f}%)")
            c3.metric("Prescrição HOJE", '✅ HIIT' if df_val.iloc[-1]['intens'] == 'HIIT' else '🟠 Recuperação')

# ════════════════════════════════════════════════════════════════════════════════
# TAB 8 — WELLNESS
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_wellness.py
# ════════════════════════════════════════════════════════════════════════════

def tab_wellness(dw):
    st.header("🧘 Wellness")
    if len(dw) == 0: st.warning("Sem dados de wellness."); return
    mets = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness'] if m in dw.columns and dw[m].notna().any()]
    if not mets: st.warning("Sem métricas wellness."); return
    sel = st.multiselect("Métricas", mets, default=mets[:5])
    if not sel: return
    fig, axes = plt.subplots(len(sel), 1, figsize=(14, 3 * len(sel)), sharex=True)
    if len(sel) == 1: axes = [axes]
    x = range(len(dw)); datas = pd.to_datetime(dw['Data']).dt.strftime('%d/%m')
    CM = {'hrv': CORES['verde'], 'rhr': CORES['vermelho'], 'sleep_quality': CORES['roxo'],
          'fatiga': CORES['laranja'], 'stress': CORES['vermelho_escuro'], 'humor': CORES['verde_escuro'], 'soreness': CORES['azul']}
    for ax, met in zip(axes, sel):
        v = pd.to_numeric(dw[met], errors='coerce').values
        ax.plot(x, v, color=CM.get(met, CORES['azul']), linewidth=2, marker='o', markersize=4)
        if len(v) >= 7: ax.plot(x, pd.Series(v).rolling(7, min_periods=3).mean(), color=CORES['preto'], linewidth=1.5, linestyle='--', alpha=0.5, label='Média 7d')
        ax.set_ylabel(met.replace('_', ' ').title(), fontweight='bold'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    step = max(1, len(x) // 12); axes[-1].set_xticks(list(x)[::step])
    axes[-1].set_xticklabels([datas.iloc[i] for i in range(0, len(datas), step)], rotation=45)
    plt.suptitle('Métricas Wellness', fontsize=14, fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()
    st.subheader("📋 Resumo (últimos 7 dias)")
    if len(dw) >= 7:
        u7 = dw.tail(7); rows = []
        for m in mets:
            col = pd.to_numeric(u7[m], errors='coerce')
            rows.append({'Métrica': m.replace('_', ' ').title(), 'Média': f"{col.mean():.1f}", 'Mín': f"{col.min():.0f}", 'Máx': f"{col.max():.0f}"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_analises.py
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES — ANÁLISES AVANÇADAS (do original Python/SQLite)
# ════════════════════════════════════════════════════════════════════════════════

def calcular_polinomios_carga(df_act_full):
    """
    Polynomial fit CTL/ATL Overall e por modalidade — igual ao original.
    Usa session_rpe para construir série CTL/ATL.
    """
    df = filtrar_principais(df_act_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    if 'moving_time' not in df.columns or 'rpe' not in df.columns:
        return None
    df['rpe_fill'] = df['rpe'].fillna(df['rpe'].median())
    df['load_val'] = (df['moving_time'] / 60) * df['rpe_fill']
    ld = df.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
    idx = pd.date_range(ld['Data'].min(), datetime.now().date())
    ld = ld.set_index('Data').reindex(idx, fill_value=0).reset_index(); ld.columns = ['Data', 'load_val']
    ld['CTL'] = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL'] = ld['load_val'].ewm(span=7, adjust=False).mean()
    ld['dias_num'] = (ld['Data'] - ld['Data'].min()).dt.days.values

    resultados = {'overall': {'CTL': {}, 'ATL': {}}}
    for metrica in ['CTL', 'ATL']:
        x, y = ld['dias_num'].values, ld[metrica].values
        for grau in [2, 3]:
            try:
                z = np.polyfit(x, y, grau)
                p = np.poly1d(z)
                r2 = np.corrcoef(y, p(x))[0, 1] ** 2
                resultados['overall'][metrica][f'grau{grau}'] = {'coef': z, 'poly': p, 'r2': r2, 'x': x, 'y': y}
            except Exception:
                pass

    # Por modalidade
    for tipo in ['Bike', 'Run', 'Row', 'Ski']:
        df_tipo = df[df['type'] == tipo].copy()
        if len(df_tipo) < 5:
            continue
        ld_t = df_tipo.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
        idx_t = pd.date_range(ld_t['Data'].min(), ld['Data'].max())
        ld_t = ld_t.set_index('Data').reindex(idx_t, fill_value=0).reset_index(); ld_t.columns = ['Data', 'load_val']
        ld_t['CTL'] = ld_t['load_val'].ewm(span=42, adjust=False).mean()
        ld_t['ATL'] = ld_t['load_val'].ewm(span=7, adjust=False).mean()
        ld_t['dias_num'] = (ld_t['Data'] - ld_t['Data'].min()).dt.days.values
        resultados[f'tipo_{tipo}'] = {'CTL': {}, 'ATL': {}}
        for metrica in ['CTL', 'ATL']:
            x, y = ld_t['dias_num'].values, ld_t[metrica].values
            for grau in [2, 3]:
                try:
                    z = np.polyfit(x, y, grau)
                    p = np.poly1d(z)
                    r2 = np.corrcoef(y, p(x))[0, 1] ** 2
                    resultados[f'tipo_{tipo}'][metrica][f'grau{grau}'] = {'coef': z, 'poly': p, 'r2': r2, 'x': x, 'y': y}
                except Exception:
                    pass
    resultados['_ld'] = ld
    return resultados

def tabela_resumo_por_tipo_df(da):
    """Tabela resumo por tipo igual ao original."""
    df = filtrar_principais(da).copy()
    if len(df) == 0:
        return pd.DataFrame()
    df['type'] = df['type'].apply(norm_tipo)
    agg = {'Data': 'count'}
    if 'moving_time' in df.columns:
        df['horas'] = pd.to_numeric(df['moving_time'], errors='coerce') / 3600
        agg['horas'] = 'sum'
    if 'power_avg' in df.columns:
        df['power_avg'] = pd.to_numeric(df['power_avg'], errors='coerce')
        agg['power_avg'] = 'mean'
    if 'rpe' in df.columns:
        df['rpe'] = pd.to_numeric(df['rpe'], errors='coerce')
        agg['rpe'] = 'mean'
    resumo = df.groupby('type').agg(agg).round(1).reset_index()
    resumo.columns = ['Modalidade'] + [c.replace('Data', 'Sessões').replace('horas', 'Horas').replace('power_avg', 'Power (W)').replace('rpe', 'RPE') for c in resumo.columns[1:]]
    return resumo

def tabela_ranking_power_df(da, n=10):
    """Top N por power_avg."""
    df = filtrar_principais(da).copy()
    if len(df) == 0 or 'power_avg' not in df.columns:
        return pd.DataFrame()
    df['power_avg'] = pd.to_numeric(df['power_avg'], errors='coerce')
    df = df.dropna(subset=['power_avg'])
    if len(df) == 0:
        return pd.DataFrame()
    df['Data'] = pd.to_datetime(df['Data']).dt.strftime('%Y-%m-%d')
    cols = [c for c in ['Data', 'type', 'name', 'power_avg', 'rpe', 'moving_time'] if c in df.columns]
    top = df.nlargest(n, 'power_avg')[cols].copy()
    if 'moving_time' in top.columns:
        top['moving_time'] = (pd.to_numeric(top['moving_time'], errors='coerce') / 3600).round(1)
        top.rename(columns={'moving_time': 'Horas'}, inplace=True)
    top.rename(columns={'power_avg': 'Power (W)', 'type': 'Tipo', 'name': 'Nome', 'rpe': 'RPE'}, inplace=True)
    return top.reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 9 — ANÁLISES AVANÇADAS
# ════════════════════════════════════════════════════════════════════════════════
def tab_analises(da_full, dw, dfs_annual=None, df_annual=None):
    """
    Aba de Análises Avançadas — equivalente completo ao código Python original.
    Inclui: tabelas, training load, polynomial, BPE, falta de estímulo,
    Annual (aquecimentos), e TODAS as saídas escritas por modalidade (6 passos).
    """
    st.header("🔬 Análises Avançadas")
    if dfs_annual is None: dfs_annual = {}
    if df_annual is None:  df_annual  = pd.DataFrame()

    if len(da_full) == 0:
        st.warning("Sem dados de atividades para análise avançada.")
        return

    # ── Secção 1: Tabelas de Resumo ─────────────────────────────────────────
    st.subheader("📋 Resumo de Atividades por Modalidade")
    df_res = tabela_resumo_por_tipo_df(da_full)
    if len(df_res) > 0:
        st.dataframe(df_res, width="stretch", hide_index=True)

    st.subheader("🏆 Top 10 Sessões por Potência Média")
    df_rank = tabela_ranking_power_df(da_full, n=10)
    if len(df_rank) > 0:
        st.dataframe(df_rank, width="stretch", hide_index=True)

    st.markdown("---")

    # ── Secção 2: Training Load Mensal Stacked ──────────────────────────────
    st.subheader("📊 Training Load Mensal por Modalidade (TRIMP = min × RPE)")
    df_tl = filtrar_principais(da_full).copy()
    df_tl = add_tempo(df_tl)
    if 'moving_time' in df_tl.columns and 'rpe' in df_tl.columns:
        df_tl['rpe_fill']    = df_tl['rpe'].fillna(df_tl['rpe'].median())
        df_tl['session_rpe'] = (pd.to_numeric(df_tl['moving_time'], errors='coerce') / 60) * df_tl['rpe_fill']
        df_tl = df_tl[df_tl['type'].isin(['Bike', 'Run', 'Row', 'Ski', 'WeightTraining'])]
        pivot_tl = df_tl.pivot_table(index='mes', columns='type', values='session_rpe', aggfunc='sum', fill_value=0).sort_index()
        CORES_MOD = {'Bike': CORES['vermelho'], 'Run': CORES['verde'], 'Row': CORES['azul'], 'Ski': CORES['roxo'], 'WeightTraining': CORES['laranja']}
        fig, ax = plt.subplots(figsize=(16, 6))
        bottom = np.zeros(len(pivot_tl))
        for tipo in [t for t in ['Bike', 'Row', 'Ski', 'Run', 'WeightTraining'] if t in pivot_tl.columns]:
            vals = pivot_tl[tipo].values
            ax.bar(range(len(pivot_tl)), vals, bottom=bottom, label=tipo,
                   color=CORES_MOD.get(tipo, '#888'), alpha=0.85, edgecolor='white', linewidth=0.5)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > 50:
                    ax.text(i, b + v/2, f'{v:.0f}', ha='center', va='center', fontsize=7, fontweight='bold', color='white')
            bottom += vals
        totais = pivot_tl.sum(axis=1).values
        for i, t in enumerate(totais):
            if t > 0: ax.text(i, t + 5, f'{t:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(pivot_tl)))
        ax.set_xticklabels(pivot_tl.index, rotation=45, ha='right')
        ax.axhline(totais.mean(), color='black', linestyle='--', alpha=0.5, label=f'Média: {totais.mean():.0f}')
        ax.set_ylabel('Training Load (TRIMP)'); ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3, axis='y')
        ax.set_title('Training Load Mensal por Modalidade', fontsize=13, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Secção 3: Polynomial CTL/ATL ────────────────────────────────────────
    st.subheader("📈 CTL/ATL — Polynomial Fit (Overall e por Modalidade)")
    with st.spinner("Calculando polynomial fits..."):
        poli = calcular_polinomios_carga(da_full)

    if poli is None:
        st.warning("Sem dados suficientes para polynomial analysis.")
    else:
        # Overall
        st.markdown("**Overall CTL vs ATL**")
        fig, ax = plt.subplots(figsize=(16, 7))
        for metrica, (cor_s, cor_l, sty) in [
            ('CTL', (CORES['azul'],    CORES['azul_escuro'],    '-')),
            ('ATL', (CORES['vermelho'],CORES['vermelho_escuro'],'--'))]:
            if metrica not in poli.get('overall', {}): continue
            dm = poli['overall'][metrica]
            gk = 'grau3' if 'grau3' in dm else 'grau2'
            if gk not in dm: continue
            d = dm[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
            xs = np.linspace(x.min(), x.max(), 200)
            ax.scatter(x, y, alpha=0.3, s=40, color=cor_s, edgecolors='white', linewidths=1, label=f'{metrica} dados')
            ax.plot(xs, poly(xs), linewidth=3, color=cor_l, linestyle=sty,
                    label=f'{metrica} Poly{gk.replace("grau","")} (R²={r2:.3f})')
        ax.set_xlabel('Dias'); ax.set_ylabel('Carga (TRIMP)')
        ax.set_title('CTL vs ATL Overall — Polynomial Fit', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Por modalidade — separado e combinado
        tipos_poli = {k.replace('tipo_', ''): k for k in poli if k.startswith('tipo_')}
        if tipos_poli:
            st.markdown("**Por Modalidade**")
            n_t = len(tipos_poli); ncols = 2; nrows = (n_t + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6*nrows))
            axes_flat = axes.flatten() if n_t > 1 else [axes] if ncols*nrows == 1 else axes.flatten()
            for idx, (tipo_n, tipo_k) in enumerate(sorted(tipos_poli.items())):
                ax = axes_flat[idx]
                for metrica, cor, sty in [('CTL', CORES['azul'], '-'), ('ATL', CORES['vermelho'], '--')]:
                    if metrica not in poli[tipo_k]: continue
                    dm = poli[tipo_k][metrica]
                    gk = 'grau3' if 'grau3' in dm else 'grau2'
                    if gk not in dm: continue
                    d = dm[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
                    xs = np.linspace(x.min(), x.max(), 150)
                    ax.scatter(x, y, alpha=0.35, s=40, color=cor, edgecolors='white', linewidths=1)
                    ax.plot(xs, poly(xs), linewidth=2.5, color=cor, linestyle=sty,
                            label=f'{metrica} R²={r2:.3f}')
                ax.set_title(f'{tipo_n} — CTL/ATL Polynomial', fontsize=11, fontweight='bold')
                ax.set_xlabel('Dias'); ax.set_ylabel('Carga'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            for idx in range(n_t, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            plt.suptitle('CTL/ATL por Modalidade — Polynomial Fit', fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Secção 4: BPE Heatmap ───────────────────────────────────────────────
    st.subheader("🗓️ BPE — Mapa de Estados Semanal")
    if len(dw) >= 14:
        mets_bpe = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness']
                    if m in dw.columns and dw[m].notna().sum() >= 14]
        n_sem_max = max(4, len(dw) // 7)
        n_sem_bpe = st.slider("Semanas BPE", 4, min(52, n_sem_max), min(16, n_sem_max), key="bpe_an")
        dados_bpe = {m: calcular_bpe(dw, m, 60).tail(n_sem_bpe) for m in mets_bpe}
        dados_bpe = {k: v for k, v in dados_bpe.items() if len(v) > 0}
        if dados_bpe:
            semanas = list(dados_bpe[list(dados_bpe.keys())[0]]['ano_semana'])
            nm = len(dados_bpe)
            mat = np.zeros((nm, len(semanas)))
            nomes_bpe = {'hrv': 'HRV', 'rhr': 'RHR (inv)', 'sleep_quality': 'Sono',
                         'fatiga': 'Energia', 'stress': 'Relaxamento', 'humor': 'Humor', 'soreness': 'Sem Dor'}
            for i, met in enumerate(dados_bpe):
                z = dados_bpe[met]['zscore'].values
                mat[i, :len(z)] = (-z if met == 'rhr' else z)[:len(semanas)]
            from matplotlib.colors import LinearSegmentedColormap as _LSC
            cmap_bpe = _LSC.from_list('bpe', [CORES['vermelho'], CORES['amarelo'], CORES['verde']], N=100)
            fig, ax = plt.subplots(figsize=(max(14, len(semanas)*0.9), max(5, nm*1.2)))
            im = ax.imshow(mat, cmap=cmap_bpe, aspect='auto', vmin=-2, vmax=2)
            ax.set_yticks(range(nm))
            ax.set_yticklabels([nomes_bpe.get(m, m) for m in dados_bpe], fontsize=11)
            sem_labels = [str(s).split('-W')[1] if '-W' in str(s) else str(s) for s in semanas]
            ax.set_xticks(range(len(semanas)))
            ax.set_xticklabels([f'S{s}' for s in sem_labels], rotation=45, fontsize=10)
            for i in range(nm):
                for j in range(len(semanas)):
                    v = mat[i, j]; cor_t = 'white' if abs(v) > 1 else 'black'
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, fontweight='bold', color=cor_t)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Z-Score BPE (múltiplos de SWC)')
            cbar.set_ticks([-2,-1,0,1,2]); cbar.set_ticklabels(['🔴-2','-1','0','+1','🟢+2'])
            ax.set_title('BPE — Blocos de Padrão Específico (Z-Score com SWC)', fontsize=13, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Mínimo 14 dias de wellness para BPE.")

    st.markdown("---")

    # ── Secção 5: Falta de Estímulo ─────────────────────────────────────────
    st.subheader("🎯 Análise de Falta de Estímulo por Modalidade")
    st.caption(
        "Need Score v4 — A=Share(25%) B=Quality(25%) C=Load(20%) D=FTLM(20%) E=A×B(10%) | "
        "Need_volume=A+C | Need_intensity=B+D | Overload por score acumulado (≥2/3)")

    # ── Controlos de prioridade e preset K ───────────────────────────────────
    st.markdown("**⚙️ Configuração de Prioridades**")
    col_k, col_sp = st.columns([1, 3])
    with col_k:
        preset_k = st.selectbox("Preset influência",
            ["Conservador (K=6)", "Balanceado (K=10)", "Agressivo (K=15)"],
            index=1, key="prio_preset")
        K = {"Conservador (K=6)": 6, "Balanceado (K=10)": 10,
             "Agressivo (K=15)": 15}[preset_k]
        st.caption(f"K={K} — quanto a preferência influencia a ordenação")

    with col_sp:
        _mods_disp = ['Bike', 'Row', 'Ski', 'Run']
        pc1, pc2, pc3, pc4 = st.columns(4)
        prio_1 = pc1.selectbox("🥇 Prioridade 1 (Foco)", _mods_disp, index=0, key="prio1")
        prio_2 = pc2.selectbox("🥈 Prioridade 2 (Foco)", _mods_disp, index=1, key="prio2")
        prio_3 = pc3.selectbox("🥉 Prioridade 3 (Manutenção)", _mods_disp, index=2, key="prio3")
        prio_4 = pc4.selectbox("4️⃣  Prioridade 4 (Manutenção)", _mods_disp, index=3, key="prio4")

    prio_rank = {prio_1: 1, prio_2: 2, prio_3: 3, prio_4: 4}
    max_rank  = 4
    grupo_foco = {prio_1, prio_2}
    grupo_man  = {prio_3, prio_4}

    st.markdown("---")
    c1, c2 = st.columns(2)
    for col_w, janela, label in [(c1, 7, "7 dias"), (c2, 14, "14 dias")]:
        res, df_debug = analisar_falta_estimulo(da_full, janela_dias=janela)
        with col_w:
            st.markdown(f"**📅 Janela {label}**")
            if not res:
                st.info("Dados insuficientes.")
                continue

            # ── Calcular Need_final com bónus de prioridade ────────────────
            rows_foco = []
            rows_man  = []
            for mod, d in res.items():
                rank  = prio_rank.get(mod, 4)
                peso  = (max_rank + 1 - rank) / max_rank
                bonus = peso * K * (1 - d['need_score'] / 100)

                need_final = d['need_score'] + bonus

                # Overload: Need_final × 0.5
                ol_flag = ""
                intens  = "NORMAL"
                if d['overload']:
                    need_final *= 0.5
                    ol_flag = " ⚠️"
                    intens  = "LOW — reduzir intensidade"

                # Cap manutenção: nunca passa de 40
                if mod in grupo_man:
                    need_final = min(need_final, 40)

                # Piso mínimo
                need_final = max(need_final, 10)

                # Prioridade final
                pf = ('ALTA' if need_final >= 70 else
                      'MÉDIA' if need_final >= 40 else 'BAIXA')

                row_d = {
                    'Modalidade':   f"{'🎯' if mod in grupo_foco else '🔧'} {mod}{ol_flag}",
                    'Need base':    f"{d['need_score']:.1f}",
                    'Bónus prio':   f"+{bonus:.1f}",
                    'Need final':   f"{need_final:.1f}",
                    'Prioridade':   pf,
                    'Vol / Int':    f"{d['need_vol']:.0f} / {d['need_int_prescr']:.0f}",
                    'Prescrição':   d.get('prescricao','—'),
                }
                if mod in grupo_foco:
                    rows_foco.append((need_final, row_d))
                else:
                    rows_man.append((need_final, row_d))

            # Ordenar cada grupo por Need_final
            rows_foco.sort(key=lambda x: x[0], reverse=True)
            rows_man.sort( key=lambda x: x[0], reverse=True)

            # Mostrar separado por grupo
            if rows_foco:
                st.markdown("🎯 **Foco**")
                st.dataframe(pd.DataFrame([r for _, r in rows_foco]),
                             width="stretch", hide_index=True)
            if rows_man:
                st.markdown("🔧 **Manutenção**")
                st.dataframe(pd.DataFrame([r for _, r in rows_man]),
                             width="stretch", hide_index=True)

            # Recomendação: top do grupo foco
            if rows_foco:
                top_mod  = rows_foco[0][1]['Modalidade'].replace('🎯 ','').replace('🔧 ','').replace(' ⚠️','')
                top_need = rows_foco[0][0]
                top_ol   = res.get(top_mod, {}).get('overload', False)
                top_d = res.get(top_mod, {})
                top_prescr = top_d.get('prescricao', '—')
                if top_ol:
                    st.warning(f"⚠️ **{top_mod}**: overload — {top_prescr}")
                else:
                    st.info(f"🎯 **{top_mod}** (score {top_need:.1f}) — {top_prescr}")

            # Debug expandível
            if df_debug is not None and len(df_debug) > 0:
                with st.expander(f"🔬 Debug componentes — {label}"):
                    st.dataframe(df_debug, width="stretch", hide_index=True)
                    csv_b = df_debug.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"⬇️ Download debug CSV ({label})",
                        data=csv_b,
                        file_name=f"need_score_debug_{label.replace(' ','_')}.csv",
                        mime="text/csv",
                        key=f"dl_debug_{janela}")

    st.markdown("---")
    st.markdown("---")


    st.markdown("---")

    # ── Secção 7: SAÍDAS ESCRITAS POR MODALIDADE (igual ao Python original) ─
    st.subheader("📝 Análise Avançada por Modalidade (6 Passos)")
    st.caption("Equivalente às saídas print() do código Python original — CV, Tendências, Correlações, Sazonalidade, RPE, Metas")

    df_full = filtrar_principais(da_full).copy()
    df_full['Data'] = pd.to_datetime(df_full['Data'])
    if 'rpe' in df_full.columns and 'RPE' not in df_full.columns:
        df_full['RPE'] = pd.to_numeric(df_full['rpe'], errors='coerce')
    if 'moving_time' in df_full.columns:
        df_full['duration_hours'] = pd.to_numeric(df_full['moving_time'], errors='coerce') / 3600
    if 'icu_eftp' in df_full.columns:
        df_full['icu_eftp'] = pd.to_numeric(df_full['icu_eftp'], errors='coerce')
    if 'AllWorkFTP' in df_full.columns:
        df_full['AllWorkFTP'] = pd.to_numeric(df_full['AllWorkFTP'], errors='coerce')
    df_full['ano']          = df_full['Data'].dt.year
    df_full['mes']          = df_full['Data'].dt.month
    df_full['trimestre']    = df_full['Data'].dt.quarter
    df_full['ano_trimestre'] = df_full['ano'].astype(str) + '-Q' + df_full['trimestre'].astype(str)

    def _rpe_cat(v):
        try:
            v = float(v)
            if 1 <= v <= 4.5:  return 'leve'
            if 4.6 <= v <= 7:  return 'moderado'
            if 8 <= v <= 10:   return 'pesado'
        except: pass
        return None

    if 'RPE' in df_full.columns:
        df_full['RPE_categoria'] = df_full['RPE'].apply(_rpe_cat)

    modalidades = ['Ski', 'Row', 'Bike', 'Run']
    tabs_mod = st.tabs([f"🎿 Ski", f"🚣 Row", f"🚴 Bike", f"🏃 Run"])

    for tab_m, modalidade in zip(tabs_mod, modalidades):
        with tab_m:
            df_mod = df_full[df_full['type'] == modalidade].copy()
            n_ativ = len(df_mod)

            if n_ativ == 0:
                st.warning(f"Sem dados para {modalidade}.")
                continue

            periodo = (f"{df_mod['Data'].min().strftime('%b %Y')} → "
                       f"{df_mod['Data'].max().strftime('%b %Y')}")
            trimestres = sorted(df_mod['ano_trimestre'].unique())

            # ── Cabeçalho ──────────────────────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Atividades", n_ativ)
            col_b.metric("Período", periodo)
            if 'RPE' in df_mod.columns:
                rpe_m = df_mod['RPE'].dropna()
                col_c.metric("RPE médio", f"{rpe_m.mean():.2f}" if len(rpe_m) > 0 else "—")
            if 'duration_hours' in df_mod.columns:
                h = df_mod['duration_hours'].dropna()
                st.caption(f"⏱️ Horas totais: {h.sum():.1f}h  |  Média/sessão: {h.mean():.2f}h")

            # ── PASSO 1: CV ─────────────────────────────────────────────────
            with st.expander("📊 PASSO 1 — Coeficiente de Variação (CV)", expanded=True):
                cv_rows = []
                for var, label in [('icu_eftp','eFTP (W)'),('AllWorkFTP','AllWorkFTP (kJ)'),
                                   ('RPE','RPE'),('duration_hours','Duração (h)')]:
                    if var not in df_mod.columns: continue
                    s = df_mod[var].dropna()
                    if len(s) < 2 or s.mean() == 0: continue
                    cv = (s.std() / s.mean()) * 100
                    if var == 'RPE':
                        interp = "Muito consistente 🟢" if cv<20 else "Consistente 🟡" if cv<35 else "Variável 🔴"
                    else:
                        interp = "Muito consistente 🟢" if cv<15 else "Consistente 🟡" if cv<30 else "Variável 🔴"
                    cv_rows.append({'Variável': label, 'Média': f"{s.mean():.2f}",
                                    'STD': f"{s.std():.2f}", 'CV%': f"{cv:.1f}%", 'Interpretação': interp})
                if cv_rows:
                    st.dataframe(pd.DataFrame(cv_rows), width="stretch", hide_index=True)
                else:
                    st.info("Sem dados suficientes para CV.")

            # ── PASSO 2: Tendências ─────────────────────────────────────────
            with st.expander("📈 PASSO 2 — Tendências (Slope)"):
                df_sort = df_mod.sort_values('Data')
                trend_rows = []
                for var, label in [('icu_eftp','eFTP (W)'),('AllWorkFTP','AllWorkFTP (kJ)'),
                                   ('RPE','RPE'),('duration_hours','Duração (h)')]:
                    if var not in df_sort.columns: continue
                    s = df_sort[var].dropna()
                    if len(s) < 3: continue
                    x = np.arange(len(s))
                    slope = np.polyfit(x, s.values, 1)[0]
                    if var == 'duration_hours': slope *= 60  # converter para min/atividade
                    unid = 'W/ativ' if var=='icu_eftp' else 'kJ/ativ' if var=='AllWorkFTP' else 'pts/ativ' if var=='RPE' else 'min/ativ'
                    if slope > 0.05:    tendencia = "↗️ Crescente"
                    elif slope < -0.05: tendencia = "↙️ Decrescente"
                    else:               tendencia = "→ Platô"
                    trend_rows.append({'Variável': label, f'Slope ({unid})': f"{slope:+.4f}", 'Tendência': tendencia})
                if trend_rows:
                    st.dataframe(pd.DataFrame(trend_rows), width="stretch", hide_index=True)

                # Correlações por trimestre (eFTP vs AllWorkFTP)
                if 'icu_eftp' in df_mod.columns and 'AllWorkFTP' in df_mod.columns and len(trimestres) > 0:
                    st.markdown("**eFTP e AllWorkFTP por trimestre:**")
                    trim_rows = []
                    for trim in trimestres:
                        dt = df_mod[df_mod['ano_trimestre'] == trim]
                        if len(dt) < 2: continue
                        trim_rows.append({'Trimestre': trim,
                                          'eFTP mediana (W)': f"{dt['icu_eftp'].median():.1f}" if dt['icu_eftp'].notna().any() else '—',
                                          'AllWorkFTP mediana (kJ)': f"{dt['AllWorkFTP'].median():.1f}" if dt['AllWorkFTP'].notna().any() else '—',
                                          'N': len(dt)})
                    if trim_rows:
                        st.dataframe(pd.DataFrame(trim_rows), width="stretch", hide_index=True)

            # ── PASSO 3: Correlações |r| > 0.4 ─────────────────────────────
            with st.expander("🔗 PASSO 3 — Correlações Avançadas (|r| > 0.4)"):
                variaveis = [v for v in ['icu_eftp','AllWorkFTP','WorkHour','RPE','duration_hours','mes']
                             if v in df_mod.columns]
                if len(variaveis) >= 2:
                    df_c = df_mod[variaveis].dropna()
                    if len(df_c) >= 3:
                        mc = df_c.corr()
                        corr_rows = []
                        for i in range(len(mc.columns)):
                            for j in range(i+1, len(mc.columns)):
                                cv = mc.iloc[i,j]
                                if abs(cv) > 0.4:
                                    v1, v2 = mc.columns[i], mc.columns[j]
                                    forca = "MUITO FORTE 🟢" if abs(cv)>0.7 else "FORTE 🟢" if abs(cv)>0.5 else "MODERADA 🟡"
                                    direcao = "positiva ↗️" if cv > 0 else "negativa ↘️"
                                    corr_rows.append({'Var 1': v1, 'Var 2': v2,
                                                      'r': f"{cv:.3f}", 'Força': forca, 'Direção': direcao})
                        if corr_rows:
                            st.dataframe(pd.DataFrame(corr_rows), width="stretch", hide_index=True)
                        else:
                            st.info("Nenhuma correlação > 0.4 encontrada.")
                    else:
                        st.info("Dados insuficientes para correlação.")

            # ── PASSO 4: Sazonalidade ───────────────────────────────────────
            with st.expander("📅 PASSO 4 — Sazonalidade (por trimestre)"):
                if len(trimestres) > 1:
                    saz_rows = []
                    for trim in trimestres:
                        dt = df_mod[df_mod['ano_trimestre'] == trim]
                        row = {'Trimestre': trim, 'N': len(dt)}
                        if 'icu_eftp' in dt.columns and dt['icu_eftp'].notna().any():
                            row['eFTP médio (W)']  = f"{dt['icu_eftp'].mean():.1f} ± {dt['icu_eftp'].std():.1f}"
                        if 'RPE' in dt.columns and dt['RPE'].notna().any():
                            row['RPE médio']       = f"{dt['RPE'].mean():.2f} ± {dt['RPE'].std():.2f}"
                        if 'duration_hours' in dt.columns and dt['duration_hours'].notna().any():
                            row['Horas total']     = f"{dt['duration_hours'].sum():.1f}h"
                            row['Horas/sessão']    = f"{dt['duration_hours'].mean():.2f}h"
                        saz_rows.append(row)
                    if saz_rows:
                        st.dataframe(pd.DataFrame(saz_rows), width="stretch", hide_index=True)
                else:
                    st.info("Apenas 1 trimestre de dados — sem análise sazonal.")

            # ── PASSO 5: RPE por Categoria ──────────────────────────────────
            with st.expander("🎯 PASSO 5 — Distribuição de RPE por Categoria"):
                if 'RPE_categoria' in df_mod.columns:
                    dist = df_mod['RPE_categoria'].value_counts()
                    total = len(df_mod)
                    rpe_rows = []
                    for cat in ['leve', 'moderado', 'pesado']:
                        n_cat = dist.get(cat, 0)
                        rpe_rows.append({'Categoria': cat.capitalize(),
                                         'N': n_cat, '%': f"{n_cat/total*100:.1f}%"})
                    st.dataframe(pd.DataFrame(rpe_rows), width="stretch", hide_index=True)

                    # Por trimestre
                    if len(trimestres) > 1:
                        st.markdown("**Por trimestre:**")
                        trim_rpe = []
                        for trim in trimestres:
                            dt = df_mod[df_mod['ano_trimestre'] == trim]
                            if len(dt) == 0: continue
                            dist_t = dt['RPE_categoria'].value_counts()
                            n_t = len(dt)
                            row = {'Trimestre': trim, 'N': n_t}
                            for cat in ['leve','moderado','pesado']:
                                pct = (dist_t.get(cat,0)/n_t*100) if n_t>0 else 0
                                row[cat.capitalize()+' %'] = f"{pct:.0f}%"
                            trim_rpe.append(row)
                        if trim_rpe:
                            st.dataframe(pd.DataFrame(trim_rpe), width="stretch", hide_index=True)
                else:
                    st.info("Sem dados de RPE para análise de categorias.")

            # ── PASSO 6: Metas baseadas em incrementos reais ────────────────
            with st.expander("🎯 PASSO 6 — Metas baseadas em incrementos reais"):
                if 'icu_eftp' in df_mod.columns and len(trimestres) > 1:
                    incrementos = []
                    for i in range(len(trimestres)-1):
                        e1 = df_mod[df_mod['ano_trimestre']==trimestres[i]]['icu_eftp'].median()
                        e2 = df_mod[df_mod['ano_trimestre']==trimestres[i+1]]['icu_eftp'].median()
                        if not pd.isna(e1) and not pd.isna(e2):
                            incrementos.append(e2 - e1)

                    if incrementos:
                        inc_med = np.mean(incrementos)
                        inc_std = np.std(incrementos)
                        ultimo  = trimestres[-1]
                        eftp_at = df_mod[df_mod['ano_trimestre']==ultimo]['icu_eftp'].median()

                        if not pd.isna(eftp_at):
                            meta_c = eftp_at + inc_med * 0.8
                            meta_m = eftp_at + inc_med
                            meta_a = eftp_at + inc_med * 1.2

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("eFTP actual",      f"{eftp_at:.1f} W",  f"({ultimo})")
                            col2.metric("Meta conservadora",f"{meta_c:.1f} W",   f"{meta_c-eftp_at:+.1f}W")
                            col3.metric("Meta moderada",    f"{meta_m:.1f} W",   f"{meta_m-eftp_at:+.1f}W")
                            col4.metric("Meta ambiciosa",   f"{meta_a:.1f} W",   f"{meta_a-eftp_at:+.1f}W")
                            st.caption(f"Incremento médio por trimestre: {inc_med:+.1f}W (±{inc_std:.1f}W) "
                                       f"baseado em {len(incrementos)} transições")

                            if 'RPE' in df_mod.columns:
                                rpe_m = df_mod['RPE'].mean()
                                if not pd.isna(rpe_m):
                                    if rpe_m < 5:
                                        rec = "💡 Intensidade baixa → meta conservadora recomendada"
                                    elif rpe_m > 7:
                                        rec = "💡 Intensidade alta → meta moderada recomendada"
                                    else:
                                        rec = "💡 Intensidade ideal → meta ambiciosa possível"
                                    st.info(rec)
                else:
                    st.info("Necessário eFTP com pelo menos 2 trimestres para calcular metas.")

    st.markdown("---")

    # ── Secção 8: Resumo Geral ──────────────────────────────────────────────
    st.subheader("📋 Resumo Geral (CTL/ATL/TSB actual)")
    ld_s, _ = calcular_series_carga(da_full)
    if len(ld_s) > 0:
        u_s = ld_s.iloc[-1]
        df7 = filtrar_principais(da_full).copy()
        df7['Data'] = pd.to_datetime(df7['Data'])
        df7 = df7[df7['Data'] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
        horas7 = pd.to_numeric(df7.get('moving_time', pd.Series()), errors='coerce').sum() / 3600
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CTL (Fitness)",  f"{u_s['CTL']:.0f}")
        c2.metric("ATL (Fadiga)",   f"{u_s['ATL']:.0f}")
        c3.metric("TSB (Forma)",    f"{u_s['CTL']-u_s['ATL']:+.0f}")
        c4.metric("Atividades 7d",  len(df7))
        c5.metric("Horas 7d",       fmt_dur(horas7))
    if len(dw) > 0:
        cw1, cw2 = st.columns(2)
        if 'hrv' in dw.columns:
            hrv7 = pd.to_numeric(dw['hrv'], errors='coerce').dropna().tail(7).mean()
            if not pd.isna(hrv7): cw1.metric("HRV médio (7d)", f"{hrv7:.0f} ms")
        if 'rhr' in dw.columns:
            rhr_u = pd.to_numeric(dw['rhr'], errors='coerce').dropna()
            if len(rhr_u) > 0: cw2.metric("RHR último", f"{rhr_u.iloc[-1]:.0f} bpm")

    # Resumo final textual
    with st.expander("✅ ANÁLISE AVANÇADA — Resumo de Interpretação"):
        st.markdown("""
**O que cada métrica significa:**
- **CV baixo** → Consistência no treino (bom para progressão)
- **Slope positivo** → Progressão ao longo do tempo
- **Correlações fortes** → Relações significativas entre variáveis
- **Sazonalidade** → Padrões por trimestre / estação
- **Metas baseadas em dados** → Progressão realista baseada no histórico real
- **BPE Z-Score > +1 SWC** → Estado acima do baseline (boa recuperação)
- **BPE Z-Score < -1 SWC** → Estado abaixo do baseline (atenção à recuperação)

**Escalas:**
- CTL/ATL usa TRIMP = (moving_time_min × RPE) — escala ~300-500 (igual ao Python original)
- BPE usa SWC (Hopkins 2009) — mais sensível que Z-Score tradicional
        """)




# ════════════════════════════════════════════════════════════════════════════════
# TAB 10 — AQUECIMENTO (Annual)
# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════
# tabs/tab_aquecimento.py — ATHELTICA Dashboard — v4
# Aba de Aquecimento — análise completa dos dados Annual
#
# CORRECÇÕES v4:
#   - Slider rolling average DENTRO de cada secção temporal (HR e O2 independentes)
#   - Detecção robusta de Drag Factor (múltiplos nomes possíveis de coluna)
#   - Diagnóstico de colunas para debug
#   - Evolução temporal O2 com slider próprio
#   - Drag Factor para AquecSki E AquecRow
#
# Equivalente ao código original Python:
#   1. HR/O2 vs Potência (Z-Score, SEM, MDC)
#   2. Evolução temporal HR com slider rolling próprio (4 métodos)
#   3. Evolução temporal O2 com slider rolling próprio (4 métodos)
#   4. HR/Pwr ratio
#   5. Drag Factor evolução temporal (AquecRow E AquecSki)
#   6. Correlação Drag Factor vs HR e O2 por W (tabela + scatter)
#   7. SEM/MDC por grupo Drag Factor (3 partes)
# ════════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════════




PLOTLY_WHITE = dict(paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(family='Arial', size=12, color='#333333'))

def _axis(title='', color='#333333', secondary=False):
    d = dict(title=dict(text=title, font=dict(color=color, size=12)),
             tickfont=dict(color=color),
             showgrid=True, gridcolor='#e8e8e8',
             linecolor='#cccccc', linewidth=1, showline=True,
             zeroline=False)
    if secondary:
        d['overlaying'] = 'y'; d['side'] = 'right'; d['showgrid'] = False
    return d

# ── Helpers ───────────────────────────────────────────────────────────────────


PLOTLY_WHITE = dict(paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(family='Arial', size=12, color='#333333'))

def _axis(title='', color='#333333', secondary=False):
    d = dict(title=dict(text=title, font=dict(color=color, size=12)),
             tickfont=dict(color=color),
             showgrid=True, gridcolor='#e8e8e8',
             linecolor='#cccccc', linewidth=1, showline=True,
             zeroline=False)
    if secondary:
        d['overlaying'] = 'y'; d['side'] = 'right'; d['showgrid'] = False
    return d

# ── Helpers ───────────────────────────────────────────────────────────────────


PLOTLY_WHITE = dict(paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(family='Arial', size=12, color='#333333'))

LEGEND_STYLE = dict(orientation='h', y=1.02, x=0,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#aaa', borderwidth=1,
                    font=dict(color='#111111', size=11))

def _xaxis(title='Data', color='#333'):
    return dict(title=dict(text=title, font=dict(color=color, size=12)),
                tickfont=dict(color=color),
                showgrid=True, gridcolor='#e8e8e8',
                linecolor='#ccc', linewidth=1, showline=True)

def _yaxis(title='', color='#333', secondary=False):
    d = dict(title=dict(text=title, font=dict(color=color, size=12)),
             tickfont=dict(color=color),
             showgrid=True, gridcolor='#e8e8e8',
             linecolor='#ccc', linewidth=1, showline=True)
    if secondary:
        d.update({'overlaying':'y','side':'right','showgrid':False})
    return d

# ── Helpers ───────────────────────────────────────────────────────────────────


PLOTLY_WHITE = dict(paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(family='Arial', size=12, color='#333333'))

LEGEND_STYLE = dict(orientation='h', y=1.02, x=0,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#aaa', borderwidth=1,
                    font=dict(color='#111111', size=11))

def _xaxis(title='Data', color='#333'):
    return dict(title=dict(text=title, font=dict(color=color, size=12)),
                tickfont=dict(color=color),
                showgrid=True, gridcolor='#e8e8e8',
                linecolor='#ccc', linewidth=1, showline=True)

def _yaxis(title='', color='#333', secondary=False):
    d = dict(title=dict(text=title, font=dict(color=color, size=12)),
             tickfont=dict(color=color),
             showgrid=True, gridcolor='#e8e8e8',
             linecolor='#ccc', linewidth=1, showline=True)
    if secondary:
        d.update({'overlaying':'y','side':'right','showgrid':False})
    return d

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extrair_pot(col):
    m = re.search(r'(\d+)[_\s]*W', str(col).upper())
    return int(m.group(1)) if m else None

def _detectar_drag_col(df):
    for c in ['Drag_Factor','Drag Factor','DragFactor','drag_factor','Drag','drag','DF']:
        if c in df.columns: return c
    for c in df.columns:
        if 'DRAG' in str(c).upper(): return c
    return None

def _calcular_icc_sem_mdc(valores, tipo='HR'):
    """
    ICC calculado dos próprios dados via diferenças consecutivas.
    SEM de curto prazo: CV_curto = min(CV_dados_consec, CV_max_literatura)
      HR:   CV_max = 4%  (literatura: 1-3%, Achten & Jeukendrup 2003)
      SmO2: CV_max = 8%  (literatura: 3-8% NIRS, Davie 2018)
    MDC₉₅ = CV_curto × média × 1.96
    """
    v = np.array(valores, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 3:
        return None
    media = np.mean(v)
    std   = np.std(v, ddof=1)
    if media == 0:
        return None

    # ICC via variância intra (diferenças consecutivas) vs variância total
    if n >= 4:
        diffs    = np.diff(v)
        var_intra = np.var(diffs, ddof=1) / 2
        var_total = np.var(v, ddof=1)
        icc = max(0.0, min(1.0, (var_total - var_intra) / var_total))
    else:
        icc = max(0.0, 1.0 - (1.0 / max(n, 2)))

    # CV de curto prazo: dos próprios dados (diferenças consecutivas)
    if n >= 4:
        cv_curto_dados = (np.std(np.diff(v), ddof=1) / np.sqrt(2)) / media
    else:
        cv_curto_dados = std / media

    # Limitar pelo máximo da literatura
    cv_max = 0.04 if tipo == 'HR' else 0.08  # 4% HR | 8% SmO2
    cv_curto = min(cv_curto_dados, cv_max)

    # SEM e MDC baseados no CV de curto prazo
    sem    = cv_curto * media / np.sqrt(2)
    sem_pc = cv_curto * 100
    mdc95  = cv_curto * media * 1.96
    mdc_pc = (mdc95 / media) * 100

    if   icc >= 0.90: icc_qual = "✅ Excelente (≥0.90)"
    elif icc >= 0.75: icc_qual = "🟢 Boa (0.75–0.90)"
    elif icc >= 0.50: icc_qual = "🟡 Moderada (0.50–0.75)"
    else:              icc_qual = "⚠️ Fraca (<0.50)"

    return dict(n=n, media=media, std=std,
                icc=icc, icc_qual=icc_qual,
                cv_curto=cv_curto, cv_dados=cv_curto_dados, cv_max=cv_max,
                sem=sem, sem_pc=sem_pc,
                mdc95=mdc95, mdc_pc=mdc_pc)

def _limpar_ruido_sem(series, icc_dict, limiar_multiplo=2.0):
    """
    Remove pontos que desviam > limiar × SEM da média local (rolling 5).
    Retorna série limpa e máscara de ruído.
    """
    if icc_dict is None or len(series) < 4:
        return series, np.zeros(len(series), dtype=bool)
    sem   = icc_dict['sem']
    media_local = series.rolling(5, min_periods=2, center=True).mean().fillna(series.mean())
    ruido = (series - media_local).abs() > limiar_multiplo * sem
    limpa = series.copy()
    limpa[ruido] = np.nan
    return limpa, ruido.values

def _calcular_tendencia_com_mdc(df_col, col_data, col_val, mdc95):
    """
    Calcula tendência para uma janela temporal.
    Valida com MDC: |Δ| > MDC → usa 4 métodos para confirmar direcção.
    Retorna dict com classificação, delta, N, passa_mdc.
    """
    from scipy.stats import theilslopes
    df = df_col[[col_data, col_val]].dropna().sort_values(col_data)
    df[col_data] = pd.to_datetime(df[col_data])
    n = len(df)
    if n < 3:
        return {'classif': '— (N<3)', 'delta': None, 'passa_mdc': None, 'n': n}

    y   = df[col_val].values.astype(float)
    x   = (df[col_data] - df[col_data].min()).dt.days.values.astype(float)
    delta = y[-1] - y[0]  # mudança observada: último - primeiro

    # Valida com MDC
    if mdc95 is not None and abs(delta) <= mdc95:
        return {'classif': f'→ Estável (Δ{delta:+.1f} ≤ MDC{mdc95:.1f})',
                'delta': delta, 'passa_mdc': False, 'n': n}

    # 4 métodos
    try:
        sl, _, _, pv, _   = linregress(x, y)
        tau, p_k           = spearmanr(x, y)
        th_sl, _, _, _     = theilslopes(y, x)
        mid                = max(1, n // 2)
        _, p_t             = scipy_stats.ttest_ind(y[:mid], y[mid:]) if mid >= 2 else (0, 1)

        conf = 0
        if pv  < 0.05 and abs(sl) > 0:                              conf += 2
        if p_k < 0.05:                                               conf += 2
        if (th_sl > 0 and sl > 0) or (th_sl < 0 and sl < 0):        conf += 1
        if p_t < 0.05:                                               conf += 1

        if conf < 2:
            classif = f'→ Estável (Δ{delta:+.1f}, conf insuf.)'
        elif sl > 0:
            classif = f'↗ Aumentando (Δ{delta:+.1f} > MDC)'
        else:
            classif = f'↘ Diminuindo (Δ{delta:+.1f} > MDC)'
    except Exception:
        classif = f'→ Estável (Δ{delta:+.1f})'

    return {'classif': classif, 'delta': delta, 'passa_mdc': True, 'n': n}

def _controles_grafico(aba, secao, n_data):
    c1, c2 = st.columns([2, 1])
    agrup_lbl = c1.selectbox("Agrupar por",
        ["Sessão (sem agrup.)", "Mês", "Trimestre", "Ano"],
        key=f"agrup_{aba}_{secao}")
    agrup_map = {"Sessão (sem agrup.)": None, "Mês":"M", "Trimestre":"Q", "Ano":"A"}
    agrup_code = agrup_map[agrup_lbl]
    roll = c2.slider("Rolling (sessões)", 1, min(12, max(1, n_data-1)), 3,
                     key=f"roll_{aba}_{secao}")
    return agrup_code, agrup_lbl, roll

def _agrupar_serie(df, col_data, col_val, agrup_code):
    d = df[[col_data, col_val]].dropna().copy()
    d[col_data] = pd.to_datetime(d[col_data])
    d['_p'] = d[col_data].dt.to_period(agrup_code)
    agg = d.groupby('_p')[col_val].agg(['mean','std','count']).reset_index()
    agg['_ts'] = agg['_p'].dt.to_timestamp()
    if agrup_code == 'M':
        agg['_lbl'] = agg['_p'].dt.strftime('%b %Y')
    elif agrup_code == 'Q':
        agg['_lbl'] = agg['_p'].apply(
            lambda p: f"Q{((p.start_time.month-1)//3)+1} {p.start_time.year}")
    else:
        agg['_lbl'] = agg['_p'].dt.strftime('%Y')
    return agg


# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════


def tab_aquecimento(dfs_annual, df_annual, di):
    st.header("🌡️ Aquecimento — HR/O2 vs Potência (Annual)")

    if dfs_annual is None or all(v.empty for v in (dfs_annual or {}).values()):
        st.warning("Planilha Annual não carregada.")
        return

    ABAS = [("🎿 Ski","AquecSki"),("🚴 Bike","AquecBike"),("🚣 Row","AquecRow")]
    tabs_aq = st.tabs([a[0] for a in ABAS])

    for tab_aq, (label, aba) in zip(tabs_aq, ABAS):
        with tab_aq:
            df_a = (dfs_annual or {}).get(aba, pd.DataFrame())
            if df_a.empty:
                st.info(f"Sem dados para {aba}.")
                continue
            df_a = df_a.copy()

            # ── Filtro de período global (só para gráficos) ───────────
            if 'DATA' in df_a.columns and df_a['DATA'].notna().any():
                d_min_h = df_a['DATA'].min().date()
                d_max_h = df_a['DATA'].max().date()
                st.caption(f"📅 Histórico: {len(df_a)} registos | "
                           f"{d_min_h.strftime('%d/%m/%Y')} → {d_max_h.strftime('%d/%m/%Y')}")
            else:
                d_min_h = d_max_h = None

            cf1, cf2 = st.columns([2,2])
            with cf1:
                opcao = st.selectbox("📅 Período para gráficos",
                    ["Todo o histórico","Últimos 60 dias","Últimos 90 dias",
                     "Últimos 180 dias","Último ano (365 dias)","Datas manuais"],
                    index=0, key=f"periodo_sel_{aba}")
            with cf2:
                dias_map_g = {"Últimos 60 dias":60,"Últimos 90 dias":90,
                              "Últimos 180 dias":180,"Último ano (365 dias)":365}
                if opcao == "Datas manuais" and d_min_h and d_max_h:
                    data_ini_man = st.date_input("Início", value=d_min_h,
                        min_value=d_min_h, max_value=d_max_h, key=f"data_ini_{aba}")
                    data_fim_man = st.date_input("Fim", value=d_max_h,
                        min_value=d_min_h, max_value=d_max_h, key=f"data_fim_{aba}")
                else:
                    data_ini_man = data_fim_man = None
                    nd = dias_map_g.get(opcao, 0)
                    if nd and d_max_h:
                        st.caption(f"De {(d_max_h-timedelta(days=nd)).strftime('%d/%m/%Y')} "
                                   f"até {d_max_h.strftime('%d/%m/%Y')}")

            if 'DATA' in df_a.columns and df_a['DATA'].notna().any():
                if opcao == "Todo o histórico":
                    df_plot = df_a.copy()
                elif opcao == "Datas manuais" and data_ini_man and data_fim_man:
                    df_plot = df_a[(df_a['DATA'].dt.date >= data_ini_man) &
                                   (df_a['DATA'].dt.date <= data_fim_man)].copy()
                else:
                    nd = dias_map_g.get(opcao, 0)
                    df_plot = (df_a[df_a['DATA'].dt.date >= d_max_h-timedelta(days=nd)].copy()
                               if nd and d_max_h else df_a.copy())
                if len(df_plot) == 0:
                    st.warning("Sem dados no período — usando histórico completo.")
                    df_plot = df_a.copy()
                else:
                    st.caption(f"✅ {len(df_plot)} registos no período")
            else:
                df_plot = df_a.copy()

            # ── Diagnóstico ────────────────────────────────────────────
            with st.expander("🔍 Diagnóstico de colunas"):
                cols_info = []
                for c in df_a.columns:
                    cols_info.append({
                        'Coluna':c, 'Tipo':str(df_a[c].dtype),
                        'N não-nulos':int(df_a[c].notna().sum()),
                        'Detecção':(
                            'HR'    if ('HR' in c.upper() and 'PWR' not in c.upper()
                                        and 'DRAG' not in c.upper() and _extrair_pot(c))
                            else 'O2'     if ('O2' in c.upper() and _extrair_pot(c))
                            else 'HR/Pwr' if ('PWR' in c.upper() and _extrair_pot(c))
                            else 'Drag'   if 'DRAG' in c.upper()
                            else 'DATA'   if c == 'DATA' else '—')})
                st.dataframe(pd.DataFrame(cols_info), width="stretch", hide_index=True)

            with st.expander("📋 Dados (mais recentes)"):
                cols_ok = [c for c in df_plot.columns
                           if not str(c).startswith('Unnamed') and df_plot[c].notna().any()]
                df_show = (df_plot[cols_ok].sort_values('DATA', ascending=False)
                           if 'DATA' in df_plot.columns else df_plot[cols_ok].iloc[::-1])
                st.dataframe(df_show.head(20), width="stretch")

            # Detectar colunas
            hr_cols  = sorted([c for c in df_a.columns if 'HR' in c.upper()
                                and 'PWR' not in c.upper() and 'DRAG' not in c.upper()
                                and _extrair_pot(c)], key=lambda c: _extrair_pot(c) or 0)
            o2_cols  = sorted([c for c in df_a.columns if 'O2' in c.upper()
                                and _extrair_pot(c)], key=lambda c: _extrair_pot(c) or 0)
            pwr_cols = sorted([c for c in df_a.columns if 'PWR' in c.upper()
                                and _extrair_pot(c)], key=lambda c: _extrair_pot(c) or 0)
            drag_col   = _detectar_drag_col(df_a)
            treino_col = next((c for c in df_a.columns
                               if 'treino' in c.lower() or 'antes' in c.lower()), None)

            CORES_PW = ['#E74C3C','#F39C12','#9B59B6','#2ECC71','#3498DB']
            CORES_O2 = ['#2471A3','#1D8348','#7D3C98','#117A65','#C0392B']

            if not hr_cols and not o2_cols:
                st.warning("Sem colunas HR/O2 detectadas.")
                continue

            # Pré-calcular ICC/SEM/MDC para todo o histórico (uma vez por coluna)
            # Passa tipo para usar CV_max correcto: HR=4%, SmO2=8%
            icc_cache = {}
            for col in hr_cols:
                if col in df_a.columns:
                    icc_cache[col] = _calcular_icc_sem_mdc(df_a[col].dropna().values, tipo='HR')
            for col in o2_cols:
                if col in df_a.columns:
                    icc_cache[col] = _calcular_icc_sem_mdc(df_a[col].dropna().values, tipo='SmO2')

            # ════════════════════════════════════════════════════════════
            # 1. HR e O2 vs Potência — Z-Score, SEM, MDC
            # ════════════════════════════════════════════════════════════
            st.subheader("📊 HR e O2 vs Potência — Z-Score (±2σ), SEM e MDC")

            rows_hr = [{'Power':_extrair_pot(c),'Value':float(v)}
                       for c in hr_cols for v in df_plot[c].dropna()]
            rows_o2 = [{'Power':_extrair_pot(c),'Value':float(v)}
                       for c in o2_cols for v in df_plot[c].dropna()]
            df_hr_l = pd.DataFrame(rows_hr)
            df_o2_l = pd.DataFrame(rows_o2)

            if len(df_hr_l) > 0 or len(df_o2_l) > 0:
                rng = np.random.default_rng(42)
                fig_s = go.Figure()

                if len(df_hr_l) > 0:
                    jit = rng.normal(0, 0.3, len(df_hr_l))
                    fig_s.add_trace(go.Scatter(x=df_hr_l['Power']+jit, y=df_hr_l['Value'],
                        mode='markers', name='HR pontos',
                        marker=dict(color='#e74c3c', opacity=0.3, size=6),
                        hovertemplate='%{x:.0f}W  HR: <b>%{y:.0f} bpm</b><extra></extra>'))
                    agg_hr = df_hr_l.groupby('Power')['Value'].mean().reset_index()
                    fig_s.add_trace(go.Scatter(x=agg_hr['Power'], y=agg_hr['Value'],
                        mode='lines+markers', name='HR média',
                        line=dict(color='#c0392b', width=3), marker=dict(size=10),
                        hovertemplate='%{x:.0f}W  HR média: <b>%{y:.0f} bpm</b><extra></extra>'))
                    for pw in sorted(df_hr_l['Power'].unique()):
                        vals = df_hr_l[df_hr_l['Power']==pw]['Value'].values
                        if len(vals) < 2: continue
                        s = _calcular_icc_sem_mdc(vals)
                        if not s: continue
                        m = s['media']
                        fig_s.add_shape(type='rect', x0=pw-2, x1=pw+2,
                            y0=m-2*s['std'], y1=m+2*s['std'],
                            fillcolor='rgba(231,76,60,0.07)', line=dict(width=0))
                        for yv, dash in [(m+2*s['std'],'dash'),(m-2*s['std'],'dash'),
                                         (m+s['mdc95'],'dot'),(m-s['mdc95'],'dot')]:
                            fig_s.add_shape(type='line', x0=pw-2, x1=pw+2, y0=yv, y1=yv,
                                line=dict(color='#c0392b', width=1.5, dash=dash))

                if len(df_o2_l) > 0:
                    jit2 = rng.normal(0, 0.3, len(df_o2_l))
                    fig_s.add_trace(go.Scatter(x=df_o2_l['Power']+jit2, y=df_o2_l['Value'],
                        mode='markers', name='O2 pontos',
                        marker=dict(color='#3498db', opacity=0.3, size=6),
                        hovertemplate='%{x:.0f}W  SmO2: <b>%{y:.1f}%</b><extra></extra>',
                        yaxis='y2'))
                    agg_o2 = df_o2_l.groupby('Power')['Value'].mean().reset_index()
                    fig_s.add_trace(go.Scatter(x=agg_o2['Power'], y=agg_o2['Value'],
                        mode='lines+markers', name='O2 média',
                        line=dict(color='#2471a3', width=3),
                        marker=dict(size=10, symbol='square'),
                        hovertemplate='%{x:.0f}W  SmO2 média: <b>%{y:.1f}%</b><extra></extra>',
                        yaxis='y2'))
                    for pw in sorted(df_o2_l['Power'].unique()):
                        vals = df_o2_l[df_o2_l['Power']==pw]['Value'].values
                        if len(vals) < 2: continue
                        s = _calcular_icc_sem_mdc(vals)
                        if not s: continue
                        m = s['media']
                        fig_s.add_shape(type='rect', x0=pw-2, x1=pw+2,
                            y0=m-2*s['std'], y1=m+2*s['std'],
                            fillcolor='rgba(52,152,219,0.07)', line=dict(width=0), yref='y2')

                fig_s.update_layout(**PLOTLY_WHITE,
                    title=dict(text=f'{aba} — HR e O2 vs Potência | Z-Score (±2σ) e MDC-95',
                               font=dict(size=14, color='#222')),
                    height=500, hovermode='x unified',
                    xaxis=_xaxis('Potência (W)'),
                    yaxis=_yaxis('HR (bpm)', '#c0392b'),
                    yaxis2=_yaxis('SmO2 (%)', '#2471a3', secondary=True),
                    legend=LEGEND_STYLE)
                st.plotly_chart(fig_s, use_container_width=True)

            st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # 2+3. Evolução temporal HR e SmO2 — com ICC/SEM/MDC
            # ════════════════════════════════════════════════════════════
            for tipo_t, cols_t, cores_t, sec_key, unid_t, cor_eixo in [
                ('HR',   hr_cols, CORES_PW, 'hr', 'bpm', '#c0392b'),
                ('SmO2', o2_cols, CORES_O2, 'o2', '%',   '#2471a3'),
            ]:
                if not cols_t: continue
                if not any(col in df_plot.columns and df_plot[col].notna().any()
                           for col in cols_t):
                    continue

                st.subheader(f"📈 Evolução temporal {tipo_t}")
                agrup_code, agrup_lbl, roll = _controles_grafico(aba, sec_key, len(df_plot))

                fig_t = go.Figure()

                for idx_c, col in enumerate(cols_t):
                    if col not in df_plot.columns: continue
                    pw  = _extrair_pot(col)
                    cor = cores_t[idx_c % len(cores_t)]
                    df_t = df_plot[['DATA', col]].dropna().sort_values('DATA')
                    if len(df_t) < 2: continue

                    # Limpeza de ruído com SEM (usando ICC do histórico completo)
                    icc_d = icc_cache.get(col)
                    serie_limpa, mascara_ruido = _limpar_ruido_sem(df_t[col], icc_d)
                    df_t = df_t.copy()
                    df_t['_limpa'] = serie_limpa

                    # Pontos marcados como ruído (mais transparentes)
                    if mascara_ruido.any():
                        df_ruido = df_t[mascara_ruido]
                        fig_t.add_trace(go.Scatter(
                            x=df_ruido['DATA'], y=df_ruido[col],
                            mode='markers', name=f'{pw}W ruído',
                            marker=dict(color=cor, opacity=0.15, size=5,
                                        symbol='x'),
                            hovertemplate=f'%{{x|%d/%m/%Y}}  {tipo_t} {pw}W (ruído): <b>%{{y:.1f}} {unid_t}</b><extra></extra>',
                            showlegend=False))

                    if agrup_code:
                        # Agrupa sobre dados limpos
                        df_t_limpa = df_t[['DATA','_limpa']].rename(columns={'_limpa': col})
                        agg = _agrupar_serie(df_t_limpa, 'DATA', col, agrup_code)
                        # Pontos originais (fundo)
                        fig_t.add_trace(go.Scatter(
                            x=df_t['DATA'], y=df_t[col],
                            mode='markers', name=f'{pw}W pontos',
                            marker=dict(color=cor, opacity=0.15, size=4),
                            showlegend=False,
                            hovertemplate=f'%{{x|%d/%m/%Y}}  {tipo_t} {pw}W: <b>%{{y:.1f}} {unid_t}</b><extra></extra>'))
                        # Média por período
                        fig_t.add_trace(go.Scatter(
                            x=agg['_ts'], y=agg['mean'],
                            mode='lines+markers', name=f'{pw}W ({agrup_lbl})',
                            line=dict(color=cor, width=2.5),
                            marker=dict(size=8),
                            customdata=np.stack([agg['_lbl'], agg['count'],
                                                 agg['std'].fillna(0)], axis=-1),
                            hovertemplate=(f'%{{customdata[0]}}<br>{tipo_t} {pw}W: '
                                           f'<b>%{{y:.1f}} {unid_t}</b><br>'
                                           f'±%{{customdata[2]:.1f}} | N=%{{customdata[1]}}'
                                           f'<extra></extra>')))
                        if len(agg) >= 3:
                            xn = np.arange(len(agg), dtype=float)
                            try:
                                sl, ic, _, pv, _ = linregress(xn, agg['mean'].values)
                                y_tr = ic + sl * xn
                                sig = '✓' if pv < 0.05 else '—'
                                classif = ('↗' if sl > 0 else '↘') if pv < 0.05 else '→'
                                fig_t.add_trace(go.Scatter(
                                    x=agg['_ts'], y=y_tr,
                                    mode='lines', name=f'{pw}W {classif} ({sig})',
                                    line=dict(color=cor, width=1.2, dash='dash'),
                                    hovertemplate=f'Tendência {pw}W: <b>%{{y:.1f}} {unid_t}</b><extra></extra>'))
                            except Exception:
                                pass
                    else:
                        # Sessão a sessão: dados limpos + rolling + tendência
                        fig_t.add_trace(go.Scatter(
                            x=df_t['DATA'], y=df_t[col],
                            mode='markers', name=f'{pw}W',
                            marker=dict(color=cor, opacity=0.3, size=6),
                            showlegend=False,
                            hovertemplate=f'%{{x|%d/%m/%Y}}  {tipo_t} {pw}W: <b>%{{y:.1f}} {unid_t}</b><extra></extra>'))
                        # Rolling sobre dados limpos
                        roll_s = df_t['_limpa'].rolling(roll, min_periods=1).mean()
                        fig_t.add_trace(go.Scatter(
                            x=df_t['DATA'], y=roll_s,
                            mode='lines', name=f'{pw}W roll({roll})',
                            line=dict(color=cor, width=2.5),
                            hovertemplate=f'%{{x|%d/%m/%Y}}  {tipo_t} {pw}W roll: <b>%{{y:.1f}} {unid_t}</b><extra></extra>'))
                        if len(df_t) >= 3:
                            xd = (df_t['DATA'] - df_t['DATA'].min()).dt.days.values.astype(float)
                            y_limpa = df_t['_limpa'].values
                            mask = ~np.isnan(y_limpa)
                            if mask.sum() >= 3:
                                try:
                                    sl, ic, _, pv, _ = linregress(xd[mask], y_limpa[mask])
                                    y_tr = ic + sl * xd
                                    sig = '✓' if pv < 0.05 else '—'
                                    classif = ('↗' if sl > 0 else '↘') if pv < 0.05 else '→'
                                    fig_t.add_trace(go.Scatter(
                                        x=df_t['DATA'], y=y_tr,
                                        mode='lines', name=f'{pw}W {classif} ({sig})',
                                        line=dict(color=cor, width=1.2, dash='dash'),
                                        hovertemplate=f'Tendência {pw}W: <b>%{{y:.1f}} {unid_t}</b><extra></extra>'))
                                except Exception:
                                    pass

                title_suf = (f" — agrupado por {agrup_lbl}"
                             if agrup_code else f" — sessão a sessão | roll={roll}")
                fig_t.update_layout(**PLOTLY_WHITE,
                    title=dict(text=f'{aba} — {tipo_t} evolução temporal{title_suf}',
                               font=dict(size=14, color='#222')),
                    height=420, hovermode='x unified',
                    xaxis=_xaxis('Data'),
                    yaxis=_yaxis(f'{tipo_t} ({unid_t})', cor_eixo),
                    legend=LEGEND_STYLE)
                st.plotly_chart(fig_t, use_container_width=True)

                # ════════════════════════════════════════════
                # TABELAS ICC/SEM/MDC + Tendência por período
                # ════════════════════════════════════════════
                st.markdown(f"**📊 Análise {tipo_t} — Confiabilidade e Tendências por Potência**")

                # SECÇÃO A: ICC / SEM / MDC — todo o histórico
                rows_icc = []
                for col in cols_t:
                    if col not in df_a.columns: continue
                    pw  = _extrair_pot(col)
                    icd = icc_cache.get(col)
                    if icd is None:
                        rows_icc.append({'Potência':f'{pw}W','N':'<3',
                            'Média':'—','SEM':'—','SEM%':'—',
                            'MDC₉₅':'—','MDC%':'—','ICC':'—','Qualidade ICC':'— (N insuf.)'})
                        continue
                    cv_src = ("dados" if icd['cv_dados'] <= icd['cv_max']
                              else f"cap {icd['cv_max']*100:.0f}% lit.")
                    rows_icc.append({
                        'Potência':      f'{pw}W',
                        'N':             icd['n'],
                        'Média':         f"{icd['media']:.1f} {unid_t}",
                        'CV curto':      f"{icd['cv_curto']*100:.1f}% ({cv_src})",
                        'SEM':           f"{icd['sem']:.2f} {unid_t}",
                        'SEM%':          f"{icd['sem_pc']:.1f}%",
                        'MDC₉₅':         f"{icd['mdc95']:.1f} {unid_t}",
                        'MDC%':          f"{icd['mdc_pc']:.1f}%",
                        'ICC':           f"{icd['icc']:.3f}",
                        'Qualidade ICC': icd['icc_qual'],
                    })
                if rows_icc:
                    st.caption("**Secção A — Confiabilidade (todo o histórico):** "
                               "SEM = erro esperado entre medições | "
                               "MDC₉₅ = mudança mínima real (95% confiança) | "
                               "ICC = consistência das medições (≠ capacidade preditiva)")
                    st.dataframe(pd.DataFrame(rows_icc), width="stretch", hide_index=True)

                # SECÇÃO B: Tendência por período, validada pelo MDC
                janelas = [("30 dias",30),("60 dias",60),("90 dias",90),
                           ("1 ano",365),("Todo histórico",None)]
                rows_tend = []
                for col in cols_t:
                    if col not in df_a.columns: continue
                    pw  = _extrair_pot(col)
                    icd = icc_cache.get(col)
                    mdc = icd['mdc95'] if icd else None
                    df_col = df_a[['DATA', col]].dropna().copy()
                    df_col['DATA'] = pd.to_datetime(df_col['DATA'])
                    hoje = df_col['DATA'].max()

                    row = {'Potência': f'{pw}W'}
                    for lbl, dias in janelas:
                        sub = (df_col[df_col['DATA'] >= hoje - pd.Timedelta(days=dias)]
                               if dias else df_col)
                        if len(sub) < 3:
                            row[lbl] = '— (N<3)'
                            continue
                        # Limpeza de ruído com SEM antes de calcular tendência
                        if icd:
                            serie_l, _ = _limpar_ruido_sem(sub[col], icd)
                            sub = sub.copy(); sub[col] = serie_l
                            sub = sub.dropna(subset=[col])
                        if len(sub) < 3:
                            row[lbl] = '— (N<3 após limpeza)'
                            continue
                        res = _calcular_tendencia_com_mdc(sub, 'DATA', col, mdc)
                        row[lbl] = res['classif']
                    rows_tend.append(row)

                if rows_tend:
                    st.caption("**Secção B — Tendência por período (validada pelo MDC):** "
                               "Δ ≤ MDC → Estável (dentro do erro de medição) | "
                               "Δ > MDC + evidência estatística → mudança provavelmente real")
                    st.dataframe(pd.DataFrame(rows_tend), width="stretch", hide_index=True)

                # Rodapé interpretação
                st.caption(
                    "⚠️ **Guia:** "
                    "1) SEM = erro esperado nas mesmas condições  "
                    "2) MDC₉₅ = mudança mínima para ser considerada real  "
                    "3) ICC = fiabilidade das medições ao longo do tempo  "
                    "4) Δ ≤ MDC = provavelmente ruído | Δ > MDC = provavelmente real  "
                    "⚠️ ICC alto = medições consistentes, NÃO significa que uma variável prediz outra")

                st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # 4. HR/Pwr ratio
            # ════════════════════════════════════════════════════════════
            if pwr_cols and 'DATA' in df_a.columns:
                st.subheader("⚡ HR/Pwr ratio — eficiência cardíaca (↘ = melhora)")
                agrup_pwr, agrup_pwr_lbl, roll_pwr = _controles_grafico(aba, 'pwr', len(df_plot))
                fig_pwr = go.Figure()
                for i, col in enumerate(pwr_cols[:4]):
                    df_p = df_plot[['DATA', col]].dropna().sort_values('DATA')
                    if len(df_p) < 2: continue
                    pw  = _extrair_pot(col)
                    cor = CORES_PW[i % len(CORES_PW)]
                    if agrup_pwr:
                        agg_p = _agrupar_serie(df_p, 'DATA', col, agrup_pwr)
                        fig_pwr.add_trace(go.Scatter(
                            x=agg_p['_ts'], y=agg_p['mean'],
                            mode='lines+markers', name=f'{pw}W',
                            line=dict(color=cor, width=2.5), marker=dict(size=8),
                            customdata=np.stack([agg_p['_lbl']], axis=-1),
                            hovertemplate=f'%{{customdata[0]}}  HR/Pwr {pw}W: <b>%{{y:.3f}} bpm/W</b><extra></extra>'))
                    else:
                        fig_pwr.add_trace(go.Scatter(
                            x=df_p['DATA'], y=df_p[col],
                            mode='markers', name=f'{pw}W',
                            marker=dict(color=cor, opacity=0.35, size=6),
                            showlegend=False,
                            hovertemplate=f'%{{x|%d/%m/%Y}}  HR/Pwr {pw}W: <b>%{{y:.3f}}</b><extra></extra>'))
                        fig_pwr.add_trace(go.Scatter(
                            x=df_p['DATA'],
                            y=df_p[col].rolling(roll_pwr, min_periods=1).mean(),
                            mode='lines', name=f'{pw}W roll({roll_pwr})',
                            line=dict(color=cor, width=2.5),
                            hovertemplate=f'%{{x|%d/%m/%Y}}  HR/Pwr {pw}W roll: <b>%{{y:.3f}}</b><extra></extra>'))
                fig_pwr.update_layout(**PLOTLY_WHITE,
                    title=dict(text=f'{aba} — HR/Pwr ratio{(" — "+agrup_pwr_lbl) if agrup_pwr else ""}',
                               font=dict(size=14, color='#222')),
                    height=380, hovermode='x unified',
                    xaxis=_xaxis('Data'),
                    yaxis=_yaxis('HR/Pwr (bpm/W)'),
                    legend=LEGEND_STYLE)
                st.plotly_chart(fig_pwr, use_container_width=True)
                st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # 5. Drag Factor
            # ════════════════════════════════════════════════════════════
            if drag_col:
                df_drag = df_plot[['DATA', drag_col]].dropna().sort_values('DATA')
                if len(df_drag) >= 2:
                    st.subheader(f"⚙️ Drag Factor — evolução temporal ({aba})")
                    agrup_drag, agrup_drag_lbl, roll_drag = _controles_grafico(aba,'drag',len(df_drag))
                    mu_d = df_drag[drag_col].mean(); sd_d = df_drag[drag_col].std()
                    fig_drag = go.Figure()
                    if agrup_drag:
                        agg_d = _agrupar_serie(df_drag, 'DATA', drag_col, agrup_drag)
                        fig_drag.add_trace(go.Scatter(
                            x=df_drag['DATA'], y=df_drag[drag_col],
                            mode='markers', name='Medições',
                            marker=dict(color='#3498db', opacity=0.2, size=5),
                            showlegend=False,
                            hovertemplate='%{x|%d/%m/%Y}  Drag: <b>%{y:.0f}</b><extra></extra>'))
                        fig_drag.add_trace(go.Scatter(
                            x=agg_d['_ts'], y=agg_d['mean'],
                            mode='lines+markers', name=f'Drag ({agrup_drag_lbl})',
                            line=dict(color='#3498db', width=2.5), marker=dict(size=8),
                            customdata=np.stack([agg_d['_lbl'], agg_d['count']], axis=-1),
                            hovertemplate='%{customdata[0]}  Drag: <b>%{y:.0f}</b>  N=%{customdata[1]}<extra></extra>'))
                    else:
                        fig_drag.add_trace(go.Scatter(
                            x=df_drag['DATA'], y=df_drag[drag_col],
                            mode='markers', name='Medições',
                            marker=dict(color='#3498db', opacity=0.45, size=7),
                            hovertemplate='%{x|%d/%m/%Y}  Drag: <b>%{y:.0f}</b><extra></extra>'))
                        fig_drag.add_trace(go.Scatter(
                            x=df_drag['DATA'],
                            y=df_drag[drag_col].rolling(roll_drag, min_periods=1).mean(),
                            mode='lines', name=f'Rolling ({roll_drag})',
                            line=dict(color='#2471a3', width=2.5),
                            hovertemplate='%{x|%d/%m/%Y}  Drag roll: <b>%{y:.0f}</b><extra></extra>'))
                    fig_drag.add_hline(y=mu_d, line_dash='dash', line_color='red',
                                       annotation_text=f'Média: {mu_d:.0f}',
                                       annotation_font=dict(color='red'))
                    fig_drag.add_hrect(y0=mu_d-sd_d, y1=mu_d+sd_d,
                                       fillcolor='rgba(52,152,219,0.08)', line_width=0)
                    fig_drag.update_layout(**PLOTLY_WHITE,
                        title=dict(text=f'Drag Factor — {aba}{(" — "+agrup_drag_lbl) if agrup_drag else ""}',
                                   font=dict(size=14, color='#222')),
                        height=360, hovermode='x unified',
                        xaxis=_xaxis('Data'),
                        yaxis=_yaxis('Drag Factor'),
                        legend=LEGEND_STYLE)
                    st.plotly_chart(fig_drag, use_container_width=True)
                    c1d, c2d, c3d = st.columns(3)
                    c1d.metric("Drag médio", f"{mu_d:.0f}")
                    c2d.metric("Mínimo",     f"{df_drag[drag_col].min():.0f}")
                    c3d.metric("Máximo",     f"{df_drag[drag_col].max():.0f}")
                    st.markdown("---")
            elif aba in ('AquecSki','AquecRow'):
                st.info(f"⚠️ Drag Factor não detectado em **{aba}**. "
                        f"Colunas: `{', '.join(df_a.columns.tolist())}`")

            # ════════════════════════════════════════════════════════════
            # 6. Quartis de Drag Factor vs HR/O2 — todo histórico
            # ════════════════════════════════════════════════════════════
            if drag_col and (hr_cols or o2_cols):
                st.subheader(f"🔗 Drag Factor (quartis) vs HR / SmO2 — {aba}")
                st.caption("Todo o histórico. Quartis de DF com range real. "
                           "Kruskal-Wallis testa diferença global entre quartis.")

                df_db = df_a.dropna(subset=[drag_col]).copy()
                if len(df_db) >= 8:
                    try:
                        df_db['_q'], bins = pd.qcut(df_db[drag_col], q=4,
                            labels=False, retbins=True, duplicates='drop')
                        q_labels = [f"Q{i+1} ({bins[i]:.0f}–{bins[i+1]:.0f})"
                                    for i in range(len(bins)-1)]
                        df_db['_qlbl'] = df_db['_q'].apply(
                            lambda x: q_labels[int(x)] if pd.notna(x) else None)
                    except Exception:
                        df_db['_qlbl'] = 'Único'; q_labels = ['Único']

                    for tipo_d, cols_d, unid_d in [('HR',hr_cols,'bpm'),('SmO2',o2_cols,'%')]:
                        if not cols_d: continue
                        drag_rows = []
                        for col in cols_d:
                            if col not in df_db.columns: continue
                            pw_d = _extrair_pot(col)
                            df_col = df_db[[col,'_qlbl']].dropna()
                            if len(df_col) < 4: continue
                            medias = {g: df_col[df_col['_qlbl']==g][col].values
                                      for g in q_labels if g in df_col['_qlbl'].values}
                            vals_list = [v for v in medias.values() if len(v) >= 2]
                            if len(vals_list) >= 2:
                                try:
                                    _, p_kw = kruskal(*vals_list)
                                    sig_g = '✓ SIG' if p_kw < 0.05 else '✗ ns'
                                except Exception:
                                    p_kw = 1.0; sig_g = '✗ ns'
                            else:
                                p_kw = 1.0; sig_g = '— (poucos dados)'

                            row = {'Potência':f'{pw_d}W','Tipo':tipo_d,
                                   'Dif. global':f"{sig_g} (p={p_kw:.3f})"}
                            for g in q_labels:
                                row[g] = (f"{np.mean(medias[g]):.1f} {unid_d}"
                                          if g in medias and len(medias[g]) >= 1 else '—')
                            medias_ord = [np.mean(medias[g]) for g in q_labels
                                          if g in medias and len(medias[g]) >= 1]
                            if len(medias_ord) >= 2:
                                delta = medias_ord[-1] - medias_ord[0]
                                row['Q1→Q4'] = (f"↗ {delta:+.1f} {unid_d} com DF↑"
                                                if (p_kw<0.05 and delta>0)
                                                else f"↘ {delta:+.1f} {unid_d} com DF↑"
                                                if (p_kw<0.05 and delta<0)
                                                else "→ sem diferença significativa")
                            drag_rows.append(row)
                        if drag_rows:
                            st.markdown(f"**{tipo_d} ({unid_d}) por quartil de Drag Factor**")
                            st.dataframe(pd.DataFrame(drag_rows), width="stretch", hide_index=True)
                else:
                    st.info("Poucos dados para análise por quartis (N < 8).")
                st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # 7. Treino Antes — todo histórico
            # ════════════════════════════════════════════════════════════
            if treino_col and (hr_cols or o2_cols):
                st.subheader(f"🏋️ Treino Antes — impacto no HR/SmO2 ({aba})")
                st.caption("Todo o histórico. Mann-Whitney U entre grupos. "
                           "Grupos: Sem treino | Com treino (todos) | Pesos | Cíclicos.")

                df_ta = df_a.copy()
                df_ta[treino_col] = (df_ta[treino_col].astype(str).str.strip().str.lower()
                                     .replace({'nan':None,'none':None,'':None,'n/a':None}))
                df_ta['_sem'] = df_ta[treino_col].isna()

                # Classificar categorias como 'pesos' ou 'ciclicos'
                cats_raw = sorted(df_ta[treino_col].dropna().unique().tolist())
                pesos_kws   = ['peso','weight','gym','musculação','musculacao','strength']
                ciclicos_kws= ['bike','run','row','ski','ciclico','aerob','swim']

                def _cat_tipo(c):
                    cl = c.lower()
                    if any(k in cl for k in pesos_kws):   return 'pesos'
                    if any(k in cl for k in ciclicos_kws): return 'ciclicos'
                    return 'outros'

                df_ta['_tipo_treino'] = df_ta[treino_col].apply(
                    lambda x: _cat_tipo(x) if pd.notna(x) else None)

                for tipo_d, cols_d, unid_d in [('HR',hr_cols,'bpm'),('SmO2',o2_cols,'%')]:
                    if not cols_d: continue
                    treino_rows = []
                    for col in cols_d:
                        if col not in df_ta.columns: continue
                        pw_d = _extrair_pot(col)
                        df_col = df_ta[[treino_col, col, '_sem','_tipo_treino']].dropna(subset=[col])
                        if len(df_col) < 4: continue

                        sem_v    = df_col[df_col['_sem']][col].values
                        com_v    = df_col[~df_col['_sem']][col].values
                        pesos_v  = df_col[df_col['_tipo_treino']=='pesos'][col].values
                        cicl_v   = df_col[df_col['_tipo_treino']=='ciclicos'][col].values

                        row = {'Potência': f'{pw_d}W', 'Tipo': tipo_d}

                        # Médias por grupo
                        for lbl, vals in [('Sem treino',sem_v),('Com treino',com_v),
                                          ('Pesos',pesos_v),('Cíclicos',cicl_v)]:
                            row[lbl] = (f"{np.mean(vals):.1f} {unid_d}"
                                        if len(vals) >= 1 else '—')

                        # Comparações estatísticas — saída compacta
                        def _comp(v1, v2, lbl, ud):
                            if len(v1) < 3 or len(v2) < 3: return None
                            try:
                                _, p = mannwhitneyu(v1, v2, alternative='two-sided')
                                delta = np.mean(v1) - np.mean(v2)
                                if p < 0.05:
                                    return f"{lbl}: {delta:+.1f} {ud}"
                                else:
                                    return f"{lbl}: sem mudança"
                            except Exception: return None

                        comps = []
                        r = _comp(com_v, sem_v, "Com treino", unid_d)
                        if r: comps.append(r)
                        r2 = _comp(pesos_v, cicl_v, "Pesos vs Cíclicos", unid_d)
                        if r2: comps.append(r2)
                        r3 = _comp(pesos_v, sem_v, "Pesos vs Sem", unid_d)
                        if r3 and r2 is None: comps.append(r3)

                        row['Resultado'] = "  |  ".join(comps) if comps else '— (N insuf.)'
                        treino_rows.append(row)

                    if treino_rows:
                        st.markdown(f"**{tipo_d} ({unid_d}) — impacto do treino anterior**")
                        st.dataframe(pd.DataFrame(treino_rows), width="stretch", hide_index=True)


def tab_corporal(dc, da_full):
    """
    Aba Composição Corporal & Nutrição.
    dc      : DataFrame Consolidado_Comida (pré-processado, todo o histórico)
    da_full : DataFrame atividades completo (para correlações)
    """
    st.header("🧬 Composição Corporal & Nutrição")

    if dc is None or len(dc) == 0:
        st.warning("Sem dados corporais. Verifica a aba 'Consolidado_Comida' na planilha.")
        return

    dc = dc.copy()
    dc['Data'] = pd.to_datetime(dc['Data'])

    # ── Limitar cada coluna até ao seu último registo válido ─────────────────
    _num_cols = ['Peso','BF','Calorias','Carb','Fat','Ptn','Net']
    for _col in _num_cols:
        if _col not in dc.columns: continue
        _last_valid = dc.loc[dc[_col].notna(), 'Data'].max()
        if pd.isna(_last_valid): continue
        dc.loc[dc['Data'] > _last_valid, _col] = np.nan
    _nc = [c for c in _num_cols if c in dc.columns]
    if _nc:
        dc = dc[dc[_nc].notna().any(axis=1)].copy()

    # ── KPIs cobertura ────────────────────────────────────────────────────────
    n_total = len(dc)
    n_peso  = dc['Peso'].notna().sum()     if 'Peso'     in dc.columns else 0
    n_cal   = dc['Calorias'].notna().sum() if 'Calorias' in dc.columns else 0
    n_bf    = dc['BF'].notna().sum()       if 'BF'       in dc.columns else 0
    d_min   = dc['Data'].min().strftime('%d/%m/%Y')
    d_max   = dc['Data'].max().strftime('%d/%m/%Y')
    n_dias  = (dc['Data'].max() - dc['Data'].min()).days + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Período",             f"{d_min} → {d_max}")
    c2.metric("⚖️ Peso (registos)",     f"{n_peso}/{n_total}")
    c3.metric("🔥 Calorias (registos)", f"{n_cal}/{n_total}")
    c4.metric("🫁 BF (registos)",       f"{n_bf}/{n_total}")
    st.caption(f"Dados esparsos: {n_dias} dias no período, {n_total} entradas. "
               "Dias sem registo são ignorados em médias e correlações.")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # 🎯 MINI-CALCULADORA — usa TODO o histórico independente de filtros
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("🎯 Calculadora de Metas — Peso, BF e Calorias", expanded=True):

        _dc_all = dc.copy()
        _dc_all['_w'] = _dc_all['Data'].dt.to_period('W')
        _wk = _dc_all.groupby('_w')[['Peso','BF','Calorias','Net']].mean()
        _wk.index = _wk.index.to_timestamp()
        _wk = _wk.sort_index()

        _peso_atual = (float(_wk['Peso'].dropna().tail(4).median())
                       if 'Peso' in _wk.columns and _wk['Peso'].notna().any() else None)
        _bf_atual   = (float(_wk['BF'].dropna().tail(4).median())
                       if 'BF'   in _wk.columns and _wk['BF'].notna().any()   else None)

        st.caption(
            "Valores actuais = **mediana das últimas 4 semanas** com dados disponíveis. "
            "Calorias estimadas pela relação histórica real Calorias ↔ Peso/BF.")

        _ci1, _ci2, _ci3 = st.columns(3)
        _peso_alvo = _ci1.number_input(
            f"⚖️ Peso-alvo (kg)  {'[actual: ' + str(round(_peso_atual,1)) + ' kg]' if _peso_atual else ''}",
            min_value=30.0, max_value=200.0,
            value=float(round(_peso_atual,1)) if _peso_atual else 75.0,
            step=0.1, key="calc_peso_alvo")
        _bf_alvo = _ci2.number_input(
            f"🫁 BF-alvo (%)  {'[actual: ' + str(round(_bf_atual,1)) + '%]' if _bf_atual else ''}",
            min_value=3.0, max_value=50.0,
            value=float(round(_bf_atual,1)) if _bf_atual else 15.0,
            step=0.1, key="calc_bf_alvo")
        _usar_peso = _ci3.checkbox("Calcular meta Peso", value=True,  key="calc_use_peso")
        _usar_bf   = _ci3.checkbox("Calcular meta BF",   value=False, key="calc_use_bf")

        if not _usar_peso and not _usar_bf:
            st.info("Selecciona pelo menos uma meta (Peso ou BF).")
        else:
            def _cal_historicas(target_col):
                df_p = _wk[[target_col, 'Calorias']].dropna()
                if len(df_p) < 8: return None
                sl, ic, rv, pv, _ = linregress(df_p[target_col].values,
                                                df_p['Calorias'].values)
                try:
                    df_p = df_p.copy()
                    df_p['_q'] = pd.qcut(df_p[target_col], q=4,
                                         labels=['Q1','Q2','Q3','Q4'],
                                         duplicates='drop')
                    qcal = df_p.groupby('_q', observed=True)['Calorias'].agg(
                        ['median','mean','count']).reset_index()
                    qtgt = df_p.groupby('_q', observed=True)[target_col].median()
                except Exception:
                    qcal, qtgt = None, None
                return {
                    'slope': sl, 'intercept': ic,
                    'r2': rv**2, 'pv': pv,
                    'cal_std': df_p['Calorias'].std(),
                    'cal_media': df_p['Calorias'].mean(),
                    'n': len(df_p),
                    'qcal': qcal, 'qtgt': qtgt,
                }

            for _var, _alvo, _atual, _usar in [
                ('Peso', _peso_alvo, _peso_atual, _usar_peso),
                ('BF',   _bf_alvo,   _bf_atual,   _usar_bf),
            ]:
                if not _usar or _atual is None: continue
                if _var not in _wk.columns: continue

                diff_val = round(_alvo - _atual, 2)
                if diff_val == 0:
                    st.success(f"✅ {_var}: já está no alvo ({_atual:.1f})!")
                    continue

                unid    = 'kg' if _var == 'Peso' else '%'
                dir_lbl = ('⬆️ ganhar ' if diff_val > 0 else '⬇️ perder ') + f"{abs(diff_val):.1f} {unid}"

                st.markdown(f"#### {_var} — actual: **{_atual:.1f}** → alvo: **{_alvo:.1f}** ({dir_lbl})")

                c_rel = _cal_historicas(_var)
                if c_rel:
                    qual = ("✅ Confiável" if c_rel['r2'] > 0.10 and c_rel['pv'] < 0.10
                            else "⚠️ Tendência fraca" if c_rel['n'] >= 8
                            else "⛔ Dados insuficientes")
                    st.caption(f"{qual} — R²={c_rel['r2']:.2f} | p={c_rel['pv']:.3f} | "
                               f"N={c_rel['n']} semanas | "
                               f"Relação: cada 1 {unid} de {_var} ↔ "
                               f"{c_rel['slope']:+.0f} kcal (histórico)")

                    cal_atual = c_rel['intercept'] + c_rel['slope'] * _atual
                    cal_alvo  = c_rel['intercept'] + c_rel['slope'] * _alvo
                    ajuste    = cal_alvo - cal_atual
                    dir_lbl2  = "défice" if diff_val < 0 else "superávit"

                    st.dataframe(pd.DataFrame([
                        {'Métrica': '📊 Cal. históricas associadas ao estado actual',
                         'Valor': f"{cal_atual:.0f} kcal"},
                        {'Métrica': f'{"➕" if diff_val > 0 else "➖"} Ajuste necessário ({dir_lbl2})',
                         'Valor': f"{ajuste:+.0f} kcal/dia"},
                        {'Métrica': '🎯 Cal. alvo — central (dos dados)',
                         'Valor': f"{cal_alvo:.0f} kcal"},
                        {'Métrica': '📉 Cal. alvo — mínimo (–1σ histórico)',
                         'Valor': f"{cal_alvo - c_rel['cal_std']:.0f} kcal"},
                        {'Métrica': '📈 Cal. alvo — máximo (+1σ histórico)',
                         'Valor': f"{cal_alvo + c_rel['cal_std']:.0f} kcal"},
                    ]), width="stretch", hide_index=True)

                    if c_rel['qcal'] is not None:
                        with st.expander(f"📊 Calorias históricas por quartil de {_var}"):
                            qrows = []
                            for _, qr in c_rel['qcal'].iterrows():
                                tgt_v = (c_rel['qtgt'].get(qr['_q'], float('nan'))
                                         if c_rel['qtgt'] is not None else float('nan'))
                                qrows.append({
                                    f'Quartil {_var}': str(qr['_q']),
                                    f'{_var} mediana': f"{tgt_v:.1f} {unid}" if not np.isnan(tgt_v) else '—',
                                    'Cal. mediana': f"{qr['median']:.0f} kcal",
                                    'Cal. média':   f"{qr['mean']:.0f} kcal",
                                    'N semanas':    int(qr['count']),
                                })
                            st.dataframe(pd.DataFrame(qrows),
                                         width="stretch", hide_index=True)
                else:
                    st.info(f"Sem dados suficientes de Calorias para estimar ({_var}).")

                st.markdown("---")

            st.caption(
                "⚠️ Calorias estimadas por regressão linear Calorias ~ Peso/BF sobre "
                "o histórico real. Não usa regras genéricas externas.")

    st.markdown("---")

    # ── Controlos: agrupamento + filtro de datas ──────────────────────────────
    st.subheader("⚙️ Filtros dos gráficos")
    _fc1, _fc2, _fc3, _fc4 = st.columns([1, 1, 1, 1])

    agrup_opts = {"Semana": "W", "Mês": "M", "Trimestre": "Q"}
    agrup_lbl  = _fc1.selectbox("Agrupar por", list(agrup_opts.keys()), key="corp_agrup")
    agrup_code = agrup_opts[agrup_lbl]
    roll_w     = _fc2.slider("Rolling (períodos)", 1, 12, 4, key="corp_roll")

    _d_min_hist = dc['Data'].min().date()
    _d_max_hist = dc['Data'].max().date()
    _tab_di = _fc3.date_input("Data início", value=_d_min_hist,
                               min_value=_d_min_hist, max_value=_d_max_hist,
                               key="corp_di")
    _tab_df = _fc4.date_input("Data fim", value=_d_max_hist,
                               min_value=_d_min_hist, max_value=_d_max_hist,
                               key="corp_df")

    dc_f = dc[(dc['Data'].dt.date >= _tab_di) & (dc['Data'].dt.date <= _tab_df)].copy()
    if len(dc_f) == 0:
        st.warning("Sem dados no período seleccionado.")
        return

    st.caption(f"📊 {len(dc_f)} registos | "
               f"{_tab_di.strftime('%d/%m/%Y')} → {_tab_df.strftime('%d/%m/%Y')}")

    # Agregação por período
    dc_f['_p'] = dc_f['Data'].dt.to_period(agrup_code)
    agg = dc_f.groupby('_p')[['Peso','BF','Calorias','Net','Carb','Fat','Ptn']].mean()
    agg.index = agg.index.to_timestamp()

    def _roll(series):
        return series.dropna().rolling(roll_w, min_periods=1).mean()

    # ── helper: formato tooltip de data por agrupamento ───────────────────────
    def _fmt_dates(idx):
        if agrup_code == 'W':
            return [d.strftime('Sem %d/%m/%y') for d in idx]
        elif agrup_code == 'M':
            return [d.strftime('%b %Y') for d in idx]
        else:
            return [f"Q{((d.month-1)//3)+1} {d.year}" for d in idx]

    # ── GRÁFICO 1: Peso + BF + Calorias ──────────────────────────────────────
    st.subheader("⚖️ Peso, % Gordura Corporal (BF) e Calorias")
    fig1 = go.Figure()

    if 'Peso' in agg.columns and agg['Peso'].notna().any():
        ps = agg['Peso'].dropna()
        pr = _roll(agg['Peso'])
        fig1.add_trace(go.Scatter(
            x=ps.index, y=ps.values,
            mode='markers', name='Peso (pontos)',
            marker=dict(color=CORES['azul'], size=6, opacity=0.4),
            hovertemplate='%{x|%d/%m/%Y}<br>Peso: <b>%{y:.1f} kg</b><extra></extra>',
            yaxis='y1'))
        fig1.add_trace(go.Scatter(
            x=pr.index, y=pr.values,
            mode='lines', name=f'Peso roll({roll_w})',
            line=dict(color=CORES['azul'], width=2.5),
            hovertemplate='%{x|%d/%m/%Y}<br>Peso roll: <b>%{y:.1f} kg</b><extra></extra>',
            yaxis='y1'))

    if 'BF' in agg.columns and agg['BF'].notna().any():
        bs = agg['BF'].dropna()
        br = _roll(agg['BF'])
        fig1.add_trace(go.Scatter(
            x=bs.index, y=bs.values,
            mode='markers', name='BF% (pontos)',
            marker=dict(color=CORES['vermelho'], size=6, opacity=0.4),
            hovertemplate='%{x|%d/%m/%Y}<br>BF: <b>%{y:.1f}%</b><extra></extra>',
            yaxis='y2'))
        fig1.add_trace(go.Scatter(
            x=br.index, y=br.values,
            mode='lines', name=f'BF roll({roll_w})',
            line=dict(color=CORES['vermelho'], width=2.5, dash='dash'),
            hovertemplate='%{x|%d/%m/%Y}<br>BF roll: <b>%{y:.1f}%</b><extra></extra>',
            yaxis='y2'))

    if 'Calorias' in agg.columns and agg['Calorias'].notna().any():
        cs = agg['Calorias'].dropna()
        cr = _roll(agg['Calorias'])
        fig1.add_trace(go.Bar(
            x=cs.index, y=cs.values,
            name='Calorias',
            marker=dict(color=CORES['laranja'], opacity=0.25),
            hovertemplate='%{x|%d/%m/%Y}<br>Calorias: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y3'))
        fig1.add_trace(go.Scatter(
            x=cr.index, y=cr.values,
            mode='lines', name=f'Cal roll({roll_w})',
            line=dict(color=CORES['laranja'], width=1.8, dash='dot'),
            hovertemplate='%{x|%d/%m/%Y}<br>Cal roll: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y3'))

    fig1.update_layout(
        title=f'Peso, BF e Calorias — {agrup_lbl} | rolling={roll_w}',
        height=420,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        yaxis=dict(title='Peso (kg)', title_font=dict(color=CORES['azul']),
                   tickfont=dict(color=CORES['azul'])),
        yaxis2=dict(title='BF (%)', title_font=dict(color=CORES['vermelho']),
                    tickfont=dict(color=CORES['vermelho']),
                    overlaying='y', side='right'),
        yaxis3=dict(title='Calorias (kcal)', title_font=dict(color=CORES['laranja']),
                    tickfont=dict(color=CORES['laranja']),
                    overlaying='y', side='right', anchor='free', position=1.0,
                    showgrid=False),
        xaxis=dict(showgrid=True),
        margin=dict(r=80),
    )
    st.plotly_chart(fig1, width="stretch")

    # ── GRÁFICO 2: Calorias + Net ─────────────────────────────────────────────
    st.subheader("🔥 Calorias e Balanço Energético (Net)")
    fig2 = go.Figure()

    if 'Calorias' in agg.columns and agg['Calorias'].notna().any():
        cs2 = agg['Calorias'].dropna()
        cr2 = _roll(agg['Calorias'])
        fig2.add_trace(go.Bar(
            x=cs2.index, y=cs2.values,
            name='Calorias',
            marker=dict(color=CORES['laranja'], opacity=0.4),
            hovertemplate='%{x|%d/%m/%Y}<br>Calorias: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y1'))
        fig2.add_trace(go.Scatter(
            x=cr2.index, y=cr2.values,
            mode='lines', name=f'Cal roll({roll_w})',
            line=dict(color=CORES['laranja'], width=2.5),
            hovertemplate='%{x|%d/%m/%Y}<br>Cal roll: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y1'))

    if 'Net' in agg.columns and agg['Net'].notna().any():
        nr = _roll(agg['Net'])
        fig2.add_trace(go.Scatter(
            x=nr.index, y=nr.values,
            mode='lines', name=f'Net roll({roll_w})',
            line=dict(color=CORES['roxo'], width=2.5, dash='dash'),
            hovertemplate='%{x|%d/%m/%Y}<br>Net: <b>%{y:.0f} kcal</b><extra></extra>',
            yaxis='y2'))
        fig2.add_hline(y=0, line_dash='dot', line_color=CORES['cinza'],
                       annotation_text='Net = 0', annotation_position='bottom right',
                       yref='y2')

    fig2.update_layout(
        title=f'Calorias e Net — {agrup_lbl} | rolling={roll_w}',
        height=380,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        yaxis=dict(title='Calorias (kcal)', title_font=dict(color=CORES['laranja']),
                   tickfont=dict(color=CORES['laranja'])),
        yaxis2=dict(title='Net (kcal)', title_font=dict(color=CORES['roxo']),
                    tickfont=dict(color=CORES['roxo']),
                    overlaying='y', side='right'),
    )
    st.plotly_chart(fig2, width="stretch")

    # ── GRÁFICO 3: Macros % stacked ───────────────────────────────────────────
    st.subheader("🥗 Distribuição de Macronutrientes (%)")
    macro_g = [c for c in ['Carb','Fat','Ptn'] if c in agg.columns]
    if macro_g:
        kcal_map = {'Carb': 4, 'Fat': 9, 'Ptn': 4}
        agg_k = agg[macro_g].copy()
        for m in macro_g:
            agg_k[m] = agg_k[m] * kcal_map.get(m, 4)
        total_k = agg_k[macro_g].sum(axis=1).replace(0, np.nan)
        agg_pct = (agg_k[macro_g].div(total_k, axis=0) * 100).dropna(how='all')

        if len(agg_pct) > 0:
            macro_cores = {'Carb': CORES['azul'], 'Fat': CORES['laranja'], 'Ptn': CORES['verde']}
            fig3 = go.Figure()
            for m in macro_g:
                vals = agg_pct[m].fillna(0)
                fig3.add_trace(go.Bar(
                    x=agg_pct.index, y=vals.values,
                    name=m,
                    marker=dict(color=macro_cores.get(m, CORES['cinza'])),
                    hovertemplate=f'%{{x|%d/%m/%Y}}<br>{m}: <b>%{{y:.1f}}%</b><extra></extra>',
                ))
            fig3.update_layout(
                barmode='stack',
                title=f'Macros % (Carb/Fat/Ptn em kcal) — {agrup_lbl}',
                height=360,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                yaxis=dict(title='% kcal de macros', range=[0, 115]),
            )
            fig3.add_hline(y=100, line_dash='dot', line_color=CORES['cinza'])
            st.plotly_chart(fig3, width="stretch")

    st.markdown("---")

    # ── GRÁFICO 4: Variação Peso e BF com bandas ─────────────────────────────
    st.subheader("📊 Variação de Peso e BF — bandas de ganho/perda esperado")
    st.caption("Peso (verde) e BF (azul) lado a lado. "
               "Bandas: limites calculados sobre o valor do período anterior. "
               "Peso: ±0.30%–0.70% | BF: ±0.25%–0.65%")

    dc_f2        = dc_f.copy()
    dc_f2['_p2'] = dc_f2['Data'].dt.to_period(agrup_code)
    agg_v        = dc_f2.groupby('_p2')[['Peso','BF']].mean()
    agg_v.index  = agg_v.index.to_timestamp()
    agg_v        = agg_v.sort_index()

    has_peso = 'Peso' in agg_v.columns and agg_v['Peso'].notna().sum() >= 3
    has_bf   = 'BF'   in agg_v.columns and agg_v['BF'].notna().sum()   >= 3

    if has_peso or has_bf:
        fig4 = go.Figure()

        if has_peso:
            peso_s     = agg_v['Peso'].dropna()
            peso_delta = peso_s.diff().dropna()
            prev_p     = peso_s.shift(1).dropna()
            xlbls      = _fmt_dates(peso_delta.index)

            fig4.add_trace(go.Bar(
                x=peso_delta.index, y=peso_delta.values,
                name='Δ Peso',
                marker=dict(color='#27ae60', opacity=0.85),
                width=1000*3600*24 * {'W':2,'M':8,'Q':20}[agrup_code],
                offset=-1000*3600*24 * {'W':1.2,'M':5,'Q':14}[agrup_code],
                customdata=np.stack([xlbls, peso_s.reindex(peso_delta.index).values], axis=-1),
                hovertemplate='%{customdata[0]}<br>Δ Peso: <b>%{y:+.2f} kg</b><br>Peso: %{customdata[1]:.1f} kg<extra></extra>',
                yaxis='y1'))

            for pct, col, lbl in [
                (0.0070, '#27ae60', '+max (+0.70%)'),
                (0.0030, '#82e0aa', '+min (+0.30%)'),
                (-0.0030, '#f1948a', '-min (-0.30%)'),
                (-0.0070, '#e74c3c', '-max (-0.70%)'),
            ]:
                fig4.add_trace(go.Scatter(
                    x=prev_p.index, y=prev_p.values * pct,
                    mode='lines', name=f'Peso {lbl}',
                    line=dict(color=col, width=1.5,
                              dash='dash' if abs(pct)==0.0070 else 'dot'),
                    hovertemplate=f'Limite Peso {lbl}: <b>%{{y:+.3f}} kg</b><extra></extra>',
                    yaxis='y1', showlegend=False))

            fig4.add_hline(y=0, line_color='black', line_width=0.8, opacity=0.5, yref='y')

        if has_bf:
            bf_s     = agg_v['BF'].dropna()
            bf_delta = bf_s.diff().dropna()
            prev_b   = bf_s.shift(1).dropna()
            xlbls_b  = _fmt_dates(bf_delta.index)

            fig4.add_trace(go.Bar(
                x=bf_delta.index, y=bf_delta.values,
                name='Δ BF',
                marker=dict(color='#2980b9', opacity=0.65),
                width=1000*3600*24 * {'W':2,'M':8,'Q':20}[agrup_code],
                offset=1000*3600*24 * {'W':0,'M':0,'Q':0}[agrup_code],
                customdata=np.stack([xlbls_b, bf_s.reindex(bf_delta.index).values], axis=-1),
                hovertemplate='%{customdata[0]}<br>Δ BF: <b>%{y:+.2f}%</b><br>BF: %{customdata[1]:.1f}%<extra></extra>',
                yaxis='y2'))

            for pct, col, lbl in [
                (0.0065, '#f39c12', '+max (+0.65%)'),
                (0.0025, '#fad7a0', '+min (+0.25%)'),
                (-0.0025, '#aed6f1', '-min (-0.25%)'),
                (-0.0065, '#2980b9', '-max (-0.65%)'),
            ]:
                fig4.add_trace(go.Scatter(
                    x=prev_b.index, y=prev_b.values * pct,
                    mode='lines', name=f'BF {lbl}',
                    line=dict(color=col, width=1.5,
                              dash='dash' if abs(pct)==0.0065 else 'dot'),
                    hovertemplate=f'Limite BF {lbl}: <b>%{{y:+.3f}}%</b><extra></extra>',
                    yaxis='y2', showlegend=False))

        fig4.update_layout(
            barmode='group',
            title=f'Variação Peso e BF por {agrup_lbl} — bandas de ganho/perda',
            height=450,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            yaxis=dict(title='Δ Peso (kg)', title_font=dict(color='#27ae60'),
                       tickfont=dict(color='#27ae60'), zeroline=True),
            yaxis2=dict(title='Δ BF (%)', title_font=dict(color='#2980b9'),
                        tickfont=dict(color='#2980b9'),
                        overlaying='y', side='right', zeroline=True),
        )
        st.plotly_chart(fig4, width="stretch")
    else:
        st.info("Dados insuficientes de Peso/BF para o gráfico de variação.")

    st.markdown("---")

    # ── CORRELAÇÕES ───────────────────────────────────────────────────────────
    st.subheader("🔗 Correlações entre variáveis corporais e de treino")
    st.caption(
        "Correlação de Spearman semanal sobre **todo o histórico**. "
        "Só mostra moderada (|r|≥0.40) ou forte (|r|≥0.60). MDC confirma variação real.")

    def _sem_mdc(series, icc=0.90):
        s = series.dropna()
        if len(s) < 5: return None, None
        sem = s.std(ddof=1) * np.sqrt(1 - icc)
        return sem, sem * 1.96 * np.sqrt(2)

    def _forca(r):
        a = abs(r)
        if a >= 0.60: return "★★★ Forte"
        if a >= 0.40: return "★★ Moderada"
        return None

    _dc_all2 = dc.copy()
    _dc_all2['_w'] = _dc_all2['Data'].dt.to_period('W')
    corp_agg = _dc_all2.groupby('_w')[['Peso','BF','Calorias','Net','Carb','Fat','Ptn']].mean()

    train_agg = pd.DataFrame()
    if da_full is not None and len(da_full) > 0:
        df_all = da_full.copy()
        df_all['Data'] = pd.to_datetime(df_all['Data'])
        df_all['_w']   = df_all['Data'].dt.to_period('W')
        df_all['_mt']  = pd.to_numeric(df_all['moving_time'], errors='coerce') / 3600
        df_cicl = df_all[df_all['type'].apply(norm_tipo) != 'WeightTraining'].copy()
        if 'icu_joules' in df_cicl.columns:
            df_cicl['_kj'] = pd.to_numeric(df_cicl['icu_joules'], errors='coerce') / 1000
        elif 'power_avg' in df_cicl.columns:
            df_cicl['_kj'] = (pd.to_numeric(df_cicl['power_avg'], errors='coerce') *
                              pd.to_numeric(df_cicl['moving_time'], errors='coerce') / 1000)
        else:
            df_cicl['_kj'] = np.nan
        df_cicl['_km'] = (pd.to_numeric(df_cicl.get('distance', pd.Series(dtype=float)),
                                         errors='coerce') / 1000)
        df_wt = df_all[df_all['type'].apply(norm_tipo) == 'WeightTraining'].copy()
        train_agg = df_cicl.groupby('_w').agg(
            Horas_cicl=('_mt', 'sum'), KJ_sem=('_kj', 'sum'), KM_sem=('_km', 'sum'))
        if len(df_wt) > 0:
            train_agg = train_agg.join(
                df_wt.groupby('_w').agg(Horas_WT=('_mt','sum')), how='outer')
        else:
            train_agg['Horas_WT'] = np.nan
        train_agg['Horas_total'] = (train_agg['Horas_cicl'].fillna(0) +
                                     train_agg['Horas_WT'].fillna(0))
        train_agg[train_agg == 0] = np.nan

    combined = (corp_agg.join(train_agg, how='outer')
                if len(train_agg) > 0 else corp_agg.copy())
    combined.index = combined.index.to_timestamp()
    combined = combined.sort_index()

    targets = [c for c in ['Peso','BF','Net','Calorias'] if c in combined.columns]
    predictors_all = [c for c in
        ['Calorias','Carb','Fat','Ptn','Net',
         'Horas_cicl','KJ_sem','KM_sem','Horas_WT','Horas_total','Peso','BF']
        if c in combined.columns]

    corr_rows = []
    for tgt in targets:
        _, mdc95 = _sem_mdc(combined[tgt])
        if mdc95 is not None:
            vr = combined[tgt].dropna()
            if (vr.max() - vr.min()) < mdc95: continue
        for pred in predictors_all:
            if pred == tgt: continue
            pair = combined[[tgt, pred]].dropna()
            if len(pair) < 8: continue
            r, pv = spearmanr(pair[pred].values, pair[tgt].values)
            if pv >= 0.10: continue
            f = _forca(r)
            if f is None: continue
            d0 = pair.index.min(); d1 = pair.index.max()
            corr_rows.append({
                'Alvo': tgt, 'Preditor': pred,
                'r': f"{r:+.2f}", 'p-value': f"{pv:.3f}",
                'N semanas': len(pair),
                'Período': f"{d0.strftime('%m/%Y')}→{d1.strftime('%m/%Y')}",
                'Força': f,
                'Efeito': (f"↗ {pred} ↑ → {tgt} ↑" if r > 0
                           else f"↘ {pred} ↑ → {tgt} ↓"),
            })

    if corr_rows:
        df_c = pd.DataFrame(corr_rows)
        df_c['_ar'] = df_c['r'].str.replace('+','',regex=False).astype(float).abs()
        df_c = (df_c.sort_values('_ar', ascending=False)
                    .drop_duplicates(subset=['Alvo','Preditor'], keep='first')
                    .drop(columns=['_ar'])
                    .sort_values(['Alvo','Força'], ascending=[True, True]))
        st.dataframe(df_c, width="stretch", hide_index=True)
    else:
        st.info("Sem correlações moderadas/fortes. Pode ser necessário mais semanas.")

    st.markdown("---")

    # ── TABELAS: base calórica por quartil de Peso e BF ───────────────────────
    st.subheader("📊 Base calórica por quartil de Peso e BF")
    st.caption("Semanas com dados simultâneos (todo o histórico). "
               "MDC± indica o erro mínimo detectável nas Calorias.")

    for alvo_q, alvo_lbl, unid in [('Peso','Peso','kg'), ('BF','BF','%')]:
        if alvo_q not in combined.columns or 'Calorias' not in combined.columns: continue
        pair_q = combined[[alvo_q,'Calorias']].dropna()
        if len(pair_q) < 8:
            st.caption(f"Poucos dados para quartis de {alvo_lbl} ({len(pair_q)} semanas).")
            continue
        pair_q = pair_q.copy()
        pair_q['_q'] = pd.qcut(pair_q[alvo_q], q=4,
                                labels=['Q1 (baixo)','Q2','Q3','Q4 (alto)'])
        rows_q = []
        for ql in ['Q1 (baixo)','Q2','Q3','Q4 (alto)']:
            g = pair_q[pair_q['_q'] == ql]
            if len(g) < 2: continue
            cv = g['Calorias']; av = g[alvo_q]
            _, mdc_c = _sem_mdc(cv)
            rows_q.append({
                f'Quartil {alvo_lbl}':        ql,
                f'Range {alvo_lbl} ({unid})': f"{av.min():.1f}–{av.max():.1f}",
                f'Média {alvo_lbl}':          f"{av.mean():.1f} {unid}",
                'N semanas':                  len(g),
                'Cal média':                  f"{cv.mean():.0f} kcal",
                'Cal mediana':                f"{cv.median():.0f} kcal",
                'Cal Q1–Q3':                  f"{cv.quantile(0.25):.0f}–{cv.quantile(0.75):.0f} kcal",
                'MDC Cal':                    f"±{mdc_c:.0f} kcal" if mdc_c else '—',
            })
        if rows_q:
            st.markdown(f"**Quartis de {alvo_lbl} — base calórica**")
            st.dataframe(pd.DataFrame(rows_q), width="stretch", hide_index=True)


# Configuração da página — DEVE ser a primeira chamada Streamlit
st.set_page_config(
    page_title="ATHELTICA",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

def render_sidebar():
    st.sidebar.image("https://img.icons8.com/emoji/96/runner-emoji.png", width=60)
    st.sidebar.title("ATHELTICA")
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Filtros Globais")

    dias_op = {"30 dias": 30, "60 dias": 60, "90 dias": 90,
               "180 dias": 180, "1 ano": 365, "2 anos": 730,
               "3 anos": 1095, "5 anos": 1825, "Todo histórico": 9999}
    periodo = st.sidebar.selectbox("📅 Período", list(dias_op.keys()), index=2)
    days_back = dias_op[periodo]

    usar_custom = st.sidebar.checkbox("📅 Datas manuais")
    if usar_custom:
        di  = st.sidebar.date_input("Início", datetime(2017, 1, 1).date())
        df_ = st.sidebar.date_input("Fim",    datetime.now().date())
        # days_back para carregar: diferença em dias + margem
        days_back = (df_ - di).days + 30
    else:
        df_ = datetime.now().date()
        di  = df_ - timedelta(days=min(days_back, 9999))

    st.sidebar.markdown("---")
    st.sidebar.header("🏃 Modalidades")
    mods_all = ['Bike', 'Row', 'Run', 'Ski']
    mods_sel = st.sidebar.multiselect("Mostrar modalidades", mods_all, default=mods_all)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Recarregar dados"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"📅 {di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}")
    st.sidebar.caption(f"🕐 Atualizado: {datetime.now().strftime('%H:%M')}")

    return days_back, di, df_, mods_sel

# ── Main ──────────────────────────────────────────────────────────────────────

# ── tab_padrao ──────────────────────────────────────────────
# tabs/tab_padrao.py — ATHELTICA Dashboard
# Aba Padrão — padrão semanal de actividade, RPE, Recovery Score, HRV/HR



# ── Helpers ───────────────────────────────────────────────────────────────────

DIAS_SEMANA = ['Seg','Ter','Qua','Qui','Sex','Sáb','Dom']
# Monday=0 ... Sunday=6
DIA_MAP = {0:'Seg',1:'Ter',2:'Qua',3:'Qui',4:'Sex',5:'Sáb',6:'Dom'}

RPE_ZONAS = {
    'Z1 (Leve)':   (1,   4.9),
    'Z2 (Mod.)':   (5,   6.9),
    'Z3 (Pesado)': (7,  10.0),
}

def _zona_rpe(v):
    try:
        v = float(v)
        if 1 <= v <= 4.9:  return 'Z1 (Leve)'
        if 5 <= v <= 6.9:  return 'Z2 (Mod.)'
        if 7 <= v <= 10:   return 'Z3 (Pesado)'
    except Exception:
        pass
    return None

def _periodos(da, dw):
    """
    Gera lista de (label, data_inicio, data_fim) para todos os períodos.
    Fixos: 15d, 30d, 90d, 180d
    Por ano: anos com dados, do mais recente para o mais antigo
    Todo histórico
    """
    hoje = pd.Timestamp.now().normalize()
    periodos = []

    for dias, lbl in [(15,'15 dias'),(30,'30 dias'),(90,'90 dias'),(180,'180 dias')]:
        periodos.append((lbl, hoje - pd.Timedelta(days=dias), hoje))

    # Anos com dados (actividades + wellness)
    anos_set = set()
    for df in [da, dw]:
        if df is not None and len(df) > 0 and 'Data' in df.columns:
            anos_set.update(pd.to_datetime(df['Data']).dt.year.unique())
    for ano in sorted(anos_set, reverse=True):
        ini = pd.Timestamp(ano, 1, 1)
        fim = min(pd.Timestamp(ano, 12, 31), hoje)
        periodos.append((str(ano), ini, fim))

    # Todo histórico
    todos = []
    for df in [da, dw]:
        if df is not None and len(df) > 0 and 'Data' in df.columns:
            todos.append(pd.to_datetime(df['Data']).min())
    if todos:
        periodos.append(('Todo histórico', min(todos), hoje))

    return periodos


def _filtrar(df, d_ini, d_fim):
    if df is None or len(df) == 0: return df
    d = df.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    return d[(d['Data'] >= d_ini) & (d['Data'] <= d_fim)]


def _cell_atividade(df_p, dia_num):
    """
    Para um dia da semana (0=Seg), retorna 'Tipo1 (X%) / Tipo2' ou '—'.
    Conta todas as datas desse dia da semana no período.
    """
    if df_p is None or len(df_p) == 0:
        return '—'
    d = df_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek
    sub = d[d['dow'] == dia_num]

    # Conta por data única (não por actividade — um dia pode ter 2 sessões)
    datas_unicas = sub['Data'].dt.normalize().unique()
    total_datas = len(pd.date_range(
        df_p['Data'].min(), df_p['Data'].max(),
        freq='W-' + ['MON','TUE','WED','THU','FRI','SAT','SUN'][dia_num]
    ))
    total_datas = max(total_datas, 1)

    if len(sub) == 0:
        return f'Rest (100%)'

    # Conta por tipo (inclui Rest para dias sem actividade)
    # Só actividades cíclicas (excluir WeightTraining)
    sub_cicl = sub[sub['type'].apply(norm_tipo) != 'WeightTraining']
    contagem = sub_cicl['type'].apply(norm_tipo).value_counts()

    # Dias sem actividade cíclica
    datas_com_ativ = sub_cicl['Data'].dt.normalize().nunique()
    datas_sem_ativ = total_datas - datas_com_ativ
    if datas_sem_ativ > 0:
        contagem['Rest'] = datas_sem_ativ

    total = contagem.sum()
    top = contagem.nlargest(2)
    itens = list(top.items())

    if len(itens) == 0:
        return '—'

    t1, n1 = itens[0]
    pct1 = int(round(n1 / total * 100))
    cell = f'{t1} ({pct1}%)'
    if len(itens) > 1:
        t2, _ = itens[1]
        cell += f' / {t2}'
    return cell


def _cell_rpe(df_p, dia_num):
    """
    Para um dia da semana, retorna a zona RPE mais frequente
    como padrão dominante (moda da zona por semana).
    """
    if df_p is None or len(df_p) == 0:
        return 'Rest'
    d = df_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek
    sub = d[d['dow'] == dia_num].copy()

    if len(sub) == 0 or 'rpe' not in sub.columns:
        return 'Rest'

    sub['_zona'] = pd.to_numeric(sub['rpe'], errors='coerce').apply(_zona_rpe)
    sub['_semana'] = sub['Data'].dt.to_period('W')

    # Para cada semana, qual foi a zona desse dia (moda se múltiplas sessões)
    padroes_semana = []
    for sem, grp in sub.groupby('_semana'):
        zonas = grp['_zona'].dropna()
        if len(zonas) == 0:
            padroes_semana.append('Rest')
        else:
            padroes_semana.append(zonas.mode().iloc[0])

    if not padroes_semana:
        return 'Rest'

    # Zona dominante ao longo das semanas
    from collections import Counter
    cnt = Counter(padroes_semana)
    return cnt.most_common(1)[0][0]


def _contagem_zonas_padrao(df_p):
    """
    Contagem de dias com cada zona dominante (pelo padrão semanal),
    mais a média de sessões Z3 por semana real.
    Retorna dict {zona: count, 'z3_por_semana': float}.
    """
    if df_p is None or len(df_p) == 0:
        return {}
    d = df_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek

    if 'rpe' not in d.columns:
        return {}

    d['_zona'] = pd.to_numeric(d['rpe'], errors='coerce').apply(_zona_rpe)

    # Padrão dominante por dia da semana
    contagem = {'Z1 (Leve)': 0, 'Z2 (Mod.)': 0, 'Z3 (Pesado)': 0, 'Rest': 0}
    for dia_num in range(7):
        zona_dom = _cell_rpe(d, dia_num)
        if zona_dom in contagem:
            contagem[zona_dom] += 1
        else:
            contagem['Rest'] += 1

    # Z3 por semana real: conta sessões Z3 por semana e tira a média
    d['_semana'] = d['Data'].dt.to_period('W')
    d_z3 = d[d['_zona'] == 'Z3 (Pesado)']
    n_semanas = d['_semana'].nunique()
    if n_semanas > 0:
        z3_por_sem = d_z3.groupby('_semana')['_zona'].count()
        media_z3 = z3_por_sem.reindex(d['_semana'].unique(), fill_value=0).mean()
    else:
        media_z3 = 0.0

    contagem['z3_por_semana'] = round(media_z3, 1)
    return contagem


def _cell_recovery(dw_p, dia_num):
    """
    Média ± DP do Recovery Score nesse dia da semana.
    """
    if dw_p is None or len(dw_p) == 0:
        return '—'
    d = dw_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek

    # Calcular recovery score
    rec = calcular_recovery(d)
    if len(rec) == 0:
        return '—'
    rec['Data'] = pd.to_datetime(rec['Data'])
    rec['dow'] = rec['Data'].dt.dayofweek
    sub = rec[rec['dow'] == dia_num]['recovery_score'].dropna()

    if len(sub) < 2:
        return f'{sub.mean():.0f}' if len(sub) == 1 else '—'
    return f'{sub.mean():.0f} ±{sub.std():.0f}'


def _cell_hrv_rhr(dw_p, dia_num, baseline_hrv, baseline_rhr, dp_hrv, dp_rhr):
    """
    Média de HRV e RHR nesse dia da semana.
    Cor: verde se dentro de ±0.5 DP do baseline, vermelho se fora.
    Retorna (texto_hrv, texto_rhr, cor_hrv, cor_rhr).
    """
    if dw_p is None or len(dw_p) == 0:
        return '—', '—', 'normal', 'normal'

    d = dw_p.copy()
    d['Data'] = pd.to_datetime(d['Data'])
    d['dow'] = d['Data'].dt.dayofweek
    sub = d[d['dow'] == dia_num]

    def _stats(col):
        if col not in sub.columns: return None, None
        vals = pd.to_numeric(sub[col], errors='coerce').dropna()
        if len(vals) == 0: return None, None
        return vals.mean(), vals.std()

    hrv_m, hrv_s = _stats('hrv')
    rhr_m, rhr_s = _stats('rhr')

    def _cor_hrv(m, base, dp):
        if m is None or base is None or dp is None or dp == 0: return 'normal'
        return 'green' if abs(m - base) <= 0.5 * dp else 'red'

    def _cor_rhr(m, base, dp):
        # RHR: mais alto = pior → invertido
        if m is None or base is None or dp is None or dp == 0: return 'normal'
        return 'green' if abs(m - base) <= 0.5 * dp else 'red'

    hrv_txt = f'{hrv_m:.0f}' if hrv_m else '—'
    rhr_txt = f'{rhr_m:.0f}' if rhr_m else '—'
    cor_hrv = _cor_hrv(hrv_m, baseline_hrv, dp_hrv)
    cor_rhr = _cor_rhr(rhr_m, baseline_rhr, dp_rhr)

    return hrv_txt, rhr_txt, cor_hrv, cor_rhr


def _render_hrv_rhr_table(rows_hrv):
    """
    Renderiza tabela HRV/RHR com cores usando HTML.
    rows_hrv: list of dicts com keys: Período, + dias da semana (cada com tuple (hrv,rhr,c_h,c_r))
    """
    cols = ['Período'] + DIAS_SEMANA

    def _cell_html(hrv_txt, rhr_txt, c_h, c_r):
        cor_h = '#27ae60' if c_h == 'green' else '#e74c3c' if c_h == 'red' else '#333333'
        cor_r = '#27ae60' if c_r == 'green' else '#e74c3c' if c_r == 'red' else '#333333'
        return (f'<span style="color:{cor_h};font-weight:bold">{hrv_txt}</span>'
                f'<span style="color:#666666"> | </span>'
                f'<span style="color:{cor_r};font-weight:bold">{rhr_txt}</span>')

    html = ('<table style="border-collapse:collapse;width:100%;font-size:12px;'
             'background:#ffffff;color:#222222">')
    # Header
    html += ('<tr style="background:#e8e8e8">')
    for c in cols:
        html += (f'<th style="border:1px solid #ccc;padding:6px 8px;'
                 f'text-align:center;color:#111111;font-weight:bold">{c}</th>')
    html += '</tr>'

    for i, row in enumerate(rows_hrv):
        bg = '#ffffff' if i % 2 == 0 else '#f9f9f9'
        html += f'<tr style="background:{bg}">'
        html += (f'<td style="border:1px solid #ccc;padding:5px 8px;'
                 f'font-weight:bold;white-space:nowrap;color:#111111">'
                 f'{row["Período"]}</td>')
        for dia in DIAS_SEMANA:
            val = row.get(dia, ('—','—','normal','normal'))
            if isinstance(val, tuple):
                h, r, ch, cr = val
                cell = _cell_html(h, r, ch, cr)
            else:
                cell = str(val)
            html += (f'<td style="border:1px solid #ccc;padding:5px 8px;'
                     f'text-align:center;color:#222222">{cell}</td>')
        html += '</tr>'
    html += '</table>'
    return html


# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

def tab_padrao(da_full, dw_full):
    """
    Aba Padrão — padrão semanal típico do atleta.
    da_full : DataFrame actividades completo (ac_full)
    dw_full : DataFrame wellness completo (wc — preproc_wellness sobre histórico total)
    """
    st.header("🔄 Padrão Semanal do Atleta")
    st.caption(
        "Análise do padrão típico por dia da semana em múltiplos períodos. "
        "Todos os períodos usam o histórico completo disponível.")

    if (da_full is None or len(da_full) == 0) and (dw_full is None or len(dw_full) == 0):
        st.warning("Sem dados disponíveis para análise de padrão.")
        return

    # Preparar dados
    da = da_full.copy() if da_full is not None and len(da_full) > 0 else pd.DataFrame()
    dw = dw_full.copy() if dw_full is not None and len(dw_full) > 0 else pd.DataFrame()
    if len(da) > 0:
        da['Data'] = pd.to_datetime(da['Data'])
    if len(dw) > 0:
        dw['Data'] = pd.to_datetime(dw['Data'])

    periodos = _periodos(da if len(da) > 0 else None,
                         dw if len(dw) > 0 else None)

    # ════════════════════════════════════════════════════════════════════
    # TABELA 1 — Padrão de actividade por dia da semana
    # ════════════════════════════════════════════════════════════════════
    st.subheader("🏃 Padrão de Actividade por Dia da Semana")
    st.caption("Top 1 com %, Top 2 sem %. 'Rest' quando maioria dos dias sem actividade.")

    if len(da) > 0:
        rows_ativ = []
        for lbl, d_ini, d_fim in periodos:
            da_p = _filtrar(da, d_ini, d_fim)
            if len(da_p) == 0:
                continue
            row = {'Período': lbl}
            for n, dia in enumerate(DIAS_SEMANA):
                row[dia] = _cell_atividade(da_p, n)
            rows_ativ.append(row)

        if rows_ativ:
            st.dataframe(pd.DataFrame(rows_ativ), width="stretch", hide_index=True)
        else:
            st.info("Sem dados de actividade suficientes.")
    else:
        st.info("Sem dados de actividade.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # TABELA 2 — Padrão de RPE por dia da semana + contagem de zonas
    # ════════════════════════════════════════════════════════════════════
    st.subheader("⚡ Padrão de RPE por Dia da Semana")
    st.caption(
        "Zona dominante por dia. "
        "Z1=RPE 1–5 | Z2=RPE 5–7 | Z3=RPE 7–10 | Rest=sem actividade. "
        "Contagem = nº de dias da semana com cada zona como dominante.")

    if len(da) > 0 and 'rpe' in da.columns:
        # RPE: usar só dados a partir de 2023
        da_rpe = da[pd.to_datetime(da['Data']).dt.year >= 2023].copy()
        rows_rpe = []
        for lbl, d_ini, d_fim in periodos:
            da_p = _filtrar(da_rpe, d_ini, d_fim)
            if len(da_p) == 0:
                continue
            row = {'Período': lbl}
            for n, dia in enumerate(DIAS_SEMANA):
                row[dia] = _cell_rpe(da_p, n)
            # Contagem de zonas pelo padrão + Z3 real por semana
            cnt = _contagem_zonas_padrao(da_p)
            row['Z1 dias'] = cnt.get('Z1 (Leve)', 0)
            row['Z2 dias'] = cnt.get('Z2 (Mod.)', 0)
            row['Z3 dias'] = cnt.get('Z3 (Pesado)', 0)
            row['Rest dias'] = cnt.get('Rest', 0)
            row['Z3/semana'] = cnt.get('z3_por_semana', 0.0)
            rows_rpe.append(row)

        if rows_rpe:
            st.dataframe(pd.DataFrame(rows_rpe), width="stretch", hide_index=True)
        else:
            st.info("Sem dados de RPE suficientes.")
    else:
        st.info("Sem dados de RPE.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # TABELA 3 — Padrão de Recovery Score por dia da semana
    # ════════════════════════════════════════════════════════════════════
    st.subheader("🔋 Padrão de Recovery Score por Dia da Semana")
    st.caption("Média ± DP do Recovery Score por dia da semana em cada período.")

    if len(dw) > 0:
        rows_rec = []
        for lbl, d_ini, d_fim in periodos:
            dw_p = _filtrar(dw, d_ini, d_fim)
            if len(dw_p) < 3:
                continue
            row = {'Período': lbl}
            for n, dia in enumerate(DIAS_SEMANA):
                row[dia] = _cell_recovery(dw_p, n)
            rows_rec.append(row)

        if rows_rec:
            st.dataframe(pd.DataFrame(rows_rec), width="stretch", hide_index=True)
        else:
            st.info("Sem dados de wellness suficientes.")
    else:
        st.info("Sem dados de wellness.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # TABELA 4 — Padrão HRV / RHR por dia da semana (com cores)
    # ════════════════════════════════════════════════════════════════════
    st.subheader("💚 Padrão HRV / RHR por Dia da Semana")
    st.caption(
        "Formato: **HRV | RHR**  "
        "🟢 verde = dentro de ±0.5 DP do baseline do período  "
        "🔴 vermelho = fora do baseline")

    if len(dw) > 0 and 'hrv' in dw.columns:
        rows_hrv = []
        for lbl, d_ini, d_fim in periodos:
            dw_p = _filtrar(dw, d_ini, d_fim)
            if len(dw_p) < 3:
                continue

            # Baseline = média e DP do período
            hrv_vals = pd.to_numeric(dw_p['hrv'], errors='coerce').dropna()
            rhr_vals  = (pd.to_numeric(dw_p['rhr'], errors='coerce').dropna()
                         if 'rhr' in dw_p.columns else pd.Series(dtype=float))

            if len(hrv_vals) == 0:
                continue

            base_hrv = hrv_vals.mean()
            dp_hrv   = hrv_vals.std()
            base_rhr = rhr_vals.mean() if len(rhr_vals) > 0 else None
            dp_rhr   = rhr_vals.std()  if len(rhr_vals) > 0 else None

            row = {'Período': lbl}
            for n, dia in enumerate(DIAS_SEMANA):
                h, r, ch, cr = _cell_hrv_rhr(dw_p, n,
                                               base_hrv, base_rhr,
                                               dp_hrv, dp_rhr)
                row[dia] = (h, r, ch, cr)
            rows_hrv.append(row)

        if rows_hrv:
            html = _render_hrv_rhr_table(rows_hrv)
            st.markdown(html, unsafe_allow_html=True)
            st.caption("HRV em ms | RHR em bpm")
        else:
            st.info("Sem dados de HRV suficientes.")
    else:
        st.info("Sem dados de HRV/RHR.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 13 — CTL vs KJ  (análise dTRIMP/dKJ)
# ════════════════════════════════════════════════════════════════════════════

def tab_ctl_kj(da_full):
    st.header("⚗️ CTL vs KJ — Coeficiente de Carga")
    st.caption(
        "Modelo comportamental: TRIMP ~ KJ_work × tipo. "
        "Bike: corrige warm-up (30min). IF removido (colinear com tipo). "
        "Densidade = IF×RPE. Eficiência = TRIMP/KJ rolling.")

    if da_full is None or len(da_full) == 0:
        st.warning("Sem dados de actividades.")
        return

    from scipy import stats as _scipy_stats

    # ── Preparar dados ────────────────────────────────────────────────────────
    df = filtrar_principais(da_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    for c in ['moving_time','icu_joules','power_avg','rpe','icu_eftp']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # KJ
    if 'icu_joules' in df.columns and df['icu_joules'].notna().any():
        df['KJ'] = df['icu_joules'] / 1000
        mask = df['KJ'].isna() | (df['KJ'] == 0)
        if 'power_avg' in df.columns:
            df.loc[mask,'KJ'] = (df['power_avg'] * df['moving_time'] / 1000)[mask]
    elif 'power_avg' in df.columns:
        df['KJ'] = df['power_avg'] * df['moving_time'] / 1000
    else:
        df['KJ'] = np.nan

    df['dur_min'] = df['moving_time'] / 60
    df['rpe_n']   = pd.to_numeric(df.get('rpe', np.nan), errors='coerce')

    # IF (apenas para densidade — não entra na regressão)
    if_cols = [c for c in df.columns if c.lower() in ('if','icu_if','intensity_factor')]
    if if_cols:
        df['IF'] = pd.to_numeric(df[if_cols[0]], errors='coerce')
    elif 'power_avg' in df.columns and 'icu_eftp' in df.columns:
        df['IF'] = (pd.to_numeric(df['power_avg'], errors='coerce') /
                    pd.to_numeric(df['icu_eftp'], errors='coerce').replace(0,np.nan)
                   ).clip(0.3, 1.8)
    else:
        df['IF'] = np.nan
    has_if = df['IF'].notna().sum() > len(df) * 0.2

    # ── Bike: correcção warm-up ───────────────────────────────────────────────
    df['work_fraction'] = ((df['moving_time'] - 1800) / df['moving_time']).clip(0.1, 1.0)
    df.loc[df['type'] != 'Bike', 'work_fraction'] = 1.0
    df['KJ_work'] = df['KJ'] * df['work_fraction']

    # ── Densidade = IF × RPE (proxy: tempo em alta intensidade) ──────────────
    if has_if:
        df['densidade'] = (df['IF'] * df['rpe_n']).clip(upper=15)
    else:
        df['densidade'] = df['rpe_n']   # fallback: só RPE

    # ── Tipo de sessão (RPE) — independente do IF ─────────────────────────────
    df['tipo'] = pd.cut(df['rpe_n'], bins=[0,5,7,10],
                        labels=['base','tempo','intervalado'], right=True)

    # ── TRIMP corrigido ───────────────────────────────────────────────────────
    if has_if:
        df['TRIMP_corr'] = df['dur_min'] * df['rpe_n'] * df['IF']
    else:
        df['TRIMP_corr'] = df['dur_min'] * df['rpe_n']

    # ── Eficiência por sessão: TRIMP / KJ_work ────────────────────────────────
    kj_ref = df['KJ_work'].where(df['KJ_work'] > 0, np.nan)
    df['eff_sessao'] = df['TRIMP_corr'] / kj_ref   # TRIMP por kJ

    # ── Filtros de qualidade ──────────────────────────────────────────────────
    n_raw = len(df)
    df = df[df['dur_min'] >= 10]
    df = df[df['rpe_n'].between(1, 10)]
    df = df[df['KJ'] > 0]
    if has_if:
        df = df[df['IF'].between(0.5, 1.5) | df['IF'].isna()]

    for col in ['KJ_work','TRIMP_corr']:
        q99 = df[col].quantile(0.99)
        df  = df[df[col] <= q99]

    st.info(f"Sessões: **{len(df)}** de {n_raw}. "
            f"IF disponível: {'✅' if has_if else '❌ — densidade usa só RPE'}")

    if len(df) < 20:
        st.warning("Dados insuficientes.")
        return

    # ── CTL/ATL real ──────────────────────────────────────────────────────────
    all_dates = pd.date_range(df['Data'].min(), pd.Timestamp.now().normalize(), freq='D')
    load_d    = df.groupby('Data')['TRIMP_corr'].sum().reindex(all_dates, fill_value=0)
    ctl_s     = load_d.ewm(span=42, adjust=False).mean()
    atl_s     = load_d.ewm(span=7,  adjust=False).mean()
    load_raw  = (df['dur_min']*df['rpe_n']).groupby(df['Data']).sum().reindex(all_dates, fill_value=0)
    ctl_raw_s = load_raw.ewm(span=42, adjust=False).mean()

    # Fadiga D-1
    ratio_s = (atl_s / ctl_s.replace(0, np.nan)).shift(1)
    df_ctx  = pd.DataFrame({'Data': all_dates, 'ATL_CTL_d1': ratio_s.values})
    df      = df.merge(df_ctx, on='Data', how='left')

    # ── 4 tabs internas ───────────────────────────────────────────────────────
    t_coef, t_eff, t_ctl, t_serie, t_debug = st.tabs([
        "📊 Coeficientes", "📈 Eficiência", "🎯 CTL pred vs real", "📉 Série CTL", "🔬 Debug"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — COEFICIENTES (modelo sem IF — TRIMP ~ KJ_work + tipo)
    # ════════════════════════════════════════════════════════════════════════
    with t_coef:
        st.subheader("dTRIMP/dKJ — Modelo: TRIMP ~ KJ_work + densidade")
        st.caption(
            "IF removido da regressão (colinear com tipo). "
            "Densidade (IF×RPE) adicionada como variável independente. "
            "Modelo: TRIMP ~ KJ_work + densidade por tipo de sessão.")

        rows_coef = []

        for mod in sorted(df['type'].unique()):
            dm = df[df['type'] == mod].copy()
            kj_col = 'KJ_work' if mod == 'Bike' else 'KJ'

            # RUN: modelo separado dur×RPE
            if mod == 'Run':
                dm_run = dm.dropna(subset=['rpe_n','dur_min'])
                if len(dm_run) >= 5:
                    x = dm_run['dur_min'].values
                    y = (dm_run['dur_min'] * dm_run['rpe_n']).values
                    sl,ic,r,p,se = _scipy_stats.linregress(x, y)
                    rows_coef.append({
                        'Modalidade':'Run','Tipo':'dur×RPE','Modelo':'dur→TRIMP',
                        'N':len(dm_run),'dTRIMP/dKJ':'—','coef_dens':'—',
                        'R²':round(r**2,3),'RMSE':round(np.sqrt(np.mean(((ic+sl*x)-y)**2)),1),
                        'Bias':round(np.mean((ic+sl*x)-y),2),
                        'KJ médio':'—','TRIMP médio':round(y.mean(),1),
                        'eff médio':'—',
                    })
                continue

            dm = dm.dropna(subset=[kj_col,'TRIMP_corr','tipo','densidade'])
            if len(dm) < 8: continue

            for tipo_seg in ['todos'] + list(dm['tipo'].dropna().unique()):
                ds = dm if tipo_seg == 'todos' else dm[dm['tipo']==tipo_seg]
                if len(ds) < 5: continue

                x_kj   = ds[kj_col].values
                x_dens = ds['densidade'].values
                y      = ds['TRIMP_corr'].values

                # OLS: TRIMP ~ intercept + KJ_work + densidade
                X = np.column_stack([np.ones(len(y)), x_kj, x_dens])
                try:
                    beta  = np.linalg.lstsq(X, y, rcond=None)[0]
                    ic_m, coef_kj, coef_dens = beta
                    y_pred = X @ beta
                    ss_res = np.sum((y - y_pred)**2)
                    ss_tot = np.sum((y - y.mean())**2)
                    r2     = max(0.0, 1 - ss_res/ss_tot) if ss_tot > 0 else 0
                    rmse   = np.sqrt(ss_res/len(y))
                    bias   = float(np.mean(y_pred - y))
                except Exception:
                    continue

                eff_m = float(ds['eff_sessao'].dropna().mean())
                rows_coef.append({
                    'Modalidade':  mod,
                    'Tipo':        str(tipo_seg),
                    'Modelo':      'KJ+dens',
                    'N':           len(ds),
                    'dTRIMP/dKJ':  round(coef_kj, 4),
                    'coef_dens':   round(coef_dens, 3),
                    'R²':          round(r2, 3),
                    'RMSE':        round(rmse, 1),
                    'Bias':        round(bias, 2),
                    'KJ médio':    f"{ds[kj_col].mean():.0f}kJ",
                    'TRIMP médio': round(ds['TRIMP_corr'].mean(), 1),
                    'eff médio':   round(eff_m, 3),
                })

        if rows_coef:
            df_coef = pd.DataFrame(rows_coef)

            def _color_coef(val):
                try:
                    v = float(val)
                    if   v < 0.3: return 'background-color:#d5f5e3'
                    elif v < 0.5: return 'background-color:#fef9e7'
                    else:         return 'background-color:#fdecea'
                except: return ''

            num_cols = [c for c in ['dTRIMP/dKJ'] if c in df_coef.columns]
            st.dataframe(df_coef.style.map(_color_coef, subset=num_cols),
                         width='stretch', hide_index=True)
            st.caption("🟢 dTRIMP/dKJ < 0.3 eficiente | 🟡 0.3–0.5 normal | 🔴 > 0.5 caro")

            csv_coef = df_coef.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download coeficientes CSV",
                               csv_coef, "dtrimp_dkj_v3.csv", "text/csv",
                               key="dl_coef_v3")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — EFICIÊNCIA FISIOLÓGICA ROLLING
    # ════════════════════════════════════════════════════════════════════════
    with t_eff:
        st.subheader("📈 Eficiência Fisiológica — TRIMP / KJ_work")
        st.caption(
            "eff = TRIMP / KJ_work por sessão. Rolling 4 semanas por modalidade. "
            "↑ Subindo = fadiga acumulada (mesmo kJ custa mais). "
            "↓ Descendo = adaptação (mesmo kJ custa menos).")

        mods_eff = [m for m in ['Bike','Row','Ski'] if m in df['type'].values]
        if not mods_eff:
            st.info("Sem dados suficientes.")
        else:
            fig_eff = go.Figure()
            CORES_MOD = {'Bike':'#e74c3c','Row':'#2980b9','Ski':'#27ae60','Run':'#8e44ad'}

            for mod in mods_eff:
                dm = df[df['type']==mod].dropna(subset=['eff_sessao']).copy()
                kj_col = 'KJ_work' if mod == 'Bike' else 'KJ'
                dm = dm[dm['eff_sessao'].between(
                    dm['eff_sessao'].quantile(0.05),
                    dm['eff_sessao'].quantile(0.95))]
                if len(dm) < 4: continue

                # Agrupar por semana e calcular mediana
                dm['_sem'] = dm['Data'].dt.to_period('W').dt.start_time
                eff_sem = (dm.groupby('_sem')['eff_sessao']
                           .median().reset_index()
                           .rename(columns={'_sem':'Data','eff_sessao':'eff'}))
                eff_sem = eff_sem.sort_values('Data')

                # Rolling 4 semanas
                eff_sem['eff_roll4'] = eff_sem['eff'].rolling(4, min_periods=2).mean()

                # Tendência recente (últimas 8 semanas)
                rec = eff_sem.tail(8)
                if len(rec) >= 4:
                    x_t = np.arange(len(rec))
                    sl_t,_,r_t,_,_ = _scipy_stats.linregress(x_t, rec['eff_roll4'].ffill())
                    tend = ('↑ fadiga' if sl_t > 0.005 else
                            '↓ adaptação' if sl_t < -0.005 else '→ estável')
                else:
                    tend = '—'

                fig_eff.add_trace(go.Scatter(
                    x=eff_sem['Data'], y=eff_sem['eff'],
                    name=f"{mod} (sessão)",
                    mode='markers',
                    marker=dict(color=CORES_MOD.get(mod,'#888'), size=4, opacity=0.4)))
                fig_eff.add_trace(go.Scatter(
                    x=eff_sem['Data'], y=eff_sem['eff_roll4'],
                    name=f"{mod} (4sem) {tend}",
                    line=dict(color=CORES_MOD.get(mod,'#888'), width=2.5)))

            fig_eff.update_layout(
                paper_bgcolor='white', plot_bgcolor='white', height=400,
                legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
                xaxis=dict(showgrid=True, gridcolor='#eee'),
                yaxis=dict(title='eff = TRIMP/KJ', showgrid=True, gridcolor='#eee'),
                title=dict(text='Eficiência Fisiológica (TRIMP/KJ) — rolling 4 semanas',
                           font=dict(size=13)))
            st.plotly_chart(fig_eff, use_container_width=True)

            # Tabela resumo de eficiência actual e tendência
            st.markdown("**Eficiência actual por modalidade × tipo:**")
            rows_eff = []
            for mod in mods_eff:
                dm = df[df['type']==mod].dropna(subset=['eff_sessao','tipo'])
                for tipo_seg in ['base','tempo','intervalado']:
                    ds = dm[dm['tipo']==tipo_seg]
                    if len(ds) < 3: continue
                    eff_all  = ds['eff_sessao'].median()
                    eff_rec  = ds[ds['Data'] >= ds['Data'].max()-pd.Timedelta(weeks=8)]['eff_sessao'].median()
                    delta    = eff_rec - eff_all
                    tend_sym = ('↑' if delta >  0.05 else '↓' if delta < -0.05 else '→')
                    rows_eff.append({
                        'Modalidade': mod,
                        'Tipo':       tipo_seg,
                        'eff hist':   round(eff_all, 3),
                        'eff 8sem':   round(eff_rec, 3),
                        'Δ':          f"{delta:+.3f}",
                        'Tendência':  tend_sym + (' fadiga' if delta>0.05 else
                                                   ' adaptação' if delta<-0.05 else ' estável'),
                    })
            if rows_eff:
                st.dataframe(pd.DataFrame(rows_eff), width='stretch', hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — CTL pred vs real
    # ════════════════════════════════════════════════════════════════════════
    with t_ctl:
        st.subheader("CTL predito vs real por modalidade")
        rows_met = []
        for mod in sorted(df['type'].unique()):
            dm = df[df['type']==mod].copy()
            kj_col = 'KJ_work' if mod == 'Bike' else 'KJ'

            if mod == 'Run':
                dm['TRIMP_pred_m'] = dm['dur_min'] * dm['rpe_n'].fillna(5)
            else:
                dm_ok = dm.dropna(subset=[kj_col,'TRIMP_corr','tipo','densidade'])
                if len(dm_ok) < 8: continue
                X_ok = np.column_stack([np.ones(len(dm_ok)),
                                        dm_ok[kj_col].values,
                                        dm_ok['densidade'].values])
                try:
                    beta = np.linalg.lstsq(X_ok, dm_ok['TRIMP_corr'].values, rcond=None)[0]
                except Exception:
                    continue
                dm['_kj_f']   = dm[kj_col].fillna(0)
                dm['_dens_f'] = dm['densidade'].fillna(dm['densidade'].median())
                dm['TRIMP_pred_m'] = (beta[0] + beta[1]*dm['_kj_f'] +
                                      beta[2]*dm['_dens_f']).clip(lower=0)

            pred_d = dm.groupby('Data')['TRIMP_pred_m'].sum().reindex(all_dates, fill_value=0)
            real_d = dm.groupby('Data')['TRIMP_corr'].sum().reindex(all_dates, fill_value=0)
            ctl_p  = pred_d.ewm(span=42, adjust=False).mean()
            ctl_r  = real_d.ewm(span=42, adjust=False).mean()
            diff   = ctl_p - ctl_r
            valid  = ctl_r > 0
            rows_met.append({
                'Modalidade':         mod,
                'CTL real':           round(float(ctl_r.iloc[-1]), 1),
                'CTL pred':           round(float(ctl_p.iloc[-1]), 1),
                'MAE':                round(float(diff[valid].abs().mean()), 2),
                'RMSE':               round(float(np.sqrt((diff[valid]**2).mean())), 2),
                'Bias':               round(float(diff[valid].mean()), 3),
                'Err%':               round(float((diff[valid].abs()/ctl_r[valid]*100).mean()), 1),
            })
        if rows_met:
            st.dataframe(pd.DataFrame(rows_met), width='stretch', hide_index=True)
            st.caption("Bias positivo = sobreestima. RMSE < 15 = bom ajuste.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — SÉRIE CTL
    # ════════════════════════════════════════════════════════════════════════
    with t_serie:
        st.subheader("Série CTL — raw vs IF corrigido")
        df_ts = pd.DataFrame({
            'Data':    all_dates,
            'CTL_raw': ctl_raw_s.values.round(2),
            'CTL_IF':  ctl_s.values.round(2),
            'ATL':     atl_s.values.round(2),
            'TSB':     (ctl_s - atl_s).values.round(2),
        })
        fig_ctl = go.Figure()
        fig_ctl.add_trace(go.Scatter(
            x=df_ts['Data'], y=df_ts['CTL_raw'],
            name='CTL raw (dur×RPE)',
            line=dict(color='#95a5a6', dash='dot', width=1.5)))
        fig_ctl.add_trace(go.Scatter(
            x=df_ts['Data'], y=df_ts['CTL_IF'],
            name='CTL IF corrigido',
            line=dict(color='#2980b9', width=2)))
        fig_ctl.add_trace(go.Scatter(
            x=df_ts['Data'], y=df_ts['ATL'],
            name='ATL (7d)',
            line=dict(color='#e74c3c', width=1.5)))
        fig_ctl.update_layout(
            paper_bgcolor='white', plot_bgcolor='white', height=380,
            legend=dict(orientation='h', y=-0.22, font=dict(color='#111')),
            xaxis=dict(showgrid=True, gridcolor='#eee'),
            yaxis=dict(title='Carga', showgrid=True, gridcolor='#eee'),
            title=dict(text='CTL raw vs IF corrigido', font=dict(size=13)))
        st.plotly_chart(fig_ctl, use_container_width=True)
        csv_ts = df_ts.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download série CTL (agrupada)", csv_ts,
                           "series_ctl_v3.csv", "text/csv", key="dl_ts_v3")

        st.markdown("---")
        st.subheader("⬇️ Export sessões — dados por actividade")
        st.caption(
            "date | modality | duration_min | kj_total | z1_kj | z2_kj | z3_kj | "
            "rpe | trimp | ctl | atl + power_avg | IF (opcionais)")

        # Construir DataFrame de sessões com CTL/ATL por sessão
        _df_exp = df.copy()
        _df_exp = _df_exp.sort_values("Data").reset_index(drop=True)

        # CTL/ATL no dia de cada sessão (join com série diária)
        _ctl_map = pd.Series(ctl_s.values,   index=all_dates).to_dict()
        _atl_map = pd.Series(atl_s.values,   index=all_dates).to_dict()
        _df_exp["ctl"] = _df_exp["Data"].map(_ctl_map).round(2)
        _df_exp["atl"] = _df_exp["Data"].map(_atl_map).round(2)

        # Colunas essenciais
        _exp_cols = {
            "date":         _df_exp["Data"].dt.strftime("%Y-%m-%d"),
            "modality":     _df_exp["type"].apply(norm_tipo),
            "duration_min": _df_exp["dur_min"].round(1),
            "kj_total":     _df_exp["KJ"].round(1),
            "z1_kj":        pd.to_numeric(_df_exp.get("z1_kj", np.nan), errors="coerce").round(1)
                            if "z1_kj" in _df_exp.columns else np.nan,
            "z2_kj":        pd.to_numeric(_df_exp.get("z2_kj", np.nan), errors="coerce").round(1)
                            if "z2_kj" in _df_exp.columns else np.nan,
            "z3_kj":        pd.to_numeric(_df_exp.get("z3_kj", np.nan), errors="coerce").round(1)
                            if "z3_kj" in _df_exp.columns else np.nan,
            "rpe":          pd.to_numeric(_df_exp["rpe_n"], errors="coerce").round(1),
            "trimp":        _df_exp["TRIMP_corr"].round(1),
            "ctl":          _df_exp["ctl"],
            "atl":          _df_exp["atl"],
        }

        # Opcionais
        if "power_avg" in _df_exp.columns:
            _exp_cols["power_avg"] = pd.to_numeric(_df_exp["power_avg"], errors="coerce").round(1)
        if "IF" in _df_exp.columns and _df_exp["IF"].notna().any():
            _exp_cols["if"] = _df_exp["IF"].round(3)

        _df_export = pd.DataFrame(_exp_cols)

        # Preview últimas 5 sessões
        st.dataframe(_df_export.tail(5), hide_index=True)
        st.caption(f"Total: {len(_df_export)} sessões | "
                   f"Período: {_df_export['date'].iloc[0]} → {_df_export['date'].iloc[-1]}")

        _csv_exp = _df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download sessões CSV",
            _csv_exp,
            "atheltica_sessions_export.csv",
            "text/csv",
            key="dl_sessions_exp")

    # ════════════════════════════════════════════════════════════════════════
    # TAB DEBUG — 5 testes visuais do sistema de sugestão
    # ════════════════════════════════════════════════════════════════════════
    with t_debug:
        st.subheader("🔬 Debug Visual — Sistema de Sugestão")
        st.caption(
            "Diagnóstico do comportamento dos 3 sinais: eff_delta, KJ/h, pwr_inc. "
            "Baseado nas sessões filtradas pelo modelo (pool por zona/RPE).")

        if len(df) < 10:
            st.warning("Dados insuficientes para debug.")
        else:
            # ── Calcular sinais por sessão ────────────────────────────────
            _debug_rows = []
            _MODS_DEBUG = [m for m in ['Bike','Row','Ski','Run'] if m in df['type'].values]

            for _mod in _MODS_DEBUG:
                _dm = df[df['type'].apply(norm_tipo) == _mod].copy()
                if len(_dm) < 5: continue
                _kj_col = 'KJ_work' if _mod == 'Bike' else 'KJ'

                # Pool completo desta modalidade (sem filtro de zona — ver evolução total)
                _dm2 = _dm.dropna(subset=[_kj_col, 'TRIMP_corr', 'rpe_n']).copy()
                _dm2 = _dm2.sort_values('Data')

                # eff por sessão
                _kj_w = _dm2[_kj_col].replace(0, np.nan)
                _dm2['_eff'] = _dm2['TRIMP_corr'] / _kj_w

                # KJ/h por sessão
                _dm2['_kjh'] = (_dm2[_kj_col] / (_dm2['dur_min'] / 60)).replace([np.inf, -np.inf], np.nan)

                # eff_delta rolling (eff_rec / eff_baseline - 1) — janela deslizante 8 semanas
                _dm2['_eff_roll']  = _dm2['_eff'].rolling(8, min_periods=3).mean()
                _dm2['_eff_base']  = _dm2['_eff'].rolling(40, min_periods=10).mean()
                _dm2['_eff_delta'] = (_dm2['_eff_roll'] / _dm2['_eff_base'] - 1).clip(-0.5, 0.5)
                # Garantir que _kjh_roll e _kjh_base existem sempre
                _dm2['_kjh_roll']  = _dm2['_kjh'].rolling(5,  min_periods=2).mean()
                _dm2['_kjh_base']  = _dm2['_kjh'].rolling(30, min_periods=8).mean()
                _dm2['_kjh_ratio'] = (_dm2['_kjh_roll'] / _dm2['_kjh_base']).clip(0.5, 1.5)

                # KJ/h rolling e baseline
                _dm2['_kjh_roll']  = _dm2['_kjh'].rolling(5, min_periods=2).mean()
                _dm2['_kjh_base']  = _dm2['_kjh'].rolling(30, min_periods=8).mean()
                _dm2['_kjh_ratio'] = (_dm2['_kjh_roll'] / _dm2['_kjh_base']).clip(0.5, 1.5)

                # pwr_inc simulado (base por zona)
                try:
                    _ni_mod = _ni_cache.get(_mod, 50)
                except NameError:
                    _ni_mod = 50
                if _ni_mod >= 75:    _pwr_base = 0.01
                elif _ni_mod >= 40:  _pwr_base = 0.02
                else:                _pwr_base = 0.02

                def _calc_pwr_inc(row):
                    ed = row['_eff_delta'] if pd.notna(row['_eff_delta']) else 0.0
                    kr = row['_kjh_ratio'] if pd.notna(row['_kjh_ratio']) else 1.0
                    p = _pwr_base
                    if   ed < -0.05: p *= 1.2
                    elif ed < 0.05:  p *= 1.0
                    elif ed < 0.12:  p *= 0.9
                    else:            p *= 0.7
                    if   kr >= 1.0:  p *= 1.1
                    elif kr >= 0.95: p *= 1.0
                    return round(p * 100, 2)  # em %

                _dm2['_pwr_inc_pct'] = _dm2.apply(_calc_pwr_inc, axis=1)
                _dm2['_mod'] = _mod
                # Garantir todas as colunas necessárias
                for _c in ['_kjh_roll','_kjh_base','_kjh_ratio','_eff_delta','_pwr_inc_pct']:
                    if _c not in _dm2.columns: _dm2[_c] = np.nan
                _debug_rows.append(_dm2[['Data','_mod','_eff','_eff_delta','_kjh',
                                          '_kjh_roll','_kjh_base','_kjh_ratio',
                                          '_pwr_inc_pct','TRIMP_corr',
                                          _kj_col, 'rpe_n']].copy())

            if not _debug_rows:
                st.info("Sem dados para debug.")
            else:
                _df_dbg = pd.concat(_debug_rows, ignore_index=True).sort_values('Data')
                _CORES_M = {'Bike':'#e74c3c','Row':'#2980b9','Ski':'#27ae60','Run':'#8e44ad'}

                # ── GRÁFICO 1: eff_delta ao longo do tempo ────────────────
                st.markdown("**Gráfico 1 — eff_delta (custo interno relativo ao baseline)**")
                st.caption(
                    "< −5%: adaptação | −5% a +5%: normal | "
                    "+5% a +12%: fadiga produtiva | > +12%: fadiga alta")

                fig_d1 = go.Figure()
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_eff_delta'])
                    if len(_s) < 3: continue
                    fig_d1.add_trace(go.Scatter(
                        x=_s['Data'], y=(_s['_eff_delta'] * 100).round(1),
                        mode='lines+markers', name=_mod,
                        line=dict(color=_CORES_M.get(_mod,'#888'), width=2),
                        marker=dict(size=4),
                        hovertemplate='%{x|%d/%m/%y}<br>eff_delta: <b>%{y:.1f}%%</b><extra></extra>'))
                # Zonas de fundo
                for y0, y1, col, lbl in [
                    (-50, -5, 'rgba(39,174,96,0.08)',  'adaptação'),
                    (-5,   5, 'rgba(52,152,219,0.06)', 'normal'),
                    (5,   12, 'rgba(243,156,18,0.10)', 'fadiga prod.'),
                    (12,  50, 'rgba(231,76,60,0.10)',  'fadiga alta'),
                ]:
                    fig_d1.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0,
                                     annotation_text=lbl, annotation_position='top left',
                                     annotation_font=dict(size=10, color='#555'))
                fig_d1.add_hline(y=0, line_dash='dash', line_color='#aaa', line_width=1)
                fig_d1.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white', height=320,
                    legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
                    xaxis=dict(showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='eff_delta (%)', showgrid=True, gridcolor='#eee'),
                    title=dict(text='eff_delta — custo interno relativo ao baseline', font=dict(size=12)))
                st.plotly_chart(fig_d1, use_container_width=True)

                # ── GRÁFICO 2: KJ/h ao longo do tempo ────────────────────
                st.markdown("**Gráfico 2 — KJ/h (densidade de trabalho)**")
                st.caption("Rolling 5 sessões vs baseline 30 sessões. Ratio abaixo de 1.0 → A reduzida.")

                fig_d2 = go.Figure()
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_kjh_roll'])
                    if len(_s) < 3: continue
                    fig_d2.add_trace(go.Scatter(
                        x=_s['Data'], y=_s['_kjh_roll'].round(0),
                        mode='lines', name=f'{_mod} roll',
                        line=dict(color=_CORES_M.get(_mod,'#888'), width=2.5),
                        hovertemplate='%{x|%d/%m/%y}<br>KJ/h: <b>%{y:.0f}</b><extra></extra>'))
                    fig_d2.add_trace(go.Scatter(
                        x=_s['Data'], y=_s['_kjh_base'].round(0),
                        mode='lines', name=f'{_mod} baseline',
                        line=dict(color=_CORES_M.get(_mod,'#888'), width=1, dash='dot'),
                        hovertemplate='%{x|%d/%m/%y}<br>KJ/h baseline: <b>%{y:.0f}</b><extra></extra>'))
                fig_d2.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white', height=320,
                    legend=dict(orientation='h', y=-0.30, font=dict(color='#111')),
                    xaxis=dict(showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='KJ/h', showgrid=True, gridcolor='#eee'),
                    title=dict(text='KJ/h rolling vs baseline por modalidade', font=dict(size=12)))
                st.plotly_chart(fig_d2, use_container_width=True)

                # ── GRÁFICO 3: pwr_inc simulado ───────────────────────────
                st.markdown("**Gráfico 3 — pwr_inc simulado (%)**")
                st.caption("Incremento de power que seria sugerido com base nos 3 sinais naquele dia.")

                fig_d3 = go.Figure()
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_pwr_inc_pct'])
                    if len(_s) < 3: continue
                    fig_d3.add_trace(go.Scatter(
                        x=_s['Data'], y=_s['_pwr_inc_pct'],
                        mode='lines+markers', name=_mod,
                        line=dict(color=_CORES_M.get(_mod,'#888'), width=2),
                        marker=dict(size=4),
                        hovertemplate='%{x|%d/%m/%y}<br>pwr_inc: <b>%{y:.2f}%%</b><extra></extra>'))
                fig_d3.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white', height=300,
                    legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
                    xaxis=dict(showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='pwr_inc (%)', showgrid=True, gridcolor='#eee'),
                    title=dict(text='pwr_inc simulado por sessão', font=dict(size=12)))
                st.plotly_chart(fig_d3, use_container_width=True)

                st.markdown("---")

                # ── TESTE 1: log eff_delta / kjh_ratio / pwr_inc ─────────
                st.markdown("**Teste 1 — Log: eff_delta | kjh_ratio | pwr_inc por sessão**")
                _cols_log = ['Data','_mod','rpe_n','_eff_delta','_kjh_ratio','_pwr_inc_pct']
                _df_log = _df_dbg[_cols_log].dropna().tail(50).copy()
                _df_log.columns = ['Data','Modalidade','RPE','eff_delta%','kjh_ratio','pwr_inc%']
                _df_log['eff_delta%'] = (_df_log['eff_delta%'] * 100).round(1)
                _df_log['kjh_ratio']  = _df_log['kjh_ratio'].round(3)
                _df_log['Data'] = _df_log['Data'].dt.strftime('%d/%m/%y')
                st.dataframe(_df_log.iloc[::-1], hide_index=True, width="stretch")

                st.markdown("---")

                # ── TESTE 2: A vs B dominância ────────────────────────────
                st.markdown("**Teste 2 — Dominância A vs B por modalidade**")
                st.caption("Simulação: quando B (intensidade) seria maior que A (volume)?")

                _rows_ab = []
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(
                        subset=['_eff_delta','_kjh_ratio','_pwr_inc_pct'])
                    if len(_s) < 3: continue

                    _kj_col = 'KJ_work' if _mod == 'Bike' else 'KJ'
                    _n_total = len(_s)

                    # B = mesmo tempo, mais power → KJ_B = ref_kj * (1 + pwr_inc/100)
                    # A = mais tempo, mesmo power → limitado por cap
                    # Simular: B > A quando pwr_inc efectivo suficiente
                    _n_b_dom = (_s['_pwr_inc_pct'] >= 1.0).sum()   # B produz ganho real
                    _n_a_dom = (_s['_kjh_ratio']   <  0.95).sum()  # A reduzida
                    _n_normal = _n_total - _n_b_dom - _n_a_dom + min(_n_b_dom, _n_a_dom)

                    _rows_ab.append({
                        'Modalidade':  _mod,
                        'N sessões':   _n_total,
                        'B dominante (pwr≥1%)':  f"{_n_b_dom} ({_n_b_dom/_n_total*100:.0f}%)",
                        'A reduzida (kjh<95%)':  f"{_n_a_dom} ({_n_a_dom/_n_total*100:.0f}%)",
                        'pwr_inc médio':         f"{_s['_pwr_inc_pct'].mean():.2f}%",
                        'pwr_inc max':           f"{_s['_pwr_inc_pct'].max():.2f}%",
                        'pwr_inc min':           f"{_s['_pwr_inc_pct'].min():.2f}%",
                    })
                if _rows_ab:
                    st.dataframe(pd.DataFrame(_rows_ab), hide_index=True, width="stretch")

                st.markdown("---")

                # ── TESTE 3: KJ/h rolling 7–14 dias ─────────────────────
                st.markdown("**Teste 3 — KJ/h rolling 7 e 14 sessões**")
                _rows_kjh = []
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_kjh']).copy()
                    if len(_s) < 5: continue
                    _s = _s.sort_values('Data')
                    _kjh7  = _s['_kjh'].rolling(7,  min_periods=3).mean().iloc[-1]
                    _kjh14 = _s['_kjh'].rolling(14, min_periods=5).mean().iloc[-1]
                    _kjh_all = _s['_kjh'].mean()
                    _rows_kjh.append({
                        'Modalidade':    _mod,
                        'KJ/h histórico': round(_kjh_all, 0),
                        'KJ/h roll 7':   round(_kjh7, 0),
                        'KJ/h roll 14':  round(_kjh14, 0),
                        'Ratio 7/hist':  f"{_kjh7/_kjh_all:.2f}" if _kjh_all > 0 else '—',
                        'Ratio 14/hist': f"{_kjh14/_kjh_all:.2f}" if _kjh_all > 0 else '—',
                    })
                if _rows_kjh:
                    st.dataframe(pd.DataFrame(_rows_kjh), hide_index=True, width="stretch")

                st.markdown("---")

                # ── TESTE 4: eff_delta vs pwr_inc scatter ─────────────────
                st.markdown("**Teste 4 — eff_delta vs pwr_inc (fadiga vs resposta)**")
                st.caption("Esperado: correlação negativa — mais fadiga → menor pwr_inc.")

                fig_d4 = go.Figure()
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(
                        subset=['_eff_delta','_pwr_inc_pct'])
                    if len(_s) < 5: continue
                    fig_d4.add_trace(go.Scatter(
                        x=(_s['_eff_delta'] * 100).round(1),
                        y=_s['_pwr_inc_pct'],
                        mode='markers', name=_mod,
                        marker=dict(color=_CORES_M.get(_mod,'#888'), size=6, opacity=0.6),
                        hovertemplate=f'{_mod}<br>eff_delta: <b>%{{x:.1f}}%%</b><br>pwr_inc: <b>%{{y:.2f}}%%</b><extra></extra>'))
                # Linhas verticais das zonas
                for xv, col in [(-5,'#27ae60'),(5,'#f39c12'),(12,'#e74c3c')]:
                    fig_d4.add_vline(x=xv, line_dash='dot', line_color=col, line_width=1.5)
                fig_d4.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white', height=320,
                    legend=dict(orientation='h', y=-0.25, font=dict(color='#111')),
                    xaxis=dict(title='eff_delta (%)', showgrid=True, gridcolor='#eee', zeroline=True),
                    yaxis=dict(title='pwr_inc (%)', showgrid=True, gridcolor='#eee'),
                    title=dict(text='eff_delta vs pwr_inc — fadiga vs resposta', font=dict(size=12)))
                st.plotly_chart(fig_d4, use_container_width=True)

                st.markdown("---")

                # ── TESTE 5: prioridade vs pwr_inc ───────────────────────
                st.markdown("**Teste 5 — Prioridade vs pwr_inc médio**")
                st.caption(
                    "Esperado: modalidades P1/P2 (Foco) com ni mais alto → "
                    "pwr_inc maior que P3/P4 (Manutenção).")

                _rows_prio = []
                for _mod in _MODS_DEBUG:
                    _s = _df_dbg[_df_dbg['_mod'] == _mod].dropna(subset=['_pwr_inc_pct'])
                    if len(_s) < 3: continue
                    try:
                        _ni_m  = _ni_cache.get(_mod, 50)
                        _ol_m  = _ol_cache.get(_mod, False)
                        _need  = _need_cache.get(_mod, 40)
                    except NameError:
                        _ni_m, _ol_m, _need = 50, False, 40
                    _p1 = st.session_state.get('vg_prio1','Bike')
                    _p2 = st.session_state.get('vg_prio2','Row')
                    _grupo = '🎯 Foco' if _mod in {_p1, _p2} else '🔧 Manutenção'
                    _rows_prio.append({
                        'Modalidade':   _mod,
                        'Grupo':        _grupo,
                        'Need_int':     _ni_m,
                        'Need_score':   round(_need, 1),
                        'Overload':     '⚠️' if _ol_m else '—',
                        'pwr_inc médio': f"{_s['_pwr_inc_pct'].mean():.2f}%",
                        'pwr_inc últimas 5': f"{_s['_pwr_inc_pct'].tail(5).mean():.2f}%",
                    })
                if _rows_prio:
                    _rows_prio.sort(key=lambda x: float(x['pwr_inc médio'].replace('%','')),
                                    reverse=True)
                    st.dataframe(pd.DataFrame(_rows_prio), hide_index=True, width="stretch")
                    st.caption(
                        "pwr_inc é determinado por eff_delta e kjh_ratio (dados históricos), "
                        "NÃO directamente pela prioridade. "
                        "A prioridade influencia Need_int → zona alvo → pwr_inc base.")

                # Download debug CSV
                st.markdown("---")
                _df_dbg_dl = _df_dbg.copy()
                _df_dbg_dl['Data'] = _df_dbg_dl['Data'].dt.strftime('%Y-%m-%d')
                _df_dbg_dl['_eff_delta_pct'] = (_df_dbg_dl['_eff_delta'] * 100).round(2)
                _df_dbg_dl = _df_dbg_dl.rename(columns={
                    '_mod':'modality','_eff':'eff','_eff_delta_pct':'eff_delta_pct',
                    '_kjh':'kjh','_kjh_ratio':'kjh_ratio','_pwr_inc_pct':'pwr_inc_pct',
                    'rpe_n':'rpe','TRIMP_corr':'trimp'})
                _cols_dl = [c for c in ['Data','modality','rpe','eff','eff_delta_pct',
                                         'kjh','kjh_ratio','pwr_inc_pct','trimp']
                            if c in _df_dbg_dl.columns]
                _csv_dbg = _df_dbg_dl[_cols_dl].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download debug CSV",
                    _csv_dbg, "atheltica_debug_signals.csv",
                    "text/csv", key="dl_dbg_signals")


def main():
    days_back, di, df_, mods_sel = render_sidebar()

    st.title("🏃 ATHELTICA Analytics Dashboard")
    st.caption(f"Período: {di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}  |  "
               f"Modalidades: {', '.join(mods_sel)}")

    # ── Carregamento de dados ────────────────────────────────────────────────
    with st.spinner("A carregar dados..."):
        wr           = carregar_wellness(days_back)
        # ar_max: SEMPRE histórico completo (desde 2017) — para tabelas eFTP/KM/Correlações
        # e para CTL/ATL convergirem correctamente. Cached, não recarrega se não mudar.
        ar_max       = carregar_atividades(9999)
        # ar: período seleccionado pelo sidebar (para gráficos e análises filtradas)
        ar           = carregar_atividades(days_back) if days_back < 9999 else ar_max
        dfs_annual, df_annual = carregar_annual()
        dc                    = carregar_corporal()

    if wr.empty and ac_full.empty:
        st.error("Não foi possível carregar dados. Verifica as credenciais e os URLs.")
        st.stop()

    # ── Preprocessing ────────────────────────────────────────────────────────
    wc       = preproc_wellness(wr)
    ac_full  = preproc_ativ(ar_max)    # histórico completo (9999d) para tabelas e PMC
    ac       = preproc_ativ(ar)        # período seleccionado no sidebar

    # Guardar histórico completo no session_state (usado pelo tab_pmc)
    st.session_state['da_full'] = ac_full
    st.session_state['mods_sel'] = mods_sel  # para PMC respeitar filtro de modalidades

    dw      = filtrar_datas(wc, di, df_)
    da      = filtrar_datas(ac, di, df_)
    da_filt = (da[da['type'].isin(mods_sel + ['WeightTraining'])]
               if len(da) > 0 and 'type' in da.columns else da)

    st.success(f"✅ {len(dw)} registros wellness  |  "
               f"{len(da_filt)} atividades ({di.strftime('%d/%m/%y')}→{df_.strftime('%d/%m/%y')})  |  "
               f"Histórico PMC: {len(ac_full)} atividades")

    # ── Diagnóstico (expansível) ─────────────────────────────────────────────
    with st.expander("🔍 Diagnóstico de dados", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Atividades (RAW — período filtrado)**")
            if len(ar) > 0:
                ar2 = ar.copy()
                ar2['Data'] = ar2['Data'].astype(str)
                cs = [c for c in ['Data', 'type', 'name', 'moving_time', 'rpe', 'power_avg', 'icu_eftp'] if c in ar2.columns]
                st.write(f"Total: {len(ar)} | Datas: {ar2['Data'].min()[:10]} → {ar2['Data'].max()[:10]}")
                st.dataframe(ar2[cs].sort_values('Data', ascending=False).head(5), hide_index=True)
        with c2:
            st.markdown("**Wellness (RAW)**")
            if len(wr) > 0:
                wr2 = wr.copy()
                wr2['Data'] = wr2['Data'].astype(str)
                cw = [c for c in ['Data', 'hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress'] if c in wr2.columns]
                st.write(f"Total: {len(wr)} | Datas: {wr2['Data'].min()[:10]} → {wr2['Data'].max()[:10]}")
                st.dataframe(wr2[cw].sort_values('Data', ascending=False).head(5), hide_index=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
        "📊 Visão Geral",
        "📈 PMC",
        "📦 Volume",
        "⚡ eFTP",
        "❤️ HR & RPE",
        "🧠 Correlações",
        "🔋 Recovery",
        "🧘 Wellness",
        "🔬 Análises",
        "🌡️ Aquecimento",
        "🧬 Corporal",
        "🔄 Padrão",
        "⚗️ CTL vs KJ",
    ])

    with tab1:  tab_visao_geral(dw, da_filt, di, df_, da_full=ac_full, wc_full=wc, dc=dc)
    with tab2:  tab_pmc(da_filt)
    with tab3:  tab_volume(da_filt, dw)
    with tab4:  tab_eftp(da_filt, mods_sel, ac_full)
    with tab5:  tab_zones(da_filt, mods_sel)
    with tab6:  tab_correlacoes(da_filt, dw)
    with tab7:  tab_recovery(dw)
    with tab8:  tab_wellness(dw)
    with tab9:  tab_analises(ac_full, dw, dfs_annual, df_annual)
    with tab10: tab_aquecimento(dfs_annual, df_annual, di)
    with tab11: tab_corporal(dc, ac_full)
    with tab12: tab_padrao(ac_full, wc)
    with tab13: tab_ctl_kj(ac_full)

if __name__ == "__main__":
    main()
