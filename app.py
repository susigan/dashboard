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
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from scipy import stats as scipy_stats
from scipy.stats import pearsonr, linregress, spearmanr, theilslopes
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
    'VirtualRun': 'Run', 'Running': 'Run', 'Run': 'Run', 'TrailRun': 'Run',
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

def analisar_falta_estimulo(df_act_full, janela_dias=14):
    """Need Score por modalidade — quanto estímulo está em falta."""
    ld, _ = calcular_series_carga(df_act_full)
    if len(ld) == 0: return None
    df = filtrar_principais(df_act_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    data_lim = pd.Timestamp.now() - pd.Timedelta(days=janela_dias)
    cr = ld[ld['Data'] >= data_lim].copy()
    if len(cr) == 0: return None
    res = {}
    for mod in ['Bike', 'Run', 'Row', 'Ski']:
        dm = df[(df['type'] == mod) & (df['Data'] >= data_lim)]
        dias_a = dm['Data'].nunique(); freq = dias_a / max(janela_dias, 1)
        atl_m = cr['ATL'].mean(); ctl_m = cr['CTL'].mean()
        gap = ((ctl_m - atl_m) / ctl_m * 100) if ctl_m > 0 else 0
        dias_b = (cr['ATL'] < cr['CTL']).sum()
        sl = np.polyfit(np.arange(len(cr)), cr['ATL'].values, 1)[0] if len(cr) > 1 else 0
        sl_n = max(0, min(1, (sl + 5) / 10))
        need = (min(1, max(0, gap / 50)) * 100 * 0.4 +
                min(1, dias_b / max(len(cr), 1)) * 100 * 0.3 +
                (1 - sl_n) * 100 * 0.2 +
                (1 - freq) * 100 * 0.1)
        prio = 'ALTA' if need >= 70 else 'MÉDIA' if need >= 40 else 'BAIXA'
        res[mod] = {'need_score': need, 'prioridade': prio, 'gap_relativo': gap,
                    'dias_atl_menor_ctl': int(dias_b), 'dias_com_atividade': dias_a,
                    'atl_medio': atl_m, 'ctl_medio': ctl_m}
    return dict(sorted(res.items(), key=lambda x: x[1]['need_score'], reverse=True))



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
    # [5] Zeros → NaN
    df = remove_zeros(df, ['moving_time', 'icu_eftp', 'AllWorkFTP'])
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


def tab_visao_geral(dw, da, di, df_):
    st.header("📊 Visão Geral")
    c1, c2, c3, c4 = st.columns(4)
    horas = (da['moving_time'].sum() / 3600) if 'moving_time' in da.columns and len(da) > 0 else None
    hrv_m = dw['hrv'].dropna().tail(7).mean() if 'hrv' in dw.columns and len(dw) > 0 else None
    rhr_u = dw['rhr'].dropna().iloc[-1] if 'rhr' in dw.columns and len(dw) > 0 and dw['rhr'].notna().any() else None
    c1.metric("🏋️ Sessões", f"{len(da)}")
    c2.metric("⏱️ Horas", f"{horas:.1f}h" if horas else "—")
    c3.metric("💚 HRV (7d)", f"{hrv_m:.0f} ms" if hrv_m else "—")
    c4.metric("❤️ RHR", f"{rhr_u:.0f} bpm" if rhr_u else "—")
    st.markdown("---")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("📈 Performance Overview")
        fig, ax = plt.subplots(figsize=(13, 4))
        if 'moving_time' in da.columns and 'rpe' in da.columns and len(da) > 0:
            dl = da.copy(); dl['Data'] = pd.to_datetime(dl['Data'])
            dl['load'] = (dl['moving_time'] / 60) * dl['rpe'].fillna(0)
            ld = dl.groupby('Data')['load'].sum().reset_index().sort_values('Data')
            ax.bar(ld['Data'], norm_serie(ld['load']), color=CORES['cinza'], alpha=0.3, label='Load (norm)', width=0.8)
        if 'hrv' in dw.columns and len(dw) > 0:
            dw2 = dw.dropna(subset=['hrv']).copy(); dw2['Data'] = pd.to_datetime(dw2['Data'])
            ax.plot(dw2['Data'], norm_serie(dw2['hrv']), color=CORES['verde'], linewidth=2, linestyle='--', label='HRV (norm)')
        ax.set_title('Performance Overview', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9); ax.tick_params(axis='x', rotation=45); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col_r:
        st.subheader("🎯 Distribuição")
        df_d = filtrar_principais(da).copy()
        if len(df_d) > 0:
            cnt = df_d['type'].apply(norm_tipo).value_counts()
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.pie(cnt.values, labels=cnt.index, autopct='%1.0f%%',
                    colors=[get_cor(t) for t in cnt.index], startangle=90,
                    pctdistance=0.75, wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
            ax2.text(0, 0, f'{cnt.sum()}', fontsize=28, fontweight='bold', ha='center', va='center')
            plt.tight_layout(); st.pyplot(fig2); plt.close()
    st.markdown("---")
    st.subheader("📋 Atividades Recentes")
    df_tab = filtrar_principais(da).sort_values('Data', ascending=False).head(10)
    if len(df_tab) > 0:
        cs = [c for c in ['Data', 'type', 'name', 'moving_time', 'rpe', 'power_avg', 'icu_eftp'] if c in df_tab.columns]
        ds = df_tab[cs].copy()
        if 'moving_time' in ds.columns:
            ds['moving_time'] = ds['moving_time'].apply(lambda x: f"{int(x/3600)}h{int((x%3600)/60):02d}m" if pd.notna(x) else '—')
        ds.columns = [c.replace('_', ' ').title() for c in ds.columns]
        st.dataframe(ds, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("📋 Resumo Semanal")
    col1, col2, col3 = st.columns(3)
    if len(da) > 0:
        dw7 = da[pd.to_datetime(da['Data']).dt.date >= (datetime.now().date() - timedelta(days=7))]
        col1.metric("Sessões (7d)", len(dw7))
        if 'moving_time' in dw7.columns: col2.metric("Horas (7d)", f"{dw7['moving_time'].sum()/3600:.1f}h")
        if 'rpe' in dw7.columns and dw7['rpe'].notna().any(): col3.metric("RPE médio (7d)", f"{dw7['rpe'].mean():.1f}")
    df_rank = filtrar_principais(da).copy()
    if 'power_avg' in df_rank.columns and df_rank['power_avg'].notna().any():
        st.subheader("🏆 Top 10 por Potência")
        top = df_rank.nlargest(10, 'power_avg')[['Data', 'type', 'name', 'power_avg', 'rpe']].copy()
        top['Data'] = pd.to_datetime(top['Data']).dt.strftime('%Y-%m-%d')
        top.columns = ['Data', 'Tipo', 'Nome', 'Power (W)', 'RPE']
        st.dataframe(top, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PMC + FTLM
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_pmc.py
# ════════════════════════════════════════════════════════════════════════════

def tab_pmc(da):
    """
    PMC usa SEMPRE o histórico completo (histórico carregado = 730d) para calcular
    CTL/ATL/FTLM correctamente — o filtro de período apenas controla o que é exibido.
    """
    st.header("📈 PMC — Performance Management Chart")
    if len(da) == 0: st.warning("Sem dados de atividades."); return

    # Usa todos os dados disponíveis (não só o período filtrado)
    # da já vem do carregar_atividades(days_back=730) via st.session_state
    da_full = st.session_state.get('da_full', da)
    # Filtrar por modalidades seleccionadas (se disponível no session_state)
    _mods = st.session_state.get('mods_sel', None)
    if _mods and 'type' in da_full.columns:
        da_full = da_full[da_full['type'].isin(_mods + ['WeightTraining'])]
    df = filtrar_principais(da_full).copy(); df['Data'] = pd.to_datetime(df['Data'])

    # MÉTRICA: session_rpe = (moving_time_min) × RPE — igual ao código original Python/SQLite
    # CTL=~350-400 com esta escala (60min×RPE6=360). icu_training_load daria CTL~40-80 (escala errada).
    if 'moving_time' in df.columns and 'rpe' in df.columns and df['rpe'].notna().sum() > 10:
        df['rpe_fill'] = df['rpe'].fillna(df['rpe'].median())
        df['load_val'] = (df['moving_time'] / 60) * df['rpe_fill']
        _load_metrica = "session_rpe = (moving_time_min × RPE) — igual ao Python/SQLite original"
    elif 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        df['load_val'] = pd.to_numeric(df['icu_training_load'], errors='coerce').fillna(0)
        _load_metrica = "icu_training_load — escala diferente (CTL ~40-80 em vez de ~350-400)"
    else:
        st.warning("Sem dados de load (rpe ou icu_training_load necessários)."); return

    ld = df.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
    idx_full = pd.date_range(ld['Data'].min(), datetime.now().date())
    ld = ld.set_index('Data').reindex(idx_full, fill_value=0).reset_index(); ld.columns = ['Data', 'load_val']

    # CTL/ATL sobre TODO o histórico (para que os valores actuais sejam correctos)
    ld['CTL'] = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL'] = ld['load_val'].ewm(span=7,  adjust=False).mean()
    ld['TSB'] = ld['CTL'] - ld['ATL']

    # FTLM — gamma otimizado sobre todo o histórico
    best_g, best_r = 0.30, -1
    for g in np.arange(0.25, 0.36, 0.01):
        ema = ld['load_val'].ewm(alpha=g, adjust=False).mean()
        if ema.std() > 0:
            r = abs(np.corrcoef(ld['load_val'].values, ema.values)[0, 1])
            if r > best_r: best_r, best_g = r, g
    ld['FTLM'] = ld['load_val'].ewm(alpha=best_g, adjust=False).mean()

    # Filtro de exibição — controla o período mostrado no gráfico
    st.info(f"📊 Métrica de load: **{_load_metrica}** | Histórico: {len(ld)} dias")
    if "session_rpe" in _load_metrica:
        st.warning("⚠️ Para resultados equivalentes ao Python/SQLite: usa icu_training_load. Exporta o histórico completo do Intervals.icu para a Google Sheet.")

    col1, col2, col3 = st.columns(3)
    dias_exib_opts = {"30 dias": 30, "60 dias": 60, "90 dias": 90, "180 dias": 180, "1 ano": 365, "Todo histórico": len(ld)}
    dias_exib_lbl = col1.selectbox("Período exibido", list(dias_exib_opts.keys()), index=2)
    dias_exib = dias_exib_opts[dias_exib_lbl]
    ld_plot = ld.tail(dias_exib).copy()

    smooth = col2.checkbox("Suavizar CTL/ATL (3d)", value=False)
    show_ftlm = col3.checkbox("Mostrar FTLM", value=True)
    if smooth:
        ld_plot = ld_plot.copy()
        ld_plot['CTL'] = ld_plot['CTL'].rolling(3, min_periods=1).mean()
        ld_plot['ATL'] = ld_plot['ATL'].rolling(3, min_periods=1).mean()

    idx = ld_plot['Data']  # para as barras de load por tipo

    fig, (ax_pmc, ax_load) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ax_pmc.plot(ld_plot['Data'], ld_plot['CTL'], label='CTL', color=CORES['azul'], linewidth=2.5)
    ax_pmc.plot(ld_plot['Data'], ld_plot['ATL'], label='ATL', color=CORES['vermelho'], linewidth=2.5)
    ax_pmc.fill_between(ld_plot['Data'], 0, ld_plot['TSB'], where=(ld_plot['TSB'] >= 0), color=CORES['verde'], alpha=0.25, label='TSB+')
    ax_pmc.fill_between(ld_plot['Data'], 0, ld_plot['TSB'], where=(ld_plot['TSB'] < 0), color=CORES['vermelho'], alpha=0.20, label='TSB-')
    ax_pmc.axhline(0, color=CORES['cinza'], linestyle='--', linewidth=0.8)
    ax_pmc.set_ylabel('CTL / ATL / TSB', fontweight='bold'); ax_pmc.grid(True, alpha=0.3)
    if show_ftlm:
        ax2 = ax_pmc.twinx()
        ax2.plot(ld_plot['Data'], ld_plot['FTLM'], label=f'FTLM (gamma={best_g:.2f})', color=CORES['laranja'], linewidth=2, linestyle='--', alpha=0.85)
        ax2.set_ylabel('FTLM', color=CORES['laranja'], fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=CORES['laranja'])
        l1, lb1 = ax_pmc.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
        ax_pmc.legend(l1 + l2, lb1 + lb2, loc='upper left', fontsize=9)
    else:
        ax_pmc.legend(loc='upper left', fontsize=9)
    # Valores actuais = sempre ultimo dia de todo o historico
    u = ld.iloc[-1]
    ax_pmc.text(0.99, 0.97, f"CTL: {u['CTL']:.0f}  |  ATL: {u['ATL']:.0f}  |  TSB: {u['TSB']:+.0f}",
                transform=ax_pmc.transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax_pmc.set_title('PMC — CTL / ATL / TSB / FTLM / Training Load', fontsize=14, fontweight='bold')
    # Stacked bars por modalidade — igual ao original plot_training_load
    if 'type' in df.columns:
        tipos_ord = [t for t in ['Bike', 'Row', 'Ski', 'Run', 'WeightTraining'] if t in df['type'].unique()]
        tipos_ord += [t for t in df['type'].unique() if t not in tipos_ord]
        bottom_vals = np.zeros(len(ld_plot))
        for tipo in tipos_ord:
            dt = df[df['type'] == tipo].groupby('Data')['load_val'].sum().reset_index()
            dt.columns = ['Data', 'lv']
            dt['Data'] = pd.to_datetime(dt['Data'])
            # Alinhar com ld_plot usando merge
            merged_bar = ld_plot[['Data']].merge(dt, on='Data', how='left').fillna(0)
            vals = merged_bar['lv'].values
            ax_load.bar(ld_plot['Data'], vals, bottom=bottom_vals,
                        color=get_cor(tipo), alpha=0.85, width=0.8, label=tipo, edgecolor='white', linewidth=0.3)
            bottom_vals += vals
        ax_load.legend(loc='upper left', fontsize=8, ncol=min(5, len(tipos_ord)))
    else:
        ax_load.bar(ld_plot['Data'], ld_plot['load_val'], color=CORES['roxo'], alpha=0.65, width=0.8, label='Training Load')
        ax_load.legend(loc='upper left', fontsize=8)
    ax_load.set_ylabel('Load\n(TRIMP)', fontweight='bold', fontsize=9); ax_load.grid(True, alpha=0.2, axis='y')
    ax_load.tick_params(axis='x', rotation=45)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.subheader("Resumo PMC")
    res = pd.DataFrame({'Metrica': ['CTL (atual)', 'ATL (atual)', 'TSB (atual)', 'CTL (max hist)', 'ATL (max hist)', 'FTLM (atual)', 'FTLM gamma'],
                        'Valor': [f"{u['CTL']:.1f}", f"{u['ATL']:.1f}", f"{u['TSB']:+.1f}",
                                  f"{ld['CTL'].max():.1f}", f"{ld['ATL'].max():.1f}", f"{u['FTLM']:.1f}", f"{best_g:.3f}"]})
    st.dataframe(res, use_container_width=True, hide_index=True)

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

def tab_eftp(da, mods_sel):
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
    df = filtrar_principais(da).copy(); df['Data'] = pd.to_datetime(df['Data']); df['ano'] = df['Data'].dt.year
    df = df[df['type'].isin(mods_sel)]
    if len(df) == 0: st.warning("Sem dados."); return
    anos = sorted(df['ano'].unique()); ano_sel = st.selectbox("Ano", anos, index=len(anos) - 1)
    df_ano = df[df['ano'] == ano_sel].copy()
    zonas_hr = [c for c in df.columns if c.lower().startswith('hr_z') and c.lower().endswith('_secs')]
    rpe_col = next((c for c in ['rpe', 'RPE', 'icu_rpe'] if c in df.columns), None)
    def gzn(col): m = re.search(r'hr_z(\d+)_secs', col.lower()); return int(m.group(1)) if m else 0
    CORES_HR = {'Baixa (Z1+Z2)': '#2ECC71', 'Moderada (Z3+Z4)': '#F39C12', 'Alta (Z5+Z6+Z7)': '#E74C3C'}
    CORES_RPE = {'Leve (1–4)': '#3498DB', 'Moderado (5–6)': '#F39C12', 'Forte (7–10)': '#E74C3C'}
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    if zonas_hr:
        dh = df_ano.copy()
        bc = [c for c in zonas_hr if gzn(c) in (1, 2)]; mc = [c for c in zonas_hr if gzn(c) in (3, 4)]; ac = [c for c in zonas_hr if gzn(c) in (5, 6, 7)]
        for cols in [bc, mc, ac]:
            for c in cols: dh[c] = pd.to_numeric(dh[c], errors='coerce').fillna(0)
        dh['zb'] = dh[bc].sum(axis=1) if bc else 0; dh['zm'] = dh[mc].sum(axis=1) if mc else 0; dh['za'] = dh[ac].sum(axis=1) if ac else 0
        dh['zt'] = dh['zb'] + dh['zm'] + dh['za']; dh = dh[dh['zt'] > 0]
        if len(dh) > 0:
            for z, p in [('zb', 'pb'), ('zm', 'pm'), ('za', 'pa')]: dh[p] = dh[z] / dh['zt'] * 100
            pt = dh.groupby('type')[['pb', 'pm', 'pa']].mean(); pt.columns = list(CORES_HR.keys())
            ts = {'Baixa (Z1+Z2)': dh['zb'].sum(), 'Moderada (Z3+Z4)': dh['zm'].sum(), 'Alta (Z5+Z6+Z7)': dh['za'].sum()}
            _zonas_bp(axes[0], axes[1], pt, ts, CORES_HR, f'❤️ HR Zones — {ano_sel}', f'❤️ HR Geral — {ano_sel}')
    if rpe_col:
        dr = df_ano.dropna(subset=[rpe_col]).copy(); dr[rpe_col] = pd.to_numeric(dr[rpe_col], errors='coerce')
        dr['rz'] = pd.cut(dr[rpe_col], bins=[0, 4.9, 6.9, 10], labels=list(CORES_RPE.keys()), right=True); dr = dr.dropna(subset=['rz'])
        if len(dr) > 0:
            piv = dr.groupby(['type', 'rz'], observed=True).size().unstack(fill_value=0)
            for z in CORES_RPE.keys():
                if z not in piv.columns: piv[z] = 0
            piv = piv[list(CORES_RPE.keys())]; pct = piv.div(piv.sum(axis=1), axis=0) * 100
            tr = {z: piv[z].sum() for z in CORES_RPE.keys()}
            _zonas_bp(axes[2], axes[3], pct, tr, CORES_RPE, f'🎯 RPE Zones — {ano_sel}', f'🎯 RPE Geral — {ano_sel}')
    plt.suptitle(f'HR Zones · RPE Zones — {ano_sel}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("🔗 Correlação HR Zones × RPE")
    if zonas_hr and rpe_col:
        df_ok = df.copy()
        for c in [c for c in zonas_hr]: df_ok[c] = pd.to_numeric(df_ok[c], errors='coerce').fillna(0)
        df_ok['zb'] = df_ok[[c for c in zonas_hr if gzn(c) in (1, 2)]].sum(axis=1)
        df_ok['zm'] = df_ok[[c for c in zonas_hr if gzn(c) in (3, 4)]].sum(axis=1)
        df_ok['za'] = df_ok[[c for c in zonas_hr if gzn(c) in (5, 6, 7)]].sum(axis=1)
        df_ok['zt'] = df_ok['zb'] + df_ok['zm'] + df_ok['za']; mhr = df_ok['zt'] > 0
        df_ok.loc[mhr, 'pb'] = df_ok.loc[mhr, 'zb'] / df_ok.loc[mhr, 'zt'] * 100
        df_ok.loc[mhr, 'pm'] = df_ok.loc[mhr, 'zm'] / df_ok.loc[mhr, 'zt'] * 100
        df_ok.loc[mhr, 'pa'] = df_ok.loc[mhr, 'za'] / df_ok.loc[mhr, 'zt'] * 100
        df_ok[rpe_col] = pd.to_numeric(df_ok[rpe_col], errors='coerce')
        df_ok['rpe_cat'] = pd.cut(df_ok[rpe_col], bins=[0, 4.9, 6.9, 10], labels=['Leve (1–4)', 'Moderado (5–6)', 'Forte (7–10)'], right=True)
        df_ok = df_ok.dropna(subset=[rpe_col, 'pb', 'pm', 'pa', 'rpe_cat']); df_ok = df_ok[df_ok['zt'] > 0]
        mods_corr = [m for m in mods_sel if m in df_ok['type'].values]
        if mods_corr and len(df_ok) >= 5:
            HR_VARS = ['pb', 'pm', 'pa']; HR_LABELS = ['HR Baixa\n(Z1+Z2)', 'HR Moderada\n(Z3+Z4)', 'HR Alta\n(Z5+Z6+Z7)']
            CORES_RPE_CAT = {'Leve (1–4)': CORES['azul'], 'Moderado (5–6)': CORES['laranja'], 'Forte (7–10)': CORES['vermelho']}
            fig, axes2 = plt.subplots(1, len(mods_corr), figsize=(5 * len(mods_corr), 5))
            if len(mods_corr) == 1: axes2 = [axes2]
            for ax, mod in zip(axes2, mods_corr):
                dm = df_ok[df_ok['type'] == mod]
                if len(dm) < 5: ax.text(0.5, 0.5, f'{mod}\nn insuficiente', ha='center', va='center', transform=ax.transAxes); continue
                cm_mat = np.zeros((3, 1)); annot = np.empty((3, 1), dtype=object)
                for i, hv in enumerate(HR_VARS):
                    x = dm[rpe_col].values; y = dm[hv].values
                    if np.std(y) > 0 and len(x) >= 5:
                        r, p = pearsonr(x, y); cm_mat[i, 0] = r
                        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                        annot[i, 0] = f'r={r:.2f}\n{sig}'
                    else: annot[i, 0] = 'n/a'
                im = ax.imshow(cm_mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
                ax.set_xticks([0]); ax.set_xticklabels(['RPE'], fontsize=10)
                ax.set_yticks(range(3)); ax.set_yticklabels(HR_LABELS, fontsize=9)
                for i in range(3):
                    tc = 'white' if abs(cm_mat[i, 0]) > 0.5 else 'black'
                    ax.text(0, i, annot[i, 0], ha='center', va='center', fontsize=10, fontweight='bold', color=tc)
                ax.set_title(f'{mod} (n={len(dm)})', fontsize=11, fontweight='bold', color=get_cor(mod))
                plt.colorbar(im, ax=ax, shrink=0.7, label='r de Pearson')
            plt.suptitle('Correlação Pearson: RPE × HR Zones', fontsize=13, fontweight='bold', y=1.03)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            rows = []
            forca = lambda r: 'Muito Forte' if abs(r) >= 0.7 else ('Forte' if abs(r) >= 0.5 else ('Moderada' if abs(r) >= 0.3 else 'Fraca'))
            for mod in mods_corr:
                dm = df_ok[df_ok['type'] == mod]
                for hv, hl in zip(HR_VARS, ['HR Baixa', 'HR Moderada', 'HR Alta']):
                    x = dm[rpe_col].values; y = dm[hv].values
                    if np.std(y) > 0 and len(x) >= 5:
                        r, p = pearsonr(x, y)
                        rows.append({'Modalidade': mod, 'HR Zone': hl, 'r': f'{r:+.3f}', 'p-value': f'{p:.4f}', 'n': len(x),
                                     'Sig': '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')), 'Força': forca(r)})
            if rows:
                st.subheader("📋 Tabela de Correlações")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — CORRELAÇÕES & IMPACTO RPE
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_correlacoes.py
# ════════════════════════════════════════════════════════════════════════════

def tab_correlacoes(da, dw):
    st.header("🧠 Correlações & Impacto RPE")
    if len(da) == 0 or len(dw) == 0: st.warning("Sem dados suficientes."); return
    rpe_col = next((c for c in ['rpe', 'RPE', 'icu_rpe'] if c in da.columns), None)

    st.subheader("💚 Impacto do RPE no HRV/RHR (dia seguinte)")
    if rpe_col and 'hrv' in dw.columns:
        da2 = filtrar_principais(da).copy(); da2['Data'] = pd.to_datetime(da2['Data']).dt.normalize()
        dw2 = dw.copy(); dw2['Data'] = pd.to_datetime(dw2['Data']).dt.normalize()
        rpe_daily = da2.groupby('Data')[rpe_col].mean().reset_index(); rpe_daily.columns = ['Data', 'rpe_avg']
        rpe_daily['rpe_cat'] = rpe_daily['rpe_avg'].apply(classificar_rpe)
        dw2_shift = dw2[['Data', 'hrv'] + (['rhr'] if 'rhr' in dw2.columns else [])].copy()
        dw2_shift['Data_prev'] = dw2_shift['Data'] - pd.Timedelta(days=1)
        merged = rpe_daily.merge(dw2_shift.rename(columns={'Data': 'Data_hrv', 'Data_prev': 'Data'}), on='Data', how='inner')
        merged = merged.dropna(subset=['hrv', 'rpe_cat'])
        if len(merged) >= 5:
            cats = ['Leve', 'Moderado', 'Pesado']; baseline_hrv = merged['hrv'].mean()
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(7, 4))
                medias_hrv = {c: merged[merged['rpe_cat'] == c]['hrv'].mean() for c in cats if c in merged['rpe_cat'].values}
                vals = [((medias_hrv.get(c, baseline_hrv) - baseline_hrv) / baseline_hrv * 100) for c in cats]
                ax.bar(cats, vals, color=[CORES['verde'] if v > 0 else CORES['vermelho'] for v in vals], alpha=0.8, edgecolor='white')
                ax.axhline(0, color=CORES['cinza'], linestyle='--'); ax.set_title('Δ HRV% (dia+1) por RPE', fontweight='bold')
                for i, v in enumerate(vals): ax.text(i, v + (1 if v >= 0 else -1.5), f'{v:+.1f}%', ha='center', fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                if 'rhr' in merged.columns:
                    fig, ax = plt.subplots(figsize=(7, 4)); baseline_rhr = merged['rhr'].mean()
                    medias_rhr = {c: merged[merged['rpe_cat'] == c]['rhr'].mean() for c in cats if c in merged['rpe_cat'].values}
                    vals_r = [medias_rhr.get(c, baseline_rhr) - baseline_rhr for c in cats]
                    ax.bar(cats, vals_r, color=[CORES['vermelho'] if v > 0 else CORES['verde'] for v in vals_r], alpha=0.8, edgecolor='white')
                    ax.axhline(0, color=CORES['cinza'], linestyle='--'); ax.set_title('Δ RHR (bpm, dia+1) por RPE', fontweight='bold')
                    for i, v in enumerate(vals_r): ax.text(i, v + (0.3 if v >= 0 else -0.6), f'{v:+.1f}', ha='center', fontsize=9, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout(); st.pyplot(fig); plt.close()
            tabela = []
            for cat in cats:
                sub = merged[merged['rpe_cat'] == cat]
                if len(sub) > 0:
                    dhrv = ((sub['hrv'].mean() - baseline_hrv) / baseline_hrv * 100)
                    drhr = (sub['rhr'].mean() - (merged['rhr'].mean() if 'rhr' in merged.columns else 0)) if 'rhr' in sub.columns else 0
                    tabela.append({'Categoria': cat, 'Δ HRV (%)': f'{dhrv:+.1f}%', 'Δ RHR (bpm)': f'{drhr:+.1f}', 'n': len(sub)})
            if tabela: st.dataframe(pd.DataFrame(tabela), use_container_width=True, hide_index=True)

    st.subheader("🔍 Scatter: RPE → HRV | HRV → RHR")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if rpe_col and 'hrv' in dw.columns and 'merged' in dir() and len(merged) > 5:
        m_clean = merged[['rpe_avg', 'hrv']].dropna()
        if len(m_clean) > 5:
            ax1.scatter(m_clean['rpe_avg'], m_clean['hrv'], c=CORES['azul'], alpha=0.5, s=30)
            z = np.polyfit(m_clean['rpe_avg'], m_clean['hrv'], 1); xl = np.linspace(m_clean['rpe_avg'].min(), m_clean['rpe_avg'].max(), 100)
            ax1.plot(xl, np.poly1d(z)(xl), '--', color=CORES['vermelho'])
            r, _ = pearsonr(m_clean['rpe_avg'], m_clean['hrv'])
            ax1.text(0.05, 0.95, f'r = {r:.3f}', transform=ax1.transAxes, fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.set_xlabel('RPE'); ax1.set_ylabel('HRV (dia+1)'); ax1.set_title('RPE → HRV', fontweight='bold'); ax1.grid(True, alpha=0.3)
    if 'hrv' in dw.columns and 'rhr' in dw.columns:
        dw3 = dw.dropna(subset=['hrv', 'rhr'])
        if len(dw3) > 5:
            ax2.scatter(dw3['hrv'], dw3['rhr'], c=CORES['roxo'], alpha=0.5, s=30)
            z2 = np.polyfit(dw3['hrv'], dw3['rhr'], 1); xl2 = np.linspace(dw3['hrv'].min(), dw3['hrv'].max(), 100)
            ax2.plot(xl2, np.poly1d(z2)(xl2), '--', color=CORES['vermelho'])
            r2 = dw3['hrv'].corr(dw3['rhr'])
            ax2.text(0.05, 0.95, f'r = {r2:.3f}', transform=ax2.transAxes, fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.set_xlabel('HRV (ms)'); ax2.set_ylabel('RHR (bpm)'); ax2.set_title('HRV vs RHR', fontweight='bold'); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📊 Correlações entre Métricas Wellness")
    mets_num = [c for c in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness'] if c in dw.columns and dw[c].notna().any()]
    if len(mets_num) >= 3:
        corr_mat = dw[mets_num].corr(method='pearson')
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
        sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax, square=True, linewidths=0.5, vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlações Wellness (Pearson)', fontsize=13, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("🏃 Impacto por Tipo de Atividade → HRV/RHR (dia+1)")
    if rpe_col and 'type' in da.columns and 'hrv' in dw.columns:
        da3 = filtrar_principais(da).copy(); da3['Data'] = pd.to_datetime(da3['Data']).dt.normalize()
        tipo_daily = da3.groupby('Data')['type'].agg(lambda x: x.mode()[0] if len(x) > 0 else None).reset_index()
        dw2b = dw.copy(); dw2b['Data'] = pd.to_datetime(dw2b['Data']).dt.normalize()
        dw2b_s = dw2b[['Data', 'hrv'] + (['rhr'] if 'rhr' in dw2b.columns else [])].copy()
        dw2b_s['Data_prev'] = dw2b_s['Data'] - pd.Timedelta(days=1)
        merged2 = tipo_daily.merge(dw2b_s.rename(columns={'Data': 'Data_hrv', 'Data_prev': 'Data'}), on='Data', how='inner')
        merged2 = merged2.dropna(subset=['hrv', 'type'])
        if len(merged2) >= 5:
            tipos_disp = [t for t in ['Bike', 'Row', 'Run', 'Ski'] if t in merged2['type'].values]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            baseline_h = merged2['hrv'].mean()
            vals_h = [((merged2[merged2['type'] == t]['hrv'].mean() - baseline_h) / baseline_h * 100) for t in tipos_disp]
            ax1.bar(tipos_disp, vals_h, color=[get_cor(t) for t in tipos_disp], alpha=0.8, edgecolor='white')
            ax1.axhline(0, color=CORES['cinza'], linestyle='--'); ax1.set_title('Atividade → HRV (dia+1)', fontweight='bold')
            ax1.set_ylabel('Δ HRV (%)'); ax1.grid(True, alpha=0.3, axis='y')
            if 'rhr' in merged2.columns:
                baseline_r = merged2['rhr'].mean()
                vals_r = [merged2[merged2['type'] == t]['rhr'].mean() - baseline_r for t in tipos_disp]
                ax2.bar(tipos_disp, vals_r, color=[get_cor(t) for t in tipos_disp], alpha=0.8, edgecolor='white')
                ax2.axhline(0, color=CORES['cinza'], linestyle='--'); ax2.set_title('Atividade → RHR (dia+1)', fontweight='bold')
                ax2.set_ylabel('Δ RHR (bpm)'); ax2.grid(True, alpha=0.3, axis='y')
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — RECOVERY
# ════════════════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: tabs/tab_recovery.py
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — RECOVERY
# ════════════════════════════════════════════════════════════════════════════════
def calcular_recovery(dw):
    if len(dw) == 0: return pd.DataFrame()
    df = dw.copy().sort_values('Data')
    df['hrv_b14'] = df['hrv'].rolling(14, min_periods=7).mean()
    df['rhr_b14'] = df['rhr'].rolling(14, min_periods=7).mean() if 'rhr' in df.columns else np.nan
    df['cv7'] = cvr(df['hrv'], 7); df['cv30'] = cvr(df['hrv'], 30)
    rows = []
    for _, row in df.iterrows():
        hv = row.get('hrv'); hb = row.get('hrv_b14'); cv = row.get('cv7'); hs = 50
        if pd.notna(hv) and pd.notna(hb) and hb > 0 and pd.notna(cv):
            inf, sup = norm_range(hb, cv)
            if inf and sup:
                band = sup - inf
                if hv >= sup: hs = min(100, 75 + (hv - sup) / band * 25 if band > 0 else 75)
                elif hv <= inf: hs = max(0, 40 - (inf - hv) / band * 40 if band > 0 else 40)
                else: hs = 50 + ((hv - inf) / band * 25 if band > 0 else 0)
        rv = row.get('rhr'); rb = row.get('rhr_b14'); rs = 50
        if pd.notna(rv) and pd.notna(rb) and rb > 0:
            pct = (rv - rb) / rb * 100; rs = 90 if pct < -10 else 75 if pct < -5 else 55 if pct < 5 else 35 if pct < 10 else 20
        sl = conv_15(row.get('sleep_quality')); fa = conv_15(row.get('fatiga'))
        st_ = conv_15(row.get('stress')); hu = conv_15(row.get('humor')); so = conv_15(row.get('soreness'))
        score = hs * 0.30 + rs * 0.15 + sl * 0.20 + fa * 0.10 + st_ * 0.10 + hu * 0.05 + so * 0.05 + 50 * 0.05
        inf2, sup2 = norm_range(hb if pd.notna(hb) else 0, cv if pd.notna(cv) else 10)
        rows.append({'Data': row['Data'], 'recovery_score': score, 'hrv': hv, 'hrv_baseline': hb,
                     'hrv_cv7': cv, 'hrv_cv30': row.get('cv30'), 'normal_range_inf': inf2, 'normal_range_sup': sup2,
                     'hrv_comp': hs, 'rhr_comp': rs, 'sleep_comp': sl, 'fatiga_comp': fa, 'stress_comp': st_})
    return pd.DataFrame(rows)

def calcular_bpe(dw, metrica='hrv', baseline_dias=60):
    """
    BPE Z-Score semanal usando metodologia do artigo (igual ao Python original):
    - Baseline = média dos últimos N dias do período TOTAL (fixo, não rolling)
    - CV%      = (STD / Média) × 100 do mesmo período de baseline
    - SWC      = 0.5 × CV% × Baseline / 100
    - Z-Score  = (Média_semanal - Baseline) / SWC
    """
    if metrica not in dw.columns or len(dw) < 14: return pd.DataFrame()
    df = dw.copy().sort_values('Data')
    df['Data'] = pd.to_datetime(df['Data'])
    df[metrica] = pd.to_numeric(df[metrica], errors='coerce')
    df_clean = df.dropna(subset=[metrica])
    if len(df_clean) < 14: return pd.DataFrame()

    # BASELINE FIXO: últimos baseline_dias do período total (igual ao original)
    n_base = min(baseline_dias, len(df_clean))
    baseline_data = df_clean[metrica].tail(n_base)
    base = baseline_data.mean()
    std_base = baseline_data.std()
    if pd.isna(base) or base <= 0 or pd.isna(std_base) or std_base <= 0:
        return pd.DataFrame()
    cv = (std_base / base) * 100
    swc = calcular_swc(base, cv)
    if swc is None or swc <= 0: return pd.DataFrame()

    # Agrupar por semana e calcular Z-Score com baseline fixo
    df_clean['semana'] = df_clean['Data'].dt.to_period('W')
    rows = []
    for sem in sorted(df_clean['semana'].unique()):
        df_sem = df_clean[df_clean['semana'] == sem]
        media_sem = df_sem[metrica].mean()
        if pd.isna(media_sem): continue
        zscore = (media_sem - base) / swc
        rows.append({
            'ano_semana': str(sem),
            'media_semanal': media_sem,
            'baseline': base,
            'swc': swc,
            'cv_percent': cv,
            'zscore': zscore,
            'n_dias': len(df_sem)
        })
    return pd.DataFrame(rows)

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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



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

def analisar_falta_estimulo(df_act_full, janela_dias=14):
    """Análise de falta de estímulo por modalidade — igual ao original."""
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

    data_limite = pd.Timestamp.now() - pd.Timedelta(days=janela_dias)
    carga_rec = ld[ld['Data'] >= data_limite].copy()
    if len(carga_rec) == 0:
        return None

    resultados = {}
    for mod in ['Bike', 'Run', 'Row', 'Ski']:
        df_mod = df[(df['type'] == mod) & (df['Data'] >= data_limite)]
        dias_ativ = df_mod['Data'].nunique()
        freq = dias_ativ / max(janela_dias, 1)

        atl_m = carga_rec['ATL'].mean()
        ctl_m = carga_rec['CTL'].mean()
        gap = ((ctl_m - atl_m) / ctl_m * 100) if ctl_m > 0 else 0
        dias_atl_baixo = (carga_rec['ATL'] < carga_rec['CTL']).sum()

        x_s = np.arange(len(carga_rec))
        slope = np.polyfit(x_s, carga_rec['ATL'].values, 1)[0] if len(carga_rec) > 1 else 0
        slope_norm = max(0, min(1, (slope + 5) / 10))

        need = (
            min(1, max(0, gap / 50)) * 100 * 0.4 +
            min(1, dias_atl_baixo / max(len(carga_rec), 1)) * 100 * 0.3 +
            (1 - slope_norm) * 100 * 0.2 +
            (1 - freq) * 100 * 0.1
        )
        prio = 'ALTA' if need >= 70 else 'MÉDIA' if need >= 40 else 'BAIXA'
        resultados[mod] = {
            'need_score': need, 'prioridade': prio,
            'gap_relativo': gap, 'dias_atl_menor_ctl': int(dias_atl_baixo),
            'dias_com_atividade': dias_ativ, 'atl_medio': atl_m, 'ctl_medio': ctl_m
        }
    return dict(sorted(resultados.items(), key=lambda x: x[1]['need_score'], reverse=True))

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
        st.dataframe(df_res, use_container_width=True, hide_index=True)

    st.subheader("🏆 Top 10 Sessões por Potência Média")
    df_rank = tabela_ranking_power_df(da_full, n=10)
    if len(df_rank) > 0:
        st.dataframe(df_rank, use_container_width=True, hide_index=True)

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
    c1, c2 = st.columns(2)
    for col_w, janela, label in [(c1, 7, "7 dias"), (c2, 14, "14 dias")]:
        res = analisar_falta_estimulo(da_full, janela)
        with col_w:
            st.markdown(f"**📅 Janela {label}**")
            if res:
                rows_fe = []
                for mod, d in res.items():
                    pe = "🔴" if d['prioridade']=='ALTA' else "🟡" if d['prioridade']=='MÉDIA' else "🟢"
                    rows_fe.append({'Modalidade': mod, 'Need Score': f"{d['need_score']:.1f}",
                                    'Prioridade': f"{pe} {d['prioridade']}",
                                    'Gap CTL-ATL': f"{d['gap_relativo']:.1f}%",
                                    'Dias ATL<CTL': d['dias_atl_menor_ctl'],
                                    'Dias c/ Ativ.': d['dias_com_atividade']})
                st.dataframe(pd.DataFrame(rows_fe), use_container_width=True, hide_index=True)
                top = list(res.keys())[0]
                pe = "🔴" if res[top]['prioridade']=='ALTA' else "🟡" if res[top]['prioridade']=='MÉDIA' else "🟢"
                st.info(f"{pe} Foco recomendado: **{top}** (Score: {res[top]['need_score']:.1f})")
            else:
                st.info("Dados insuficientes.")

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
                    st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
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
                    st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

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
                        st.dataframe(pd.DataFrame(trim_rows), use_container_width=True, hide_index=True)

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
                            st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)
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
                        st.dataframe(pd.DataFrame(saz_rows), use_container_width=True, hide_index=True)
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
                    st.dataframe(pd.DataFrame(rpe_rows), use_container_width=True, hide_index=True)

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
                            st.dataframe(pd.DataFrame(trim_rpe), use_container_width=True, hide_index=True)
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
        c5.metric("Horas 7d",       f"{horas7:.1f}h")
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

def _extrair_pot(col):
    """Extrai valor de potência do nome da coluna (ex: HR_140W → 140)."""
    m = re.search(r'(\d+)[_\s]*W', str(col).upper())
    return int(m.group(1)) if m else None

def _detectar_drag_col(df):
    """
    Detecção robusta da coluna Drag Factor.
    Tenta múltiplos nomes possíveis usados na planilha.
    """
    candidatos = [
        'Drag_Factor', 'Drag Factor', 'DragFactor', 'drag_factor',
        'DRAG_FACTOR', 'Drag', 'drag', 'DF', 'df',
    ]
    # Primeiro tenta nomes exactos
    for c in candidatos:
        if c in df.columns:
            return c
    # Depois tenta substring 'DRAG'
    for c in df.columns:
        if 'DRAG' in str(c).upper():
            return c
    return None

def _calcular_sem_mdc(valores):
    """
    SEM e MDC com ICC=0.9.
    SEM = SD × √(1-ICC), MDC₉₅ = SEM × 1.96 × √2
    Igual ao calcular_SEM_MDC do original Python.
    """
    if len(valores) < 2:
        return None
    media = np.mean(valores)
    std   = np.std(valores, ddof=1)
    if media == 0:
        return None
    sem   = std * np.sqrt(1 - 0.9)
    mdc95 = sem * 1.96 * np.sqrt(2)
    mdc90 = sem * 1.645 * np.sqrt(2)
    q1, q3 = np.percentile(valores, 25), np.percentile(valores, 75)
    iqr   = q3 - q1
    return dict(
        mean=media, std=std, SEM=sem, MDC_95=mdc95, MDC_90=mdc90,
        cv=(std / media * 100), n=len(valores),
        z_up=media + 2*std, z_dn=media - 2*std,
        mdc_up=media + mdc95, mdc_dn=media - mdc95,
        iqr_up=q3 + 1.5*iqr, iqr_dn=q1 - 1.5*iqr,
    )

def _analisar_tendencia(df_long):
    """
    4 métodos de análise temporal por potência.
    Igual ao analisar_tendencia_temporal_melhorada() do original Python.
    Retorna dict {pw: resultados}.
    """
    if len(df_long) == 0:
        return {}
    resultados = {}
    for pw in sorted(df_long['Power'].unique()):
        df_pw = df_long[df_long['Power'] == pw].copy().sort_values('Data')
        if len(df_pw) < 3:
            continue
        df_pw['Dias'] = (df_pw['Data'] - df_pw['Data'].min()).dt.days
        X = df_pw['Dias'].values
        y = df_pw['Value'].values

        sl, ic, rv, pv, se = linregress(X, y)
        tau, p_k = spearmanr(X, y)

        bl     = y[:2].mean() if len(y) >= 2 else y[0]
        bl_std = np.std(y[:2], ddof=0) if len(y) >= 2 else 0
        sem_bl = bl_std * np.sqrt(1 - 0.9)
        mdc_bl = sem_bl * 1.96 * np.sqrt(2)
        rec    = y[-2:].mean() if len(y) >= 2 else y[-1]
        mud    = rec - bl
        mud_mdc = mud / mdc_bl if mdc_bl > 0 else 0

        th_sl, th_ic, _, _ = theilslopes(y, X)

        conf = 0
        if pv < 0.05 and abs(sl) > 0:                           conf += 2
        if p_k < 0.05:                                           conf += 2
        if abs(mud_mdc) >= 1:                                    conf += 1
        if (th_sl > 0 and sl > 0) or (th_sl < 0 and sl < 0):   conf += 1

        mudanca_real = conf >= 2
        classif = (("↗ AUMENTANDO" if (sl > 0 or th_sl > 0) else "↘ DIMINUINDO")
                   if mudanca_real else "→ SEM MUDANÇA")

        mid = len(y) // 2
        _, p_t = scipy_stats.ttest_ind(y[:mid], y[mid:]) if mid >= 2 else (0, 1)
        mud_pct = (mud / bl * 100) if bl != 0 else 0

        resultados[pw] = dict(
            n=len(df_pw), dias_total=int(X[-1] - X[0]),
            slope=sl, intercept=ic, r_value=rv, p_value=pv,
            tau_kendall=tau, p_kendall=p_k,
            theil_slope=th_sl, theil_intercept=th_ic,
            baseline=bl, recente=rec,
            mudanca_absoluta=mud, mudanca_percentual=mud_pct,
            mudanca_mdc_multiplos=mud_mdc, mdc_95_baseline=mdc_bl,
            media_inicio=np.mean(y[:mid]), media_fim=np.mean(y[mid:]),
            p_teste_t=p_t,
            classificacao=classif, mudanca_real=mudanca_real,
            confianca_score=conf, confianca_percentual=(conf / 6) * 100,
            valores=y, dias=X,
        )
    return resultados

def _secao_temporal(df_a, cols, tipo_label, unidade, aba, di,
                     cores_pw, slider_key):
    """
    Secção completa de evolução temporal com slider rolling próprio.
    Usada para HR e O2 independentemente — cada uma com o seu slider.

    Parâmetros:
        df_a        : DataFrame completo da aba (histórico total)
        cols        : lista de colunas a plotar (ex: hr_cols ou o2_cols)
        tipo_label  : 'HR' ou 'SmO2'
        unidade     : 'bpm' ou '%'
        aba         : nome da aba (AquecSki, etc.)
        di          : data início do período seleccionado
        cores_pw    : lista de cores por potência
        slider_key  : chave única para o slider Streamlit
    """
    if not cols:
        return
    if 'DATA' not in df_a.columns or not df_a['DATA'].notna().any():
        return
    # Verificar que há dados nas colunas
    has_data = any(df_a[c].notna().any() for c in cols if c in df_a.columns)
    if not has_data:
        st.info(f"Sem dados de {tipo_label} para análise temporal.")
        return

    st.subheader(f"📈 Evolução temporal {tipo_label} — "
                 f"4 métodos (Linear, Mann-Kendall, MDC, Theil-Sen)")

    # ── Slider rolling DEDICADO a esta métrica ───────────────────────────────
    n_max = max(2, len(df_a) - 1)
    roll_w = st.slider(
        f"🔄 Rolling average {tipo_label} (sessões)",
        min_value=1, max_value=min(10, n_max), value=min(3, n_max),
        key=slider_key,
        help=(f"1 = sem suavização | 3 = média de 3 sessões\n"
              f"Afecta apenas os gráficos de {tipo_label}"),
    )

    # Construir DataFrame longo
    rows_t = []
    for col in cols:
        if col not in df_a.columns:
            continue
        p = _extrair_pot(col)
        for _, row in df_a.iterrows():
            if pd.notna(row.get(col)) and pd.notna(row.get('DATA')):
                rows_t.append({'Data': row['DATA'], 'Power': p,
                                'Value': float(row[col])})
    if not rows_t:
        st.info(f"Sem dados de {tipo_label} para análise temporal.")
        return

    df_temporal = pd.DataFrame(rows_t)
    res_tend    = _analisar_tendencia(df_temporal)

    # Gráficos em subplots (um por potência)
    # Todos os W num único gráfico (igual ao comportamento anterior)
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle(
        f'{aba} — Evolução temporal {tipo_label} '
        f'(histórico completo | rolling={roll_w})',
        fontsize=14, fontweight='bold')

    for idx, col in enumerate(cols):
        if col not in df_a.columns:
            continue
        pw  = _extrair_pot(col)
        cor = cores_pw[idx % len(cores_pw)]
        df_t = df_a[['DATA', col]].dropna().sort_values('DATA')
        if len(df_t) < 2:
            continue

        # Scatter + rolling
        ax.scatter(df_t['DATA'], df_t[col],
                   color=cor, alpha=0.35, s=40,
                   edgecolors='white', linewidth=1, zorder=4)
        rolled = df_t[col].rolling(roll_w, min_periods=1).mean()
        ax.plot(df_t['DATA'], rolled, color=cor, linewidth=2.5,
               label=f'{pw}W (rolling {roll_w})', alpha=0.9)

        # Tendência linear
        if pw in res_tend:
            r = res_tend[pw]
            y_lin = r['intercept'] + r['slope'] * r['dias']
            ax.plot(df_t['DATA'], y_lin,
                   color=cor, linewidth=1.5, linestyle='--', alpha=0.6,
                   label=f'{pw}W trend {r["classificacao"]} ' 
                         f'({r["slope"]*30:.2f}/mês p={r["p_value"]:.3f})')

    # Linha vertical: início do período
    ax.axvline(pd.Timestamp(di), color='black', linestyle=':',
              linewidth=2, alpha=0.7, label=f'Início período ({di})')
    ax.set_ylabel(f'{tipo_label} ({unidade})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Data')
    ax.set_title(f'{aba} — {tipo_label} por potência (todos os W juntos)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Saídas escritas detalhadas por potência
    with st.expander(f"📊 Análise detalhada {tipo_label} — "
                     f"saídas escritas completas (4 métodos)"):
        for pw, r in sorted(res_tend.items()):
            st.markdown(f"**⚡ {pw}W** — n={r['n']} medições "
                        f"em {r['dias_total']} dias")
            c1, c2, c3 = st.columns(3)
            c1.metric("Classificação", r['classificacao'])
            c2.metric("Confiança",     f"{r['confianca_percentual']:.0f}%")
            c3.metric("Mudança Real",  "✓ SIM" if r['mudanca_real'] else "✗ NÃO")
            rows_tb = [
                {'Método': 'Regressão Linear',
                 'Slope': (f"{r['slope']:.4f} {unidade}/dia "
                            f"({r['slope']*30:.3f}/mês)"),
                 'R²': f"{r['r_value']**2:.4f}",
                 'p-value': f"{r['p_value']:.4f}",
                 'Sig.': '✓' if r['p_value'] < 0.05 else '✗'},
                {'Método': 'Mann-Kendall (τ)',
                 'Slope': f"τ={r['tau_kendall']:.4f}",
                 'R²': '—', 'p-value': f"{r['p_kendall']:.4f}",
                 'Sig.': '✓' if r['p_kendall'] < 0.05 else '✗'},
                {'Método': 'Theil-Sen (robusto)',
                 'Slope': f"{r['theil_slope']:.4f} {unidade}/dia",
                 'R²': '—', 'p-value': '—', 'Sig.': '—'},
                {'Método': 'Teste T (início vs fim)',
                 'Slope': (f"Δ={r['mudanca_absoluta']:+.1f} {unidade} "
                            f"({r['mudanca_percentual']:+.1f}%)"),
                 'R²': '—', 'p-value': f"{r['p_teste_t']:.4f}",
                 'Sig.': '✓' if r['p_teste_t'] < 0.05 else '✗'},
            ]
            st.dataframe(pd.DataFrame(rows_tb),
                         use_container_width=True, hide_index=True)
            if r['mudanca_real']:
                if '↗' in r['classificacao']:
                    st.warning(f"⚠️ {tipo_label} AUMENTANDO em {pw}W → "
                               "Possível fadiga acumulada ou perda de eficiência")
                else:
                    st.success(f"✓ {tipo_label} DIMINUINDO em {pw}W → "
                               "Melhoria de eficiência / adaptação ao treino")
            else:
                st.info(f"→ {tipo_label} ESTÁVEL em {pw}W — continue a monitorar")
            st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

def tab_aquecimento(dfs_annual, df_annual, di):
    st.header("🌡️ Aquecimento — HR/O2 vs Potência (Annual)")

    if dfs_annual is None or all(v.empty for v in (dfs_annual or {}).values()):
        st.warning(
            "Planilha Annual não carregada. Verifica ANNUAL_SPREADSHEET_ID "
            "e que a planilha está partilhada como 'Qualquer pessoa com o link'.")
        return

    ABAS = [("🎿 Ski", "AquecSki"), ("🚴 Bike", "AquecBike"), ("🚣 Row", "AquecRow")]
    tabs_aq = st.tabs([a[0] for a in ABAS])

    for tab_aq, (label, aba) in zip(tabs_aq, ABAS):
        with tab_aq:
            df_a = (dfs_annual or {}).get(aba, pd.DataFrame())
            if df_a.empty:
                st.info(f"Sem dados para {aba}.")
                continue
            df_a = df_a.copy()

            # ── Filtro de período com dropdown + datas manuais ──────────
            if 'DATA' in df_a.columns and df_a['DATA'].notna().any():
                d_min_hist = df_a['DATA'].min().date()
                d_max_hist = df_a['DATA'].max().date()
                n_total_aba = len(df_a)
                st.caption(
                    f"📅 Histórico: {n_total_aba} registos | "
                    f"{d_min_hist.strftime('%d/%m/%Y')} → "
                    f"{d_max_hist.strftime('%d/%m/%Y')}")
            else:
                d_min_hist = None
                d_max_hist = None

            col_filt1, col_filt2 = st.columns([2, 2])

            with col_filt1:
                opcao_periodo = st.selectbox(
                    "📅 Período para gráficos",
                    options=[
                        "Todo o histórico",
                        "Últimos 60 dias",
                        "Últimos 90 dias",
                        "Últimos 180 dias",
                        "Último ano (365 dias)",
                        "Datas manuais",
                    ],
                    index=0,
                    key=f"periodo_sel_{aba}",
                )

            with col_filt2:
                if opcao_periodo == "Datas manuais" and d_min_hist and d_max_hist:
                    data_ini_man = st.date_input(
                        "Data início",
                        value=d_min_hist,
                        min_value=d_min_hist,
                        max_value=d_max_hist,
                        key=f"data_ini_{aba}",
                    )
                    data_fim_man = st.date_input(
                        "Data fim",
                        value=d_max_hist,
                        min_value=d_min_hist,
                        max_value=d_max_hist,
                        key=f"data_fim_{aba}",
                    )
                else:
                    data_ini_man = None
                    data_fim_man = None
                    # Show info about selected range
                    if opcao_periodo != "Todo o histórico" and d_max_hist:
                        dias_map = {
                            "Últimos 60 dias": 60,
                            "Últimos 90 dias": 90,
                            "Últimos 180 dias": 180,
                            "Último ano (365 dias)": 365,
                        }
                        n_dias = dias_map.get(opcao_periodo, 0)
                        if n_dias:
                            d_calc = d_max_hist - timedelta(days=n_dias)
                            st.caption(f"De {d_calc.strftime('%d/%m/%Y')} "
                                       f"até {d_max_hist.strftime('%d/%m/%Y')}")

            # Aplicar filtro
            if 'DATA' in df_a.columns and df_a['DATA'].notna().any():
                if opcao_periodo == "Todo o histórico":
                    df_plot = df_a.copy()
                elif opcao_periodo == "Datas manuais" and data_ini_man and data_fim_man:
                    df_plot = df_a[
                        (df_a['DATA'].dt.date >= data_ini_man) &
                        (df_a['DATA'].dt.date <= data_fim_man)
                    ].copy()
                else:
                    dias_map = {
                        "Últimos 60 dias": 60,
                        "Últimos 90 dias": 90,
                        "Últimos 180 dias": 180,
                        "Último ano (365 dias)": 365,
                    }
                    n_dias = dias_map.get(opcao_periodo, 0)
                    if n_dias and d_max_hist:
                        d_corte = d_max_hist - timedelta(days=n_dias)
                        df_plot = df_a[df_a['DATA'].dt.date >= d_corte].copy()
                    else:
                        df_plot = df_a.copy()

                if len(df_plot) == 0:
                    st.warning("Sem dados no período seleccionado — "
                               "a usar histórico completo.")
                    df_plot = df_a.copy()
                else:
                    st.caption(f"✅ {len(df_plot)} registos no período seleccionado")
            else:
                df_plot = df_a.copy()

            # ── Diagnóstico de colunas ───────────────────────────────────
            with st.expander("🔍 Diagnóstico de colunas detectadas"):
                st.write(f"**Colunas disponíveis em {aba}:**")
                cols_info = []
                for c in df_a.columns:
                    n_vals = df_a[c].notna().sum()
                    cols_info.append({
                        'Coluna': c,
                        'Tipo': str(df_a[c].dtype),
                        'N não-nulos': n_vals,
                        'Detecção': (
                            'HR' if ('HR' in c.upper() and 'PWR' not in c.upper()
                                     and 'DRAG' not in c.upper()
                                     and _extrair_pot(c))
                            else 'O2' if ('O2' in c.upper() and _extrair_pot(c))
                            else 'HR/Pwr' if ('PWR' in c.upper() and _extrair_pot(c))
                            else 'Drag' if 'DRAG' in c.upper() or 'DRAG' in c.upper()
                            else 'DATA' if c == 'DATA'
                            else '—'
                        )
                    })
                st.dataframe(pd.DataFrame(cols_info),
                             use_container_width=True, hide_index=True)

            # ── Tabela de dados ──────────────────────────────────────────
            with st.expander("📋 Dados (mais recentes primeiro)"):
                cols_ok = [c for c in df_plot.columns
                           if not str(c).startswith('Unnamed')
                           and df_plot[c].notna().any()]
                df_show = (df_plot[cols_ok].sort_values('DATA', ascending=False)
                           if 'DATA' in df_plot.columns
                           else df_plot[cols_ok].iloc[::-1])
                st.dataframe(df_show.head(20), use_container_width=True)

            # Detectar colunas por tipo (detecção robusta)
            hr_cols = sorted(
                [c for c in df_a.columns
                 if 'HR' in c.upper() and 'PWR' not in c.upper()
                 and 'DRAG' not in c.upper() and _extrair_pot(c)],
                key=lambda c: _extrair_pot(c) or 0)

            o2_cols = sorted(
                [c for c in df_a.columns
                 if 'O2' in c.upper() and _extrair_pot(c)],
                key=lambda c: _extrair_pot(c) or 0)

            pwr_cols = sorted(
                [c for c in df_a.columns
                 if 'PWR' in c.upper() and _extrair_pot(c)],
                key=lambda c: _extrair_pot(c) or 0)

            # Detecção robusta Drag Factor
            drag_col = _detectar_drag_col(df_a)

            CORES_HR  = ['#E74C3C', '#F39C12', '#9B59B6', '#2ECC71', '#3498DB']
            CORES_O2  = ['#2471A3', '#1D8348', '#7D3C98', '#117A65', '#C0392B']
            CORES_PW2 = ['#E74C3C', '#F39C12', '#9B59B6', '#2ECC71']

            if not hr_cols and not o2_cols:
                st.warning("Sem colunas HR/O2 detectadas. "
                           "Verifica o diagnóstico de colunas acima.")
                continue

            # ════════════════════════════════════════════════════════════
            # 1. HR e O2 vs Potência — Z-Score, SEM, MDC
            # ════════════════════════════════════════════════════════════
            st.subheader("📊 HR e O2 vs Potência — Z-Score (±2σ), SEM e MDC")

            rows_hr, rows_o2 = [], []
            for col in hr_cols:
                p = _extrair_pot(col)
                for v in df_plot[col].dropna():
                    rows_hr.append({'Power': p, 'Value': float(v)})
            for col in o2_cols:
                p = _extrair_pot(col)
                for v in df_plot[col].dropna():
                    rows_o2.append({'Power': p, 'Value': float(v)})

            df_hr_l = pd.DataFrame(rows_hr)
            df_o2_l = pd.DataFrame(rows_o2)
            res_hr_stat, res_o2_stat = {}, {}

            if len(df_hr_l) > 0 or len(df_o2_l) > 0:
                fig, ax1 = plt.subplots(figsize=(16, 8))
                rng = np.random.default_rng(42)

                # HR — eixo esquerdo
                if len(df_hr_l) > 0:
                    jit = rng.normal(0, 0.3, len(df_hr_l))
                    ax1.scatter(df_hr_l['Power'] + jit, df_hr_l['Value'],
                               alpha=0.25, color='red', s=50,
                               edgecolors='darkred', linewidth=0.5,
                               label='HR pontos', zorder=3)
                    agg = (df_hr_l.groupby('Power')['Value'].mean()
                           .reset_index().sort_values('Power'))
                    ax1.plot(agg['Power'], agg['Value'],
                            color='darkred', linewidth=3, marker='o',
                            markersize=10, zorder=10, label='HR média')
                    first_hr = True
                    for pw in sorted(df_hr_l['Power'].unique()):
                        vals = df_hr_l[df_hr_l['Power'] == pw]['Value'].values
                        if len(vals) < 2:
                            continue
                        s = _calcular_sem_mdc(vals)
                        if not s:
                            continue
                        res_hr_stat[pw] = s
                        ax1.fill_between([pw-2, pw+2],
                                        [s['z_dn']]*2, [s['z_up']]*2,
                                        color='red', alpha=0.08,
                                        label='Z-Score ±2σ' if first_hr else '')
                        ax1.hlines([s['z_up'], s['z_dn']], pw-2, pw+2,
                                  colors='darkred', linestyles='--',
                                  linewidth=1.5, alpha=0.7)
                        ax1.hlines([s['mdc_up'], s['mdc_dn']], pw-1.5, pw+1.5,
                                  colors='red', linestyles=':',
                                  linewidth=1.5, alpha=0.8,
                                  label='MDC-95' if first_hr else '')
                        ax1.text(pw, s['z_up'] + 1.5, f"{s['z_up']:.0f}",
                                fontsize=8, ha='center',
                                color='darkred', fontweight='bold')
                        ax1.text(pw, s['z_dn'] - 2.5, f"{s['z_dn']:.0f}",
                                fontsize=8, ha='center',
                                color='darkred', fontweight='bold')
                        first_hr = False
                    ax1.set_ylabel('HR (bpm)', fontsize=12,
                                  fontweight='bold', color='darkred')
                    ax1.tick_params(axis='y', labelcolor='darkred')

                # O2 — eixo direito
                if len(df_o2_l) > 0:
                    ax2 = ax1.twinx()
                    jit2 = rng.normal(0, 0.3, len(df_o2_l))
                    ax2.scatter(df_o2_l['Power'] + jit2, df_o2_l['Value'],
                               alpha=0.25, color='blue', s=50,
                               edgecolors='darkblue', linewidth=0.5,
                               label='O2 pontos', zorder=3)
                    agg2 = (df_o2_l.groupby('Power')['Value'].mean()
                            .reset_index().sort_values('Power'))
                    ax2.plot(agg2['Power'], agg2['Value'],
                            color='darkblue', linewidth=3, marker='s',
                            markersize=10, zorder=10, label='O2 média')
                    for pw in sorted(df_o2_l['Power'].unique()):
                        vals = df_o2_l[df_o2_l['Power'] == pw]['Value'].values
                        if len(vals) < 2:
                            continue
                        s = _calcular_sem_mdc(vals)
                        if not s:
                            continue
                        res_o2_stat[pw] = s
                        ax2.fill_between([pw-2, pw+2],
                                        [s['z_dn']]*2, [s['z_up']]*2,
                                        color='blue', alpha=0.08)
                        ax2.hlines([s['z_up'], s['z_dn']], pw-2, pw+2,
                                  colors='darkblue', linestyles='--',
                                  linewidth=1.5, alpha=0.7)
                    ax2.set_ylabel('SmO2 (%)', fontsize=12,
                                  fontweight='bold', color='darkblue')
                    ax2.tick_params(axis='y', labelcolor='darkblue')
                    l1, lb1 = ax1.get_legend_handles_labels()
                    l2, lb2 = ax2.get_legend_handles_labels()
                    ax1.legend(l1+l2, lb1+lb2, loc='upper left',
                              fontsize=9, ncol=2)
                else:
                    ax1.legend(loc='upper left', fontsize=9)

                ax1.set_xlabel('Potência (W)', fontsize=12, fontweight='bold')
                ax1.set_title(
                    f'{aba} — HR e O2 vs Potência\nZ-Score (±2σ) e MDC-95',
                    fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.2, linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Tabela de limites estatísticos
                with st.expander("📊 Limites estatísticos por potência "
                                 "(SEM, MDC-90, MDC-95, Z-Score, IQR)"):
                    rows_stat = []
                    for tipo_s, res_d, unid_s in [
                        ('HR', res_hr_stat, 'bpm'),
                        ('SmO2', res_o2_stat, '%'),
                    ]:
                        for pw, s in sorted(res_d.items()):
                            rows_stat.append({
                                'Tipo': tipo_s, 'Potência (W)': pw,
                                'Unidade': unid_s, 'N': s['n'],
                                'Média': f"{s['mean']:.1f}",
                                'STD': f"{s['std']:.1f}",
                                'CV%': f"{s['cv']:.1f}%",
                                'SEM': f"±{s['SEM']:.2f}",
                                'MDC-90': f"±{s['MDC_90']:.2f}",
                                'MDC-95': f"±{s['MDC_95']:.2f}",
                                'Z inf': f"{s['z_dn']:.1f}",
                                'Z sup': f"{s['z_up']:.1f}",
                                'MDC inf': f"{s['mdc_dn']:.1f}",
                                'MDC sup': f"{s['mdc_up']:.1f}",
                                'IQR inf': f"{s['iqr_dn']:.1f}",
                                'IQR sup': f"{s['iqr_up']:.1f}",
                            })
                    if rows_stat:
                        st.dataframe(pd.DataFrame(rows_stat),
                                     use_container_width=True, hide_index=True)

            st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # 2. Evolução temporal HR — slider rolling PRÓPRIO
            # ════════════════════════════════════════════════════════════
            _secao_temporal(
                df_a=df_plot,
                cols=hr_cols,
                tipo_label='HR',
                unidade='bpm',
                aba=aba,
                di=di,
                cores_pw=CORES_HR,
                slider_key=f"roll_hr_{aba}",
            )

            st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # 3. Evolução temporal O2 — slider rolling PRÓPRIO e INDEPENDENTE
            # ════════════════════════════════════════════════════════════
            _secao_temporal(
                df_a=df_plot,
                cols=o2_cols,
                tipo_label='SmO2',
                unidade='%',
                aba=aba,
                di=di,
                cores_pw=CORES_O2,
                slider_key=f"roll_o2_{aba}",
            )

            st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # 4. HR/Pwr ratio — eficiência cardíaca
            # ════════════════════════════════════════════════════════════
            if pwr_cols and 'DATA' in df_a.columns and df_a['DATA'].notna().any():
                st.subheader("⚡ HR/Pwr ratio — eficiência cardíaca (↘ = melhora)")
                roll_pwr = st.slider(
                    "🔄 Rolling average HR/Pwr (sessões)",
                    1, min(10, max(1, len(df_a)-1)), 3,
                    key=f"roll_pwr_{aba}")
                fig3, ax3 = plt.subplots(figsize=(14, 5))
                for i, col in enumerate(pwr_cols[:4]):
                    df_p = df_plot[['DATA', col]].dropna().sort_values('DATA')
                    if len(df_p) < 2:
                        continue
                    pw  = _extrair_pot(col)
                    cor = CORES_PW2[i % len(CORES_PW2)]
                    ax3.scatter(df_p['DATA'], df_p[col],
                               color=cor, alpha=0.35, s=40, zorder=4)
                    ax3.plot(df_p['DATA'],
                            df_p[col].rolling(roll_pwr, min_periods=1).mean(),
                            color=cor, linewidth=2.5, marker='D',
                            markersize=5, alpha=0.85, label=f'{pw}W')
                ax3.set_xlabel('Data')
                ax3.set_ylabel('HR/Pwr (bpm/W)')
                ax3.set_title(f'{aba} — HR/Pwr ratio',
                             fontsize=12, fontweight='bold')
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()
                st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # 5. Drag Factor — evolução temporal
            #    Disponível para AquecRow E AquecSki (e Bike se tiver)
            # ════════════════════════════════════════════════════════════
            if drag_col:
                df_drag = df_plot[['DATA', drag_col]].dropna().sort_values('DATA')
                if len(df_drag) >= 2:
                    st.subheader(f"⚙️ Drag Factor — evolução temporal ({aba})")
                    roll_drag = st.slider(
                        "🔄 Rolling average Drag Factor (sessões)",
                        1, min(10, max(1, len(df_drag)-1)), 3,
                        key=f"roll_drag_{aba}")
                    fig4, ax4 = plt.subplots(figsize=(14, 5))
                    ax4.scatter(df_drag['DATA'], df_drag[drag_col],
                               color=CORES['azul'], alpha=0.45, s=60,
                               edgecolors='white', linewidth=1,
                               zorder=4, label='Medições')
                    ax4.plot(
                        df_drag['DATA'],
                        df_drag[drag_col].rolling(roll_drag, min_periods=1).mean(),
                        color=CORES['azul'], linewidth=2.5,
                        label=f'Rolling {roll_drag}')
                    mu_d = df_drag[drag_col].mean()
                    sd_d = df_drag[drag_col].std()
                    ax4.axhline(mu_d, color='red', linestyle='--',
                               linewidth=1.5, alpha=0.7,
                               label=f"Média: {mu_d:.0f}")
                    ax4.fill_between(df_drag['DATA'],
                                    mu_d - sd_d, mu_d + sd_d,
                                    alpha=0.12, color=CORES['azul'],
                                    label='±1 STD')
                    ax4.set_xlabel('Data')
                    ax4.set_ylabel('Drag Factor')
                    ax4.set_title(f'Drag Factor — {aba}',
                                 fontsize=12, fontweight='bold')
                    ax4.legend(fontsize=9)
                    ax4.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close()

                    c1d, c2d, c3d = st.columns(3)
                    c1d.metric("Drag médio", f"{mu_d:.0f}")
                    c2d.metric("Mínimo", f"{df_drag[drag_col].min():.0f}")
                    c3d.metric("Máximo", f"{df_drag[drag_col].max():.0f}")
                    st.markdown("---")

            elif aba in ('AquecSki', 'AquecRow'):
                # Aviso específico se Ski ou Row não têm Drag Factor
                st.info(
                    f"⚠️ Coluna Drag Factor não detectada em **{aba}**.\n\n"
                    f"Colunas disponíveis: `{', '.join(df_a.columns.tolist())}`\n\n"
                    "O Drag Factor deve ter 'Drag' no nome da coluna na planilha.")

            # ════════════════════════════════════════════════════════════
            # 6. Correlação Drag Factor vs HR e O2 por Potência
            # ════════════════════════════════════════════════════════════
            if drag_col and (hr_cols or o2_cols):
                st.subheader(f"🔗 Correlação Drag Factor vs HR / O2 — {aba}")
                st.caption("Tabela completa + scatter por potência "
                           "(igual ao analisar_com_sem_mdc() do original Python)")

                corr_rows = []
                for col in hr_cols + o2_cols:
                    tipo_c = 'HR' if col in hr_cols else 'SmO2'
                    df_c   = df_plot[[drag_col, col]].dropna()
                    if len(df_c) < 5:
                        continue
                    sl_c, ic_c, rv_c, pv_c, _ = linregress(
                        df_c[drag_col].values, df_c[col].values)
                    pw = _extrair_pot(col)
                    if   pv_c < 0.05 and sl_c > 0:
                        interp = '↗ Drag↑ → valor↑ (pior eficiência)'
                    elif pv_c < 0.05 and sl_c < 0:
                        interp = '↘ Drag↑ → valor↓ (melhor eficiência)'
                    else:
                        interp = '→ sem relação significativa'
                    corr_rows.append({
                        'Tipo': tipo_c, 'Potência (W)': pw, 'N': len(df_c),
                        'r': f"{rv_c:.3f}", 'R²': f"{rv_c**2:.3f}",
                        'Slope': f"{sl_c:.4f}",
                        'p-value': f"{pv_c:.4f}",
                        'Sig.': '✓ SIG' if pv_c < 0.05 else '✗',
                        'Interpretação': interp,
                    })

                if corr_rows:
                    st.dataframe(pd.DataFrame(corr_rows),
                                 use_container_width=True, hide_index=True)

                    n_sc = len(corr_rows)
                    if n_sc > 0:
                        nc = min(2, n_sc)
                        nr = (n_sc + 1) // 2
                        fig5, axes5 = plt.subplots(nr, nc,
                                                    figsize=(14, 5 * nr))
                        axes5_flat = (axes5.flatten()
                                      if hasattr(axes5, 'flatten') else [axes5])
                        CORES_SC = ['#E74C3C', '#F39C12', '#9B59B6',
                                    '#2ECC71', '#3498DB', '#1ABC9C']
                        for idx_s, crow in enumerate(corr_rows):
                            if idx_s >= len(axes5_flat):
                                break
                            axs   = axes5_flat[idx_s]
                            pw_s  = crow['Potência (W)']
                            t_s   = crow['Tipo']
                            col_s = next(
                                (c for c in (hr_cols if t_s == 'HR'
                                             else o2_cols)
                                 if _extrair_pot(c) == pw_s), None)
                            if col_s is None:
                                continue
                            df_sc = df_plot[[drag_col, col_s]].dropna()
                            if len(df_sc) < 3:
                                continue
                            cor_s = CORES_SC[idx_s % len(CORES_SC)]
                            axs.scatter(df_sc[drag_col], df_sc[col_s],
                                       color=cor_s, alpha=0.6, s=70,
                                       edgecolors='white', linewidth=1)
                            x_sc = df_sc[drag_col].values
                            sl_sc, ic_sc, rv_sc, pv_sc, _ = linregress(
                                x_sc, df_sc[col_s].values)
                            x_ln = np.linspace(x_sc.min(), x_sc.max(), 100)
                            axs.plot(x_ln, sl_sc * x_ln + ic_sc,
                                    'r--', linewidth=2, alpha=0.8)
                            sig   = '✓' if pv_sc < 0.05 else '✗'
                            unid_s = 'bpm' if t_s == 'HR' else '%'
                            axs.set_xlabel('Drag Factor')
                            axs.set_ylabel(f'{t_s} ({unid_s})')
                            axs.set_title(
                                f'Drag Factor vs {t_s} {pw_s}W | '
                                f'r={rv_sc:.3f} {sig}',
                                fontsize=10, fontweight='bold')
                            axs.grid(True, alpha=0.3)
                        for idx_s in range(len(corr_rows), len(axes5_flat)):
                            axes5_flat[idx_s].set_visible(False)
                        plt.suptitle(
                            f'{aba} — Drag Factor vs HR e O2 por potência',
                            fontsize=13, fontweight='bold', y=1.01)
                        plt.tight_layout()
                        st.pyplot(fig5)
                        plt.close()
                st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # 7. SEM/MDC por grupo de Drag Factor (3 partes)
            # ════════════════════════════════════════════════════════════
            if drag_col and (hr_cols or o2_cols):
                st.subheader(f"🔬 SEM/MDC por grupo de Drag Factor — {aba}")
                st.caption("Partes 1, 2 e 3 do analisar_com_sem_mdc() "
                           "do código original Python.")

                df_db = df_plot.dropna(subset=[drag_col, 'DATA']).copy()
                if len(df_db) >= 6:
                    try:
                        df_db['Drag_Quartil'] = pd.qcut(
                            df_db[drag_col], q=3,
                            labels=['Baixo DF', 'Médio DF', 'Alto DF'],
                            duplicates='drop')
                    except Exception:
                        df_db['Drag_Quartil'] = 'Único'

                    for tipo_d, cols_d, unid_d in [
                        ('HR', hr_cols, 'bpm'),
                        ('SmO2', o2_cols, '%'),
                    ]:
                        if not cols_d:
                            continue

                        # PARTE 1: SEM/MDC por grupo
                        st.markdown(
                            f"**PARTE 1 — {tipo_d} ({unid_d}): "
                            f"SEM/MDC por grupo de Drag Factor**")
                        sem_rows = []
                        for col in cols_d:
                            pw_d   = _extrair_pot(col)
                            df_col = df_db[[col, 'Drag_Quartil']].dropna()
                            for grupo in sorted(df_col['Drag_Quartil'].unique()):
                                vals_g = (df_col[df_col['Drag_Quartil'] == grupo]
                                          [col].values)
                                if len(vals_g) < 2:
                                    continue
                                s_g = _calcular_sem_mdc(vals_g)
                                if not s_g:
                                    continue
                                cv_i = ('✓ Boa'       if s_g['cv'] < 10
                                        else '🟡 Moderada' if s_g['cv'] < 15
                                        else '⚠️ Alta')
                                sem_rows.append({
                                    'Potência (W)': pw_d, 'Grupo Drag': grupo,
                                    'N': s_g['n'],
                                    'Média': f"{s_g['mean']:.1f} {unid_d}",
                                    'STD': f"{s_g['std']:.1f}",
                                    'CV%': f"{s_g['cv']:.1f}%",
                                    'SEM': f"±{s_g['SEM']:.2f}",
                                    'MDC-90': f"±{s_g['MDC_90']:.2f}",
                                    'MDC-95': f"±{s_g['MDC_95']:.2f}",
                                    'Confiabilidade': cv_i,
                                })
                        if sem_rows:
                            st.dataframe(pd.DataFrame(sem_rows),
                                         use_container_width=True,
                                         hide_index=True)

                        # PARTE 2: Mudanças reais vs erro de medição
                        st.markdown(
                            f"**PARTE 2 — {tipo_d}: "
                            f"Mudanças reais vs erro de medição**")
                        comp_rows = []
                        for col in cols_d:
                            pw_d   = _extrair_pot(col)
                            df_col = df_db[[col, 'Drag_Quartil']].dropna()
                            grupos_u = sorted(df_col['Drag_Quartil'].unique())
                            mdc_grp  = {}
                            for g in grupos_u:
                                vals_g = (df_col[df_col['Drag_Quartil'] == g]
                                          [col].values)
                                if len(vals_g) >= 2:
                                    s_g = _calcular_sem_mdc(vals_g)
                                    if s_g:
                                        mdc_grp[g] = s_g
                            for g1, g2 in combinations(grupos_u, 2):
                                if g1 not in mdc_grp or g2 not in mdc_grp:
                                    continue
                                m1, m2  = mdc_grp[g1]['mean'], mdc_grp[g2]['mean']
                                mdc_ref = mdc_grp[g1]['MDC_95']
                                diff    = abs(m2 - m1)
                                mult    = diff / mdc_ref if mdc_ref > 0 else 0
                                verdict = ('✓ DIFERENÇA REAL' if mult >= 1
                                           else '⚠️ Próximo MDC' if mult >= 0.5
                                           else '✗ Dentro do erro')
                                comp_rows.append({
                                    'Potência (W)': pw_d,
                                    'Grupo 1': f"{g1} ({m1:.1f})",
                                    'Grupo 2': f"{g2} ({m2:.1f})",
                                    'Diferença': f"{diff:.1f} {unid_d}",
                                    'MDC-95 ref': f"{mdc_ref:.1f}",
                                    'Múltiplos MDC': f"{mult:.2f}x",
                                    'Veredicto': verdict,
                                })
                        if comp_rows:
                            st.dataframe(pd.DataFrame(comp_rows),
                                         use_container_width=True,
                                         hide_index=True)

                        # PARTE 3: Tendência excede MDC?
                        st.markdown(
                            f"**PARTE 3 — {tipo_d}: "
                            f"Tendência temporal excede MDC?**")
                        tend_rows = []
                        for col in cols_d:
                            pw_d   = _extrair_pot(col)
                            df_col = (df_db[[col, 'Drag_Quartil', 'DATA']]
                                      .dropna())
                            if len(df_col) < 5:
                                continue
                            n_bl  = max(2, int(len(df_col) * 0.3))
                            df_bl = df_col.nsmallest(n_bl, 'DATA')
                            s_bl  = _calcular_sem_mdc(df_bl[col].values)
                            if not s_bl:
                                continue
                            mdc_b = s_bl['MDC_95']
                            for grupo in sorted(df_col['Drag_Quartil'].unique()):
                                df_g = (df_col[df_col['Drag_Quartil'] == grupo]
                                        .sort_values('DATA'))
                                if len(df_g) < 3:
                                    continue
                                v_ini = df_g[col].iloc[0]
                                v_fim = df_g[col].iloc[-1]
                                mud_g = v_fim - v_ini
                                mult  = abs(mud_g) / mdc_b if mdc_b > 0 else 0
                                x_g   = ((df_g['DATA'] - df_g['DATA'].min())
                                         .dt.days.values)
                                y_g   = df_g[col].values
                                if len(np.unique(x_g)) >= 2:
                                    sl_g, _, _, pv_g, _ = linregress(x_g, y_g)
                                else:
                                    sl_g, pv_g = 0, 1
                                verdict = ('✓ MUDANÇA REAL' if mult >= 1
                                           else '⚠️ Possível mudança' if mult >= 0.5
                                           else '✗ Dentro do erro')
                                tend_rows.append({
                                    'Potência (W)': pw_d, 'Grupo': grupo,
                                    'N': len(df_g),
                                    'MDC-95 ref': f"{mdc_b:.1f}",
                                    'Mudança início→fim': f"{mud_g:+.1f} {unid_d}",
                                    'Múltiplos MDC': f"{mult:.2f}x",
                                    'Slope (dia)': f"{sl_g:.4f}",
                                    'p-value': f"{pv_g:.4f}",
                                    'Sig.': '✓' if pv_g < 0.05 else '✗',
                                    'Veredicto': verdict,
                                })
                        if tend_rows:
                            st.dataframe(pd.DataFrame(tend_rows),
                                         use_container_width=True,
                                         hide_index=True)

                    with st.expander("📚 Explicação SEM/MDC"):
                        st.markdown("""
**SEM** = SD × √(1 − ICC), com ICC = 0.9
**MDC₉₅** = SEM × 1.96 × √2 — mudança mínima REAL com 95% confiança
**MDC₉₀** = SEM × 1.645 × √2 — mudança mínima REAL com 90% confiança

| CV% | Confiabilidade |
|---|---|
| < 10% | ✓ Boa |
| 10–15% | 🟡 Moderada |
| > 15% | ⚠️ Alta variabilidade |

**Grupos Drag Factor:** Baixo DF / Médio DF / Alto DF (tercis via `pd.qcut`)
                        """)

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
               "180 dias": 180, "1 ano": 365, "2 anos": 730}
    periodo = st.sidebar.selectbox("📅 Período", list(dias_op.keys()), index=2)
    days_back = dias_op[periodo]

    usar_custom = st.sidebar.checkbox("Datas manuais")
    if usar_custom:
        di = st.sidebar.date_input("Início", datetime.now().date() - timedelta(days=days_back))
        df_ = st.sidebar.date_input("Fim",   datetime.now().date())
    else:
        df_ = datetime.now().date()
        di  = df_ - timedelta(days=days_back)

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
def main():
    days_back, di, df_, mods_sel = render_sidebar()

    st.title("🏃 ATHELTICA Analytics Dashboard")
    st.caption(f"Período: {di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}  |  "
               f"Modalidades: {', '.join(mods_sel)}")

    # ── Carregamento de dados ────────────────────────────────────────────────
    with st.spinner("A carregar dados..."):
        wr           = carregar_wellness(days_back)
        # PMC precisa sempre do histório máximo para CTL/ATL convergirem
        ar_full      = carregar_atividades(730)
        ar           = carregar_atividades(days_back) if days_back < 730 else ar_full
        dfs_annual, df_annual = carregar_annual()   # planilha Annual (AquecSki/Bike/Row)

    if wr.empty and ar_full.empty:
        st.error("Não foi possível carregar dados. Verifica as credenciais e os URLs.")
        st.stop()

    # ── Preprocessing ────────────────────────────────────────────────────────
    wc       = preproc_wellness(wr)
    ac_full  = preproc_ativ(ar_full)   # histórico completo para PMC e Análises
    ac       = preproc_ativ(ar)        # período filtrado

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
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
    ])

    with tab1:  tab_visao_geral(dw, da_filt, di, df_)
    with tab2:  tab_pmc(da_filt)
    with tab3:  tab_volume(da_filt, dw)
    with tab4:  tab_eftp(da_filt, mods_sel)
    with tab5:  tab_zones(da_filt, mods_sel)
    with tab6:  tab_correlacoes(da_filt, dw)
    with tab7:  tab_recovery(dw)
    with tab8:  tab_wellness(dw)
    with tab9:  tab_analises(ac_full, dw, dfs_annual, df_annual)
    with tab10: tab_aquecimento(dfs_annual, df_annual, di)

if __name__ == "__main__":
    main()
