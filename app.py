# ════════════════════════════════════════════════════════════════════════════════
# ATHELTICA DASHBOARD — Streamlit App (Versão Completa)
# Tabs: Visão Geral | PMC | Volume | eFTP | HR+RPE Zones | Correlações | Recovery | Wellness
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
# PÁGINA
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ATHELTICA", page_icon="🏃", layout="wide",
                   initial_sidebar_state="expanded")

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════════════
WELLNESS_URL = "https://docs.google.com/spreadsheets/d/10pefcY6VI4Z45M8Y69D6JxIoqOkjzSlSpV1PMLXoYlI/edit#gid=286320937"
TRAINING_URL = "https://docs.google.com/spreadsheets/d/1RE4SISd53WmAgQo8J-k2SE_OG0w5m4dbgLHvZHPxKvw/edit?usp=sharing"
SCOPES = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

CORES = {
    'verde': '#2ECC71', 'verde_escuro': '#1D8348', 'azul': '#3498DB', 'azul_escuro': '#2471A3',
    'laranja': '#F39C12', 'amarelo': '#F4D03F', 'vermelho': '#E74C3C', 'vermelho_escuro': '#C0392B',
    'roxo': '#9B59B6', 'cinza': '#7F8C8D', 'preto': '#2C3E50', 'branco': '#FFFFFF',
}
CORES_ATIV = {
    'Bike': '#E74C3C', 'Run': '#2ECC71', 'Row': '#3498DB',
    'Ski': '#9B59B6', 'WeightTraining': '#F39C12', 'Other': '#7F8C8D'
}

MAPA_WELLNESS = {
    'hrv': ['HRV', 'hrv', 'Heart Rate Variability'],
    'rhr': ['HRR', 'RHR', 'rhr', 'RestingHR', 'Resting HR'],
    'sleep_hours': ['Horas de Sono', 'Sleep', 'sleep', 'Hours Sleep'],
    'sleep_quality': ['Sono Qualidade', 'Sono_Qualidade', 'Sleep Quality'],
    'stress': ['Stress Do dia', 'Stress', 'stress'],
    'fatiga': ['Cansaço/Vontade de Treinar', 'Fatiga', 'Fadiga'],
    'humor': ['Humor', 'humor', 'Mood'],
    'soreness': ['Cansaço Muscular Geral', 'Muscle Soreness', 'Soreness'],
    'peso': ['Peso', 'Weight'], 'fat': ['FAT', 'Fat', 'Gordura'],
}
MAPA_TRAINING = {
    'date': ['start_date_local', 'date', 'Date', 'data'],
    'start_date_local': ['start_date_local'],
    'moving_time': ['moving_time', 'duration', 'Duration'],
    'distance': ['distance', 'Distance'],
    'power_avg': ['icu_average_watts', 'average_watts', 'AvgPower'],
    'power_max': ['MaxPwr', 'max_power', 'Peak5m'],
    'hr_avg': ['average_heartrate', 'avg_hr'],
    'hr_max': ['max_heartrate', 'max_hr'],
    'rpe': ['icu_rpe', 'RPE', 'rpe'],
    'elevation': ['total_elevation_gain', 'elevation'],
    'type': ['type', 'sport'], 'name': ['name', 'Name'],
    'xss': ['SS', 'XSS', 'xss', 'strain_score'],
    'pmax': ['Pmax', 'pmax', 'p_max_usage'], 'p_max': ['p_max', 'P_max'],
    'icu_pm_p_max': ['icu_pm_p_max'], 'icu_pm_cp': ['icu_pm_cp', 'cp'],
    'icu_training_load': ['icu_training_load'],
    'glycolytic': ['Glycolytic', 'glycolytic', 'glycolytic_usage'],
    'aerobic': ['Aerobic', 'aerobic', 'cp_usage'],
    'cadence_avg': ['average_cadence', 'cadence'],
    'decoupling': ['CardiacDrift', 'decoupling'],
    'icu_ftp': ['icu_ftp', 'FTP', 'ftp'],
    'icu_eftp': ['icu_eftp', 'eFTP', 'estimated_cp', 'est_cp', 'EFTP'],
    'z1_secs': ['z1_secs', 'zone1_seconds'], 'z2_secs': ['z2_secs', 'zone2_seconds'],
    'z3_secs': ['z3_secs', 'zone3_seconds'], 'z4_secs': ['z4_secs', 'zone4_seconds'],
    'z5_secs': ['z5_secs', 'zone5_seconds'], 'z6_secs': ['z6_secs', 'zone6_seconds'],
    'hr_z1_secs': ['hr_z1_secs', 'hr_zone1_seconds'], 'hr_z2_secs': ['hr_z2_secs', 'hr_zone2_seconds'],
    'hr_z3_secs': ['hr_z3_secs', 'hr_zone3_seconds'], 'hr_z4_secs': ['hr_z4_secs', 'hr_zone4_seconds'],
    'hr_z5_secs': ['hr_z5_secs', 'hr_zone5_seconds'], 'hr_z6_secs': ['hr_z6_secs', 'hr_zone6_seconds'],
    'hr_z7_secs': ['hr_z7_secs', 'hr_zone7_seconds'],
    'icu_joules': ['icu_joules'], 'icu_weight': ['icu_weight'],
    'AllWorkFTP': ['AllWorkFTP'], 'WorkHourKgoverCP': ['WorkHourKgoverCP'], 'WorkHour': ['WorkHour'],
}
TYPE_MAP = {
    'VirtualSki': 'Ski', 'AlpineSki': 'Ski', 'Ski': 'Ski', 'NordicSki': 'Ski',
    'VirtualRow': 'Row', 'Rowing': 'Row', 'Row': 'Row',
    'VirtualRide': 'Bike', 'Cycling': 'Bike', 'Ride': 'Bike', 'Bike': 'Bike',
    'MountainBike': 'Bike', 'GravelRide': 'Bike',
    'VirtualRun': 'Run', 'Running': 'Run', 'Run': 'Run', 'TrailRun': 'Run',
    'WeightTraining': 'WeightTraining',
}
VALID_TYPES = ['Bike', 'Row', 'Run', 'Ski', 'WeightTraining']

# ════════════════════════════════════════════════════════════════════════════════
# AUTENTICAÇÃO
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_gc():
    try:
        creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Erro autenticação: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════════
def detectar_col(df, lst):
    for n in lst:
        if n in df.columns: return n
    return None

def br_float(v):
    if pd.isna(v) or v == '': return None
    if isinstance(v, (int, float)): return float(v)
    try: return float(str(v).replace(',', '.').strip())
    except: return None

def parse_date(v):
    if pd.isna(v): return None
    if isinstance(v, (pd.Timestamp, datetime)): return v
    s = str(v).strip()
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try: return datetime.strptime(s, fmt)
        except: pass
    return None

def norm_tipo(t):
    if not isinstance(t, str): return 'Other'
    return TYPE_MAP.get(t.strip(), 'Other')

def get_cor(tipo): return CORES_ATIV.get(tipo, CORES_ATIV['Other'])

def norm_serie(s, lo=0, hi=100):
    mn, mx = s.min(), s.max()
    if mx == mn: return pd.Series([50] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * (hi - lo) + lo

def filtrar_principais(df):
    if len(df) == 0: return df
    df = df.copy()
    df['_t'] = df['type'].apply(norm_tipo)
    df['_d'] = pd.to_datetime(df['Data']).dt.date
    por_dia = df.groupby('_d')['_t'].apply(list).to_dict()
    df['_ok'] = df.apply(lambda r: not (r['_t'] == 'WeightTraining' and len(por_dia.get(r['_d'], [])) > 1), axis=1)
    return df[df['_ok']].drop(columns=['_t', '_d', '_ok'])

def add_tempo(df):
    if len(df) > 0 and 'Data' in df.columns:
        df = df.copy()
        df['mes'] = pd.to_datetime(df['Data']).dt.strftime('%Y-%m')
        df['ano'] = pd.to_datetime(df['Data']).dt.year
        df['trimestre'] = pd.to_datetime(df['Data']).dt.to_period('Q').astype(str)
    return df

def cvr(s, w=7):
    m = s.rolling(w, min_periods=3).mean()
    sd = s.rolling(w, min_periods=3).std()
    return (sd / m) * 100

def conv_15(v):
    if pd.isna(v): return 50
    return max(0, min(100, (float(v) - 1) * 25))

def norm_range(base, cv):
    if pd.isna(base) or pd.isna(cv) or base <= 0: return None, None
    m = 0.5 * (cv / 100) * base
    return base - m, base + m

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

def classificar_rpe(v):
    if pd.isna(v): return None
    v = float(v)
    if 1 <= v <= 4.9: return 'Leve'
    if 5 <= v <= 6.9: return 'Moderado'
    if 7 <= v <= 10: return 'Pesado'
    return None

def calcular_swc(base, cv):
    if pd.isna(base) or pd.isna(cv) or base <= 0 or cv <= 0: return None
    return 0.5 * (cv / 100) * base

# ════════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════════
def preproc_wellness(df):
    if len(df) == 0:
        return df

    df = df.copy().sort_values('Data')

    # duplicatas
    df = df.drop_duplicates(subset=['Data'], keep='first')

    # Z-SCORE
    for c in ['hrv','rhr','sleep_hours','sleep_quality','stress','fatiga','humor','soreness','peso','fat']:
        if c in df.columns:
            df[c] = remove_zscore(df[c], 3.0)

    # zeros inválidos
    df = remove_zeros(df, ['hrv','rhr','sleep_hours'])

    # preenchimento (igual pipeline original)
    for c in ['hrv','rhr','sleep_quality','fatiga','stress','humor','soreness']:
        if c in df.columns:
            df = fill_missing(df, c, 7)

    return df.reset_index(drop=True)

def preproc_ativ(df):
    if len(df) == 0:
        return df

    df = df.copy().sort_values('Data')

    # duplicatas (igual original)
    sub = [c for c in ['Data', 'type', 'moving_time'] if c in df.columns]
    if len(sub) >= 2:
        df = df.drop_duplicates(subset=sub, keep='first')

    # tipos válidos
    if 'type' in df.columns:
        df = df[df['type'].isin(VALID_TYPES)]

    # duração mínima
    if 'moving_time' in df.columns:
        df['moving_time'] = pd.to_numeric(df['moving_time'], errors='coerce')
        df = df[df['moving_time'] > 60]

    # Z-SCORE (igual Untitled13)
    for col, thr in [('icu_eftp', 3.5), ('AllWorkFTP', 3.5)]:
        if col in df.columns:
            df[col] = remove_zscore(df[col], thr)

    # remover zeros inválidos
    df = remove_zeros(df, ['moving_time', 'icu_eftp', 'AllWorkFTP'])

    return df.reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="A carregar wellness...")
def carregar_wellness(days_back):
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        ws = gc.open_by_url(WELLNESS_URL).worksheet("Respostas ao formulário 1")
        df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]
        cd = detectar_col(df, ['Data', 'data', 'Date', 'Carimbo de data/hora'])
        if cd:
            df['Data'] = df[cd].apply(parse_date)
            df = df.dropna(subset=['Data']).sort_values('Data')
        for var, lst in MAPA_WELLNESS.items():
            col = detectar_col(df, lst)
            if col: df[var] = df[col].apply(br_float)
        dm = datetime.now() - timedelta(days=days_back)
        return df[df['Data'] >= dm].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro wellness: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="A carregar atividades...")
def carregar_atividades(days_back):
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        ws = gc.open_by_url(TRAINING_URL).worksheet("intervals.icu_activities-export")
        df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]
        cd = detectar_col(df, ['Date', 'start_date_local', 'date', 'data', 'Data'])
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
        st.error(f"Erro atividades: {e}")
        return pd.DataFrame()

def filtrar_datas(df, di, df_):
    if len(df) == 0: return df
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'])
    return df[(df['Data'].dt.date >= di) & (df['Data'].dt.date <= df_)].reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.title("🏃 ATHELTICA")
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Filtros")
    dias_op = {"30 dias": 30, "60 dias": 60, "90 dias": 90, "180 dias": 180, "1 ano": 365, "2 anos": 730}
    per_lbl = st.sidebar.selectbox("Período", list(dias_op.keys()), index=2)
    days_back = dias_op[per_lbl]
    custom = st.sidebar.checkbox("Datas manuais")
    if custom:
        di = st.sidebar.date_input("Início", value=datetime.now().date() - timedelta(days=days_back))
        df_ = st.sidebar.date_input("Fim", value=datetime.now().date())
    else:
        df_ = datetime.now().date()
        di = df_ - timedelta(days=days_back)
    st.sidebar.markdown("---")
    mods_sel = st.sidebar.multiselect("Modalidades", ['Bike', 'Row', 'Run', 'Ski'], default=['Bike', 'Row', 'Run', 'Ski'])
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Recarregar dados"):
        st.cache_data.clear(); st.rerun()
    st.sidebar.caption(f"{di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}")
    st.sidebar.caption(f"Atualizado: {datetime.now().strftime('%H:%M')}")
    return days_back, di, df_, mods_sel

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — VISÃO GERAL
# ════════════════════════════════════════════════════════════════════════════════
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
def tab_analises(da_full, dw):
    st.header("🔬 Análises Avançadas")
    if len(da_full) == 0:
        st.warning("Sem dados de atividades para análise avançada.")
        return

    # ── Secção 1: Tabelas de Resumo ─────────────────────────────────────────
    st.subheader("📋 Resumo de Atividades por Modalidade")
    df_res = tabela_resumo_por_tipo_df(da_full)
    if len(df_res) > 0:
        st.dataframe(df_res, use_container_width=True, hide_index=True)
    else:
        st.info("Sem dados para tabela de resumo.")

    st.subheader("🏆 Top 10 Sessões por Potência Média")
    df_rank = tabela_ranking_power_df(da_full, n=10)
    if len(df_rank) > 0:
        st.dataframe(df_rank, use_container_width=True, hide_index=True)
    else:
        st.info("Sem dados de potência para ranking.")

    st.markdown("---")

    # ── Secção 2: Training Load Mensal Stacked ──────────────────────────────
    st.subheader("📊 Training Load Mensal por Modalidade (Stacked)")
    df_tl = filtrar_principais(da_full).copy()
    df_tl = add_tempo(df_tl)
    if 'moving_time' in df_tl.columns and 'rpe' in df_tl.columns:
        df_tl['rpe_fill'] = df_tl['rpe'].fillna(df_tl['rpe'].median())
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
                    ax.text(i, b + v / 2, f'{v:.0f}', ha='center', va='center', fontsize=7, fontweight='bold', color='white')
            bottom += vals
        totais = pivot_tl.sum(axis=1).values
        for i, t in enumerate(totais):
            if t > 0:
                ax.text(i, t + 5, f'{t:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(pivot_tl)))
        ax.set_xticklabels(pivot_tl.index, rotation=45, ha='right')
        ax.axhline(totais.mean(), color='black', linestyle='--', alpha=0.5, label=f'Média: {totais.mean():.0f}')
        ax.set_ylabel('Training Load (TRIMP = min × RPE)', fontweight='bold')
        ax.set_title('Training Load Mensal por Modalidade', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Sem dados de moving_time e RPE para Training Load.")

    st.markdown("---")

    # ── Secção 3: Polynomial CTL/ATL ────────────────────────────────────────
    st.subheader("📈 CTL/ATL — Polynomial Fit (Overall e por Modalidade)")
    with st.spinner("Calculando polynomial fits..."):
        poli = calcular_polinomios_carga(da_full)

    if poli is None:
        st.warning("Sem dados suficientes para polynomial analysis.")
    else:
        ld = poli.get('_ld')

        # Overall
        st.markdown("**Overall CTL vs ATL**")
        fig, ax = plt.subplots(figsize=(16, 7))
        CORES_POLI = {'CTL': (CORES['azul'], CORES['azul_escuro']), 'ATL': (CORES['vermelho'], CORES['vermelho_escuro'])}
        for metrica, (cor_s, cor_l) in CORES_POLI.items():
            if metrica not in poli.get('overall', {}): continue
            dados_m = poli['overall'][metrica]
            gk = 'grau3' if 'grau3' in dados_m else 'grau2'
            if gk not in dados_m: continue
            d = dados_m[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
            xs = np.linspace(x.min(), x.max(), 200)
            ax.scatter(x, y, alpha=0.3, s=40, color=cor_s, edgecolors='white', linewidths=1, label=f'{metrica} dados')
            ax.plot(xs, poly(xs), linewidth=3, color=cor_l, linestyle='-' if metrica == 'CTL' else '--',
                    label=f'{metrica} Poly{gk.replace("grau","")} (R²={r2:.3f})')
        ax.set_xlabel('Dias desde início', fontweight='bold'); ax.set_ylabel('Carga (TRIMP)', fontweight='bold')
        ax.set_title('CTL vs ATL Overall — Polynomial Fit', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Por modalidade
        tipos_poli = {k.replace('tipo_', ''): k for k in poli if k.startswith('tipo_')}
        if tipos_poli:
            st.markdown("**Por Modalidade (CTL/ATL separados)**")
            n_t = len(tipos_poli); ncols = 2; nrows = (n_t + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6 * nrows))
            axes_flat = axes.flatten() if nrows > 1 else (axes if ncols == 1 else axes.flatten())
            for idx, (tipo_n, tipo_k) in enumerate(sorted(tipos_poli.items())):
                ax = axes_flat[idx]
                for metrica, cor, sty in [('CTL', CORES['azul'], '-'), ('ATL', CORES['vermelho'], '--')]:
                    if metrica not in poli[tipo_k]: continue
                    dados_m = poli[tipo_k][metrica]
                    gk = 'grau3' if 'grau3' in dados_m else 'grau2'
                    if gk not in dados_m: continue
                    d = dados_m[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
                    xs = np.linspace(x.min(), x.max(), 150)
                    ax.scatter(x, y, alpha=0.35, s=40, color=cor, edgecolors='white', linewidths=1)
                    ax.plot(xs, poly(xs), linewidth=2.5, color=cor, linestyle=sty,
                            label=f'{metrica} Poly{gk.replace("grau","")} R²={r2:.3f}')
                ax.set_title(f'{tipo_n} — CTL/ATL Polynomial', fontsize=11, fontweight='bold')
                ax.set_xlabel('Dias'); ax.set_ylabel('Carga'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            for idx in range(n_t, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            plt.suptitle('CTL/ATL por Modalidade — Polynomial Fit', fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Combinado (todos tipos no mesmo gráfico)
            st.markdown("**CTL e ATL — Todos os Tipos Combinados**")
            fig, axes2 = plt.subplots(1, 2, figsize=(18, 7))
            CORES_TIPO = {'Bike': CORES['vermelho'], 'Run': CORES['verde'], 'Row': CORES['azul'], 'Ski': CORES['roxo']}
            for metrica, ax_m in zip(['CTL', 'ATL'], axes2):
                for tipo_n in sorted(tipos_poli):
                    tipo_k = tipos_poli[tipo_n]
                    if metrica not in poli[tipo_k]: continue
                    dados_m = poli[tipo_k][metrica]
                    gk = 'grau3' if 'grau3' in dados_m else 'grau2'
                    if gk not in dados_m: continue
                    d = dados_m[gk]; x, y, poly, r2 = d['x'], d['y'], d['poly'], d['r2']
                    cor = CORES_TIPO.get(tipo_n, CORES['cinza'])
                    xs = np.linspace(x.min(), x.max(), 150)
                    ax_m.scatter(x, y, alpha=0.25, s=30, color=cor, edgecolors='white', linewidths=0.5)
                    ax_m.plot(xs, poly(xs), linewidth=2.5, color=cor, label=f'{tipo_n} R²={r2:.3f}')
                ax_m.set_title(f'{metrica} — Todos os Tipos', fontsize=12, fontweight='bold')
                ax_m.set_xlabel('Dias'); ax_m.set_ylabel(metrica)
                ax_m.legend(fontsize=10); ax_m.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Secção 4: BPE Heatmap (mapa de estados) ─────────────────────────────
    st.subheader("🗓️ BPE — Mapa de Estados Semanal (Heatmap)")
    if len(dw) < 14:
        st.info("Mínimo 14 dias de wellness necessários para BPE.")
    else:
        mets_bpe = [m for m in ['hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress', 'humor', 'soreness']
                    if m in dw.columns and dw[m].notna().sum() >= 14]
        if not mets_bpe:
            st.info("Sem métricas wellness com dados suficientes para BPE.")
        else:
            n_sem_max = max(4, len(dw) // 7)
            n_sem_bpe = st.slider("Semanas BPE (heatmap)", 4, min(52, n_sem_max), min(16, n_sem_max), key="bpe_heat")
            dados_bpe = {}
            for met in mets_bpe:
                s = calcular_bpe(dw, met, 60)
                if len(s) > 0:
                    dados_bpe[met] = s.tail(n_sem_bpe)

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
                fig, ax = plt.subplots(figsize=(max(14, len(semanas) * 0.9), max(6, nm * 1.3)))
                im = ax.imshow(mat, cmap=cmap_bpe, aspect='auto', vmin=-2, vmax=2)
                ax.set_yticks(range(nm))
                ax.set_yticklabels([nomes_bpe.get(m, m) for m in dados_bpe], fontsize=11)
                sem_labels = [s.split('-W')[1] if '-W' in str(s) else str(s) for s in semanas]
                ax.set_xticks(range(len(semanas)))
                ax.set_xticklabels([f'S{s}' for s in sem_labels], rotation=45, fontsize=10)
                for i in range(nm):
                    for j in range(len(semanas)):
                        v = mat[i, j]
                        cor_txt = 'white' if abs(v) > 1 else 'black'
                        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, fontweight='bold', color=cor_txt)
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Z-Score BPE (múltiplos de SWC)', fontsize=10)
                cbar.set_ticks([-2, -1, 0, 1, 2])
                cbar.set_ticklabels(['🔴 -2', '-1', '0', '+1', '🟢 +2'])
                ax.set_title('BPE — Blocos de Padrão Específico (Z-Score com SWC)', fontsize=13, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

                # BPE Timeline por métrica
                st.markdown("**BPE — Timeline por Métrica**")
                met_sel = st.selectbox("Métrica", list(dados_bpe.keys()), format_func=lambda m: nomes_bpe.get(m, m), key="bpe_timeline_met")
                df_tl_bpe = dados_bpe[met_sel]
                fig2, ax2 = plt.subplots(figsize=(14, 5))
                ax2.axhspan(1, 3, alpha=0.1, color=CORES['verde'], label='Acima (+1 SWC)')
                ax2.axhspan(-1, 1, alpha=0.1, color=CORES['amarelo'], label='Normal (±1 SWC)')
                ax2.axhspan(-3, -1, alpha=0.1, color=CORES['vermelho'], label='Abaixo (-1 SWC)')
                ax2.axhline(0, color='black', linewidth=1, linestyle='--')
                ax2.axhline(1, color=CORES['verde'], linewidth=1, linestyle=':')
                ax2.axhline(-1, color=CORES['vermelho'], linewidth=1, linestyle=':')
                z_vals = (-df_tl_bpe['zscore'].values if met_sel == 'rhr' else df_tl_bpe['zscore'].values)
                cores_z = [CORES['verde'] if v > 1 else CORES['vermelho'] if v < -1 else CORES['amarelo'] for v in z_vals]
                ax2.bar(range(len(df_tl_bpe)), z_vals, color=cores_z, alpha=0.8, edgecolor='white')
                ax2.plot(range(len(df_tl_bpe)), z_vals, 'o-', color='black', linewidth=1.5, markersize=5)
                for j, v in enumerate(z_vals):
                    ax2.text(j, v + (0.1 if v >= 0 else -0.15), f'{v:.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8, fontweight='bold')
                ax2.set_xticks(range(len(df_tl_bpe)))
                ax2.set_xticklabels([f'S{s.split("-W")[1] if "-W" in str(s) else s}' for s in df_tl_bpe['ano_semana']], rotation=45, fontsize=10)
                ax2.set_ylabel('Z-Score BPE (SWC)', fontweight='bold')
                ax2.set_title(f'BPE Timeline — {nomes_bpe.get(met_sel, met_sel)}', fontsize=12, fontweight='bold')
                ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis='y')
                plt.tight_layout(); st.pyplot(fig2); plt.close()

                # Resumo textual BPE
                with st.expander("📖 Interpretação BPE & Metodologia"):
                    st.markdown("""
**Fórmula BPE (Blocos de Padrão Específico):**
- **Baseline** = Média dos últimos 60 dias do período total
- **CV%** = (STD / Média) × 100 do baseline
- **SWC** = 0.5 × CV% × Baseline / 100 *(Smallest Worthwhile Change — Hopkins et al. 2009)*
- **Z-Score** = (Média_semanal − Baseline) / SWC

**Interpretação:**
| Z-Score | Estado |
|---|---|
| > +2 SWC | 🟢🟢 Muito Acima — Excelente recuperação |
| > +1 SWC | 🟢 Acima — Boa recuperação |
| ±1 SWC | 🟡 Normal — variação esperada |
| < -1 SWC | 🟠 Abaixo — atenção |
| < -2 SWC | 🔴 Muito Abaixo — alerta! |

> **Nota:** RHR é invertido (baixo = bom). O Z-Score BPE usa SWC (< STD) → detecta mudanças menores que o Z-Score tradicional.
                    """)

    st.markdown("---")

    # ── Secção 5: Análise de Falta de Estímulo ──────────────────────────────
    st.subheader("🎯 Análise de Falta de Estímulo por Modalidade")
    col_j1, col_j2 = st.columns(2)
    res_7 = analisar_falta_estimulo(da_full, 7)
    res_14 = analisar_falta_estimulo(da_full, 14)

    for janela_label, res in [("7 dias", res_7), ("14 dias", res_14)]:
        with (col_j1 if janela_label == "7 dias" else col_j2):
            st.markdown(f"**📅 Janela {janela_label}**")
            if res:
                rows_fe = []
                for mod, d in res.items():
                    prio_emoji = "🔴" if d['prioridade'] == 'ALTA' else "🟡" if d['prioridade'] == 'MÉDIA' else "🟢"
                    rows_fe.append({
                        'Modalidade': mod,
                        'Need Score': f"{d['need_score']:.1f}",
                        'Prioridade': f"{prio_emoji} {d['prioridade']}",
                        'Gap CTL-ATL': f"{d['gap_relativo']:.1f}%",
                        'Dias ATL<CTL': d['dias_atl_menor_ctl'],
                        'Dias c/ Atividade': d['dias_com_atividade']
                    })
                st.dataframe(pd.DataFrame(rows_fe), use_container_width=True, hide_index=True)
                top = list(res.keys())[0]
                prio_top = res[top]['prioridade']
                prio_e = "🔴" if prio_top == 'ALTA' else "🟡" if prio_top == 'MÉDIA' else "🟢"
                st.info(f"{prio_e} Foco recomendado ({janela_label}): **{top}** (Need Score: {res[top]['need_score']:.1f})")
            else:
                st.info("Dados insuficientes.")

    if res_7 and res_14:
        top7, top14 = list(res_7.keys())[0], list(res_14.keys())[0]
        if top7 == top14:
            st.success(f"✅ Foco consistente em **{top7}** nas duas janelas temporais.")
        else:
            st.warning(f"⚠️ Prioridade divergente: **{top7}** (7d, urgência) vs **{top14}** (14d, tendência)")

    st.markdown("---")

    # ── Secção 6: Saídas Textuais — Resumo Semanal ──────────────────────────
    st.subheader("📋 Resumo Geral (CTL/ATL/TSB + Atividades 7d)")
    if da_full is not None and len(da_full) > 0:
        df_s = filtrar_principais(da_full).copy()
        df_s['Data'] = pd.to_datetime(df_s['Data'])
        if 'moving_time' in df_s.columns and 'rpe' in df_s.columns:
            df_s['rpe_fill'] = df_s['rpe'].fillna(df_s['rpe'].median())
            df_s['load_val'] = (df_s['moving_time'] / 60) * df_s['rpe_fill']
            ld_s = df_s.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
            idx_s = pd.date_range(ld_s['Data'].min(), datetime.now().date())
            ld_s = ld_s.set_index('Data').reindex(idx_s, fill_value=0).reset_index(); ld_s.columns = ['Data', 'load_val']
            ld_s['CTL'] = ld_s['load_val'].ewm(span=42, adjust=False).mean()
            ld_s['ATL'] = ld_s['load_val'].ewm(span=7, adjust=False).mean()
            u_s = ld_s.iloc[-1]
            df_7d = df_s[df_s['Data'] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
            horas_7d = pd.to_numeric(df_7d['moving_time'], errors='coerce').sum() / 3600
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("CTL (Fitness)", f"{u_s['CTL']:.0f}")
            col2.metric("ATL (Fadiga)", f"{u_s['ATL']:.0f}")
            col3.metric("TSB (Forma)", f"{u_s['CTL']-u_s['ATL']:+.0f}")
            col4.metric("Atividades 7d", len(df_7d))
            col5.metric("Horas 7d", f"{horas_7d:.1f}h")
    if dw is not None and len(dw) > 0:
        hrv_7 = pd.to_numeric(dw['hrv'], errors='coerce').dropna().tail(7).mean() if 'hrv' in dw.columns else None
        rhr_u = pd.to_numeric(dw['rhr'], errors='coerce').dropna().iloc[-1] if 'rhr' in dw.columns and len(dw) > 0 else None
        col1w, col2w = st.columns(2)
        if hrv_7: col1w.metric("HRV médio (7d)", f"{hrv_7:.0f} ms")
        if rhr_u: col2w.metric("RHR último", f"{rhr_u:.0f} bpm")

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    days_back, di, df_, mods_sel = render_sidebar()
    st.title("🏃 ATHELTICA Analytics Dashboard")
    st.caption(f"Período: {di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}  |  Modalidades: {', '.join(mods_sel)}")
    with st.spinner("A carregar dados..."):
        wr = carregar_wellness(days_back)
        # PMC precisa sempre de historico maximo para CTL/ATL convergirem correctamente
        ar_pmc = carregar_atividades(730)   # historico completo para PMC
        ar = carregar_atividades(days_back) if days_back < 730 else ar_pmc
    if wr.empty and ar_pmc.empty:
        st.error("Não foi possível carregar dados."); st.stop()
    # Preprocessar tudo
    wc = preproc_wellness(wr)
    ac_full = preproc_ativ(ar_pmc)       # historico completo preprocessado
    ac = preproc_ativ(ar)                # periodo do filtro preprocessado
    # Guardar historico completo no session_state para o PMC
    st.session_state['da_full'] = ac_full
    dw = filtrar_datas(wc, di, df_); da = filtrar_datas(ac, di, df_)
    a_filt = da.copy()

if len(da_filt) > 0 and 'type' in da_filt.columns:
    da_filt['type'] = da_filt['type'].apply(norm_tipo)
    da_filt = da_filt[da_filt['type'].isin(mods_sel + ['WeightTraining'])]
    st.success(f"✅ {len(dw)} registros wellness  |  {len(da_filt)} atividades ({di.strftime('%d/%m/%y')}→{df_.strftime('%d/%m/%y')})  |  Histórico PMC: {len(ac_full)} ativ.")
    with st.expander("🔍 Diagnóstico", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Atividades (RAW)**")
            if len(ar) > 0:
                ar2 = ar.copy(); ar2['Data'] = pd.to_datetime(ar2['Data'])
                st.write(f"Total: {len(ar)} | Data máx: {ar2['Data'].max().date()}")
                cs = [c for c in ['Data', 'type', 'name', 'power_avg', 'rpe', 'icu_eftp'] if c in ar2.columns]
                st.dataframe(ar2[cs].sort_values('Data', ascending=False).head(5), hide_index=True)
        with c2:
            st.markdown("**Wellness (RAW)**")
            if len(wr) > 0:
                wr2 = wr.copy(); wr2['Data'] = pd.to_datetime(wr2['Data'])
                st.write(f"Total: {len(wr)} | Data máx: {wr2['Data'].max().date()}")
                cw = [c for c in ['Data', 'hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress'] if c in wr2.columns]
                st.dataframe(wr2[cw].sort_values('Data', ascending=False).head(5), hide_index=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Visão Geral", "📈 PMC", "📦 Volume", "⚡ eFTP",
        "❤️ HR & RPE", "🧠 Correlações", "🔋 Recovery", "🧘 Wellness", "🔬 Análises"
    ])
    with tab1: tab_visao_geral(dw, da_filt, di, df_)
    with tab2: tab_pmc(ac_full)
    with tab3: tab_volume(da_filt, dw)
    with tab4: tab_eftp(da_filt, mods_sel)
    with tab5: tab_zones(da_filt, mods_sel)
    with tab6: tab_correlacoes(da_filt, dw)
    with tab7: tab_recovery(dw)
    with tab8: tab_wellness(dw)
    with tab9: tab_analises(ac_full, dw)

if __name__ == "__main__":
    main()
