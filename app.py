# ════════════════════════════════════════════════════════════════════════════════
# ATHELTICA DASHBOARD — Streamlit App
# Lê dados do Google Sheets, preprocessa e exibe gráficos interativos.
# ════════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import json
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ATHELTICA Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════════════

WELLNESS_URL  = "https://docs.google.com/spreadsheets/d/10pefcY6VI4Z45M8Y69D6JxIoqOkjzSlSpV1PMLXoYlI/edit#gid=286320937"
TRAINING_URL  = "https://docs.google.com/spreadsheets/d/1RE4SISd53WmAgQo8J-k2SE_OG0w5m4dbgLHvZHPxKvw/edit?usp=sharing"
SCOPES        = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

CORES = {
    'verde': '#2ECC71', 'verde_escuro': '#1D8348',
    'azul': '#3498DB',  'azul_escuro': '#2471A3',
    'laranja': '#F39C12','amarelo': '#F4D03F',
    'vermelho': '#E74C3C','vermelho_escuro': '#C0392B',
    'roxo': '#9B59B6',  'cinza': '#7F8C8D',
    'preto': '#2C3E50', 'branco': '#FFFFFF',
}
CORES_ATIVIDADE = {
    'Bike': '#E74C3C', 'Run': '#2ECC71', 'Row': '#3498DB',
    'Ski': '#9B59B6',  'WeightTraining': '#F39C12', 'Other': '#7F8C8D'
}

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
    'icu_eftp':         ['icu_eftp', 'eFTP', 'estimated_cp', 'est_cp'],
    'z1_secs':          ['z1_secs', 'zone1_seconds'],
    'z2_secs':          ['z2_secs', 'zone2_seconds'],
    'z3_secs':          ['z3_secs', 'zone3_seconds'],
    'z4_secs':          ['z4_secs', 'zone4_seconds'],
    'z5_secs':          ['z5_secs', 'zone5_seconds'],
    'z6_secs':          ['z6_secs', 'zone6_seconds'],
    'hr_z1_secs':       ['hr_z1_secs', 'hr_zone1_seconds'],
    'hr_z2_secs':       ['hr_z2_secs', 'hr_zone2_seconds'],
    'hr_z3_secs':       ['hr_z3_secs', 'hr_zone3_seconds'],
    'hr_z4_secs':       ['hr_z4_secs', 'hr_zone4_seconds'],
    'hr_z5_secs':       ['hr_z5_secs', 'hr_zone5_seconds'],
    'hr_z6_secs':       ['hr_z6_secs', 'hr_zone6_seconds'],
    'hr_z7_secs':       ['hr_z7_secs', 'hr_zone7_seconds'],
    'icu_joules':       ['icu_joules'],
    'icu_weight':       ['icu_weight'],
    'AllWorkFTP':       ['AllWorkFTP'],
    'WorkHourKgoverCP': ['WorkHourKgoverCP'],
    'WorkHour':         ['WorkHour'],
}
TYPE_MAP = {
    'VirtualSki': 'Ski', 'AlpineSki': 'Ski', 'Ski': 'Ski', 'NordicSki': 'Ski',
    'VirtualRow': 'Row', 'Rowing': 'Row', 'Row': 'Row',
    'VirtualRide': 'Bike', 'Cycling': 'Bike', 'Ride': 'Bike',
    'Bike': 'Bike', 'MountainBike': 'Bike', 'GravelRide': 'Bike',
    'VirtualRun': 'Run', 'Running': 'Run', 'Run': 'Run', 'TrailRun': 'Run',
    'WeightTraining': 'WeightTraining',
}

WELLNESS_DESCRICOES = {
    'sleep_quality': {1:'Muito ruim',2:'Ruim',3:'Regular',4:'Bom',5:'Excelente'},
    'fatiga':        {1:'Muito cansado',2:'Cansado',3:'Regular',4:'Bem',5:'Fresco'},
    'stress':        {1:'Muito estressado',2:'Estressado',3:'Normal',4:'Calmo',5:'Relaxado'},
    'humor':         {1:'Muito ruim',2:'Ruim',3:'Normal',4:'Bom',5:'Excelente'},
    'soreness':      {1:'Muita dor',2:'Bastante',3:'Moderada',4:'Leve',5:'Nenhuma'},
}

# ════════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — mesma lógica do DataPreprocessor original
# ════════════════════════════════════════════════════════════════════════════════

from scipy import stats as scipy_stats

VALID_TYPES = ['Bike', 'Row', 'Run', 'Ski', 'WeightTraining']

def remover_picos_zscore(series, threshold=3.0):
    """Remove outliers com Z-score > threshold, substituindo por NaN."""
    valores = pd.to_numeric(series, errors='coerce')
    validos = valores[valores.notna()]
    if len(validos) < 4:
        return valores
    z = np.abs(scipy_stats.zscore(validos))
    picos = validos.index[z > threshold]
    valores.loc[picos] = np.nan
    return valores

def remover_zeros_invalidos(df, colunas):
    """Substitui zeros por NaN em colunas onde zero é inválido."""
    df = df.copy()
    for col in colunas:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan
    return df

def preencher_faltantes(df, coluna, janela=7):
    """Preenche NaN com média rolling, fallback para média global."""
    if coluna not in df.columns:
        return df
    df = df.copy()
    df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
    mask = df[coluna].isna()
    if mask.any():
        roll = df[coluna].rolling(window=janela, min_periods=2, center=True).mean()
        df.loc[mask, coluna] = roll[mask]
        # segundo passo: janela maior
        mask2 = df[coluna].isna()
        if mask2.any():
            roll2 = df[coluna].rolling(window=14, min_periods=2, center=True).mean()
            df.loc[mask2, coluna] = roll2[mask2]
        # fallback: média global
        mask3 = df[coluna].isna()
        if mask3.any():
            df.loc[mask3, coluna] = df[coluna].mean()
    return df

def preprocessar_wellness(df_well):
    """
    Limpa e normaliza o DataFrame de wellness:
    - Remove picos Z-score > 3
    - Remove zeros inválidos (hrv, rhr, sleep_hours)
    - Preenche faltantes com rolling mean
    - Remove duplicatas de data
    """
    if len(df_well) == 0:
        return df_well

    df = df_well.copy().sort_values('Data')

    # Remover duplicatas de data (manter primeiro registo do dia)
    df = df.drop_duplicates(subset=['Data'], keep='first')

    # Colunas numéricas wellness
    cols_num = [c for c in ['hrv','rhr','sleep_hours','sleep_quality',
                             'stress','fatiga','humor','soreness','peso','fat']
                if c in df.columns]

    # Remover picos Z-score
    for col in cols_num:
        df[col] = remover_picos_zscore(df[col], threshold=3.0)

    # Remover zeros inválidos
    df = remover_zeros_invalidos(df, ['hrv','rhr','sleep_hours'])

    # Preencher faltantes
    for col in cols_num:
        df = preencher_faltantes(df, col, janela=7)

    return df.reset_index(drop=True)

def preprocessar_atividades(df_act):
    """
    Limpa e normaliza o DataFrame de atividades:
    - Filtra tipos válidos
    - Remove picos Z-score no eFTP
    - Remove zeros inválidos
    - Remove atividades com moving_time < 60s
    - Remove duplicatas
    """
    if len(df_act) == 0:
        return df_act

    df = df_act.copy().sort_values('Data')

    # Remover duplicatas
    subset_dup = [c for c in ['Data','type','moving_time'] if c in df.columns]
    if len(subset_dup) >= 2:
        df = df.drop_duplicates(subset=subset_dup, keep='first')

    # Filtrar tipos válidos
    if 'type' in df.columns:
        df = df[df['type'].isin(VALID_TYPES)]

    # Remover moving_time < 60s
    if 'moving_time' in df.columns:
        df = df[pd.to_numeric(df['moving_time'], errors='coerce') > 60]

    # Remover picos eFTP
    if 'icu_eftp' in df.columns:
        df['icu_eftp'] = remover_picos_zscore(df['icu_eftp'], threshold=3.5)

    # Remover zeros inválidos
    df = remover_zeros_invalidos(df, ['moving_time','icu_eftp'])

    return df.reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# AUTENTICAÇÃO GOOGLE SHEETS
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_gspread_client():
    """Autentica no Google Sheets usando Service Account em st.secrets."""
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Erro de autenticação Google: {e}")
        st.info("Configura as credenciais em Settings → Secrets do Streamlit Cloud.")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════════

def detectar_coluna(df, possiveis):
    for n in possiveis:
        if n in df.columns: return n
    return None

def br_to_float(val):
    if pd.isna(val) or val == '': return None
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        try: return float(val.replace(',', '.').strip())
        except: return None
    return None

def robust_date_parser(v):
    if pd.isna(v): return None
    if isinstance(v, (pd.Timestamp, datetime)): return v
    s = str(v).strip()
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try: return datetime.strptime(s, fmt)
        except: continue
    return None

def normalizar_tipo(t):
    if not isinstance(t, str): return 'Other'
    return TYPE_MAP.get(t.strip(), 'Other')

def get_cor_atividade(tipo):
    return CORES_ATIVIDADE.get(tipo, CORES_ATIVIDADE['Other'])

def normalizar_serie(serie, min_val=0, max_val=100):
    s_min, s_max = serie.min(), serie.max()
    if s_max == s_min: return pd.Series([50]*len(serie), index=serie.index)
    return ((serie - s_min) / (s_max - s_min)) * (max_val - min_val) + min_val

def filtrar_atividades_principais(df):
    if len(df) == 0: return df
    df = df.copy()
    df['_tipo'] = df['type'].apply(normalizar_tipo)
    df['_dia']  = pd.to_datetime(df['Data']).dt.date
    por_dia = df.groupby('_dia')['_tipo'].apply(list).to_dict()
    df['_manter'] = df.apply(
        lambda r: not (r['_tipo'] == 'WeightTraining' and len(por_dia.get(r['_dia'], [])) > 1), axis=1)
    return df[df['_manter']].drop(columns=['_tipo','_dia','_manter'])

def calcular_cv_rolling(series, window=7):
    m = series.rolling(window=window, min_periods=3).mean()
    s = series.rolling(window=window, min_periods=3).std()
    return (s / m) * 100

def converter_1_5_para_0_100(v):
    if pd.isna(v): return 50
    return max(0, min(100, (float(v) - 1) * 25))

def calcular_normal_range(baseline, cv_pct):
    if pd.isna(baseline) or pd.isna(cv_pct) or baseline <= 0: return None, None
    m = 0.5 * (cv_pct / 100) * baseline
    return baseline - m, baseline + m

# ════════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS (com cache 1h)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="A carregar dados do Google Sheets...")
def carregar_wellness(days_back):
    gc = get_gspread_client()
    if gc is None: return pd.DataFrame()
    try:
        ws = gc.open_by_url(WELLNESS_URL).worksheet("Respostas ao formulário 1")
        df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        col_data = detectar_coluna(df, ['Data','data','Date','Carimbo de data/hora'])
        if col_data:
            df['Data'] = df[col_data].apply(robust_date_parser)
            df = df.dropna(subset=['Data']).sort_values('Data')
        for var, possiveis in MAPA_WELLNESS.items():
            col = detectar_coluna(df, possiveis)
            if col: df[var] = df[col].apply(br_to_float)
        data_min = datetime.now() - timedelta(days=days_back)
        return df[df['Data'] >= data_min].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro ao carregar wellness: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="A carregar atividades do Google Sheets...")
def carregar_atividades(days_back):
    gc = get_gspread_client()
    if gc is None: return pd.DataFrame()
    try:
        ws = gc.open_by_url(TRAINING_URL).worksheet("intervals.icu_activities-export")
        df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        # Detectar coluna de data — testa nomes comuns
        col_data = detectar_coluna(df, ['Date','start_date_local','date','data','Data'])
        if col_data:
            df['Data'] = df[col_data].apply(lambda x: robust_date_parser(str(x)[:10]))
            df = df.dropna(subset=['Data']).sort_values('Data')

        # Mapear colunas padronizadas
        TEXTO_COLS = ['type','name','date','start_date_local']
        for var, possiveis in MAPA_TRAINING.items():
            col = detectar_coluna(df, possiveis)
            if col:
                df[var] = df[col] if var in TEXTO_COLS else df[col].apply(br_to_float)

        # Normalizar tipo de actividade
        if 'type' in df.columns:
            df['type'] = df['type'].apply(normalizar_tipo)

        # Filtrar por período
        data_min = datetime.now() - timedelta(days=days_back)
        df = df[df['Data'] >= data_min].reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar atividades: {e}")
        return pd.DataFrame()

def filtrar_por_datas(df, data_ini, data_fim):
    if len(df) == 0: return df
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'])
    mask = (df['Data'].dt.date >= data_ini) & (df['Data'].dt.date <= data_fim)
    return df[mask].reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTROS GLOBAIS
# ════════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.image("https://img.icons8.com/emoji/96/runner-emoji.png", width=60)
    st.sidebar.title("ATHELTICA")
    st.sidebar.markdown("---")

    st.sidebar.header("⚙️ Filtros Globais")

    # Período
    dias_opcoes = {"30 dias": 30, "60 dias": 60, "90 dias": 90,
                   "180 dias": 180, "1 ano": 365, "2 anos": 730}
    periodo_label = st.sidebar.selectbox("📅 Período de análise", list(dias_opcoes.keys()), index=2)
    days_back = dias_opcoes[periodo_label]

    # Datas customizadas
    usar_datas_custom = st.sidebar.checkbox("Definir datas manualmente")
    if usar_datas_custom:
        data_ini = st.sidebar.date_input("Data início", value=datetime.now().date() - timedelta(days=days_back))
        data_fim = st.sidebar.date_input("Data fim",   value=datetime.now().date())
    else:
        data_fim = datetime.now().date()
        data_ini = data_fim - timedelta(days=days_back)

    st.sidebar.markdown("---")

    # Modalidades
    st.sidebar.header("🏃 Modalidades")
    modalidades_disp = ['Bike', 'Row', 'Run', 'Ski']
    modalidades_sel  = st.sidebar.multiselect(
        "Mostrar modalidades", modalidades_disp, default=modalidades_disp)

    st.sidebar.markdown("---")

    # Botão reload
    if st.sidebar.button("🔄 Recarregar dados"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dados de {data_ini.strftime('%d/%m/%Y')} a {data_fim.strftime('%d/%m/%Y')}")
    st.sidebar.caption(f"Carregado: {datetime.now().strftime('%H:%M')}")

    return days_back, data_ini, data_fim, modalidades_sel

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — KPIs + VISÃO GERAL
# ════════════════════════════════════════════════════════════════════════════════

def tab_visao_geral(df_well, df_act, data_ini, data_fim):
    st.header("📊 Visão Geral")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    n_sess   = len(df_act)
    hrv_med  = df_well['hrv'].dropna().tail(7).mean()  if 'hrv'  in df_well.columns else None
    rhr_ult  = df_well['rhr'].dropna().iloc[-1]        if 'rhr'  in df_well.columns and len(df_well) > 0 else None
    horas    = (df_act['moving_time'].sum() / 3600)    if 'moving_time' in df_act.columns else None

    with col1:
        st.metric("🏋️ Sessões", f"{n_sess}")
    with col2:
        st.metric("⏱️ Horas totais", f"{horas:.1f}h" if horas else "—")
    with col3:
        st.metric("💚 HRV médio (7d)", f"{hrv_med:.0f} ms" if hrv_med else "—")
    with col4:
        st.metric("❤️ RHR último", f"{rhr_ult:.0f} bpm" if rhr_ult else "—")

    st.markdown("---")

    # Performance Overview (CTL/ATL/HRV/Readiness normalizados)
    st.subheader("📈 Performance Overview")
    fig, ax = plt.subplots(figsize=(14, 5))
    tem = False
    if 'moving_time' in df_act.columns and 'rpe' in df_act.columns:
        df_load = df_act.copy()
        df_load['Data'] = pd.to_datetime(df_load['Data'])
        df_load['load'] = (df_load['moving_time'] / 60) * df_load['rpe'].fillna(0)
        load_daily = df_load.groupby('Data')['load'].sum().reset_index()
        load_norm  = normalizar_serie(load_daily['load'])
        ax.bar(load_daily['Data'], load_norm, color=CORES['cinza'], alpha=0.3, label='Load (norm)')
        tem = True
    if 'hrv' in df_well.columns:
        dw = df_well.dropna(subset=['hrv']).copy()
        dw['Data'] = pd.to_datetime(dw['Data'])
        ax.plot(dw['Data'], normalizar_serie(dw['hrv']), color=CORES['verde'],
                linewidth=2, linestyle='--', label='HRV (norm)')
        tem = True
    if not tem:
        ax.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Performance Overview', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Tabela atividades recentes
    st.subheader("📋 Atividades Recentes")
    df_tab = filtrar_atividades_principais(df_act).sort_values('Data', ascending=False).head(10).copy()
    if len(df_tab) > 0:
        cols_show = [c for c in ['Data','type','name','moving_time','rpe','power_avg','icu_eftp'] if c in df_tab.columns]
        df_show = df_tab[cols_show].copy()
        if 'moving_time' in df_show.columns:
            df_show['moving_time'] = df_show['moving_time'].apply(
                lambda x: f"{int(x/3600)}h{int((x%3600)/60):02d}m" if pd.notna(x) else '—')
        df_show.columns = [c.replace('_',' ').title() for c in df_show.columns]
        st.dataframe(df_show, use_container_width=True)
    else:
        st.info("Sem atividades no período.")

    # Distribuição donut
    st.subheader("🎯 Distribuição por Modalidade")
    df_dist = filtrar_atividades_principais(df_act).copy()
    if len(df_dist) > 0:
        df_dist['type'] = df_dist['type'].apply(normalizar_tipo)
        contagem = df_dist['type'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(contagem.values, labels=contagem.index, autopct='%1.1f%%',
                colors=[get_cor_atividade(t) for t in contagem.index],
                startangle=90, pctdistance=0.75,
                wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
        ax2.text(0, 0, f'{contagem.sum()}', fontsize=36, fontweight='bold', ha='center', va='center')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PMC (CTL/ATL/TSB + Training Load)
# ════════════════════════════════════════════════════════════════════════════════

def tab_pmc(df_act):
    st.header("📈 PMC — Performance Management Chart")

    if len(df_act) == 0:
        st.warning("Sem dados de atividades no período.")
        return

    # Calcular CTL/ATL/TSB a partir do training load diário
    df = filtrar_atividades_principais(df_act).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    if 'moving_time' not in df.columns or 'rpe' not in df.columns:
        st.warning("Colunas moving_time ou rpe não disponíveis.")
        return

    df['load'] = (df['moving_time'] / 60) * df['rpe'].fillna(0)

    # Usar icu_training_load se disponível, senão session_rpe
    if 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        load_col = 'icu_training_load'
    else:
        load_col = 'load'

    load_daily = df.groupby('Data')[load_col].sum().reset_index()
    load_daily.columns = ['Data', 'load_val']

    # Preencher datas sem treino com 0
    idx = pd.date_range(load_daily['Data'].min(), load_daily['Data'].max())
    load_daily = load_daily.set_index('Data').reindex(idx, fill_value=0).reset_index()
    load_daily.columns = ['Data', 'load_val']

    # CTL (42d EMA) / ATL (7d EMA)
    load_daily['CTL'] = load_daily['load_val'].ewm(span=42, adjust=False).mean()
    load_daily['ATL'] = load_daily['load_val'].ewm(span=7,  adjust=False).mean()
    load_daily['TSB'] = load_daily['CTL'] - load_daily['ATL']

    # Filtro de período
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        smooth = st.checkbox("Suavizar CTL/ATL (rolling 3d)", value=False)

    if smooth:
        load_daily['CTL'] = load_daily['CTL'].rolling(3, min_periods=1).mean()
        load_daily['ATL'] = load_daily['ATL'].rolling(3, min_periods=1).mean()

    # Gráfico PMC
    fig, (ax_pmc, ax_load) = plt.subplots(2, 1, figsize=(16, 9),
                                            gridspec_kw={'height_ratios': [2.5, 1]},
                                            sharex=True)
    fig.subplots_adjust(hspace=0.05)

    ax_pmc.plot(load_daily['Data'], load_daily['CTL'], label='CTL', color=CORES['azul'], linewidth=2.5)
    ax_pmc.plot(load_daily['Data'], load_daily['ATL'], label='ATL', color=CORES['vermelho'], linewidth=2.5)
    ax_pmc.fill_between(load_daily['Data'], 0, load_daily['TSB'],
                         where=(load_daily['TSB'] >= 0), color=CORES['verde'], alpha=0.25, label='TSB+')
    ax_pmc.fill_between(load_daily['Data'], 0, load_daily['TSB'],
                         where=(load_daily['TSB'] < 0),  color=CORES['vermelho'], alpha=0.20, label='TSB-')
    ax_pmc.axhline(0, color=CORES['cinza'], linestyle='--', linewidth=0.8)
    ax_pmc.set_ylabel('CTL / ATL / TSB', fontweight='bold')
    ax_pmc.legend(loc='upper left', fontsize=9)
    ax_pmc.grid(True, alpha=0.3)

    ult = load_daily.iloc[-1]
    ax_pmc.text(0.99, 0.97,
                f"CTL: {ult['CTL']:.0f}  |  ATL: {ult['ATL']:.0f}  |  TSB: {ult['TSB']:+.0f}",
                transform=ax_pmc.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax_pmc.set_title('PMC — CTL / ATL / TSB / Training Load', fontsize=14, fontweight='bold')

    ax_load.bar(load_daily['Data'], load_daily['load_val'],
                color=CORES['roxo'], alpha=0.65, width=0.8, label='Training Load')
    ax_load.set_ylabel('Load', fontweight='bold', fontsize=9)
    ax_load.legend(loc='upper left', fontsize=8)
    ax_load.grid(True, alpha=0.2, axis='y')
    ax_load.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Tabela resumo PMC
    st.subheader("📋 Resumo PMC")
    resumo = pd.DataFrame({
        'Métrica': ['CTL (atual)', 'ATL (atual)', 'TSB (atual)', 'CTL (máx)', 'ATL (máx)'],
        'Valor':   [f"{ult['CTL']:.1f}", f"{ult['ATL']:.1f}", f"{ult['TSB']:+.1f}",
                    f"{load_daily['CTL'].max():.1f}", f"{load_daily['ATL'].max():.1f}"]
    })
    st.dataframe(resumo, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — eFTP EVOLUÇÃO
# ════════════════════════════════════════════════════════════════════════════════

def tab_eftp(df_act, modalidades_sel):
    st.header("⚡ Evolução do eFTP por Modalidade")

    eftp_col = next((c for c in ['icu_eftp','eFTP','eftp'] if c in df_act.columns), None)
    if eftp_col is None:
        st.warning("Coluna eFTP não encontrada nos dados.")
        return

    df = filtrar_atividades_principais(df_act).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    df['type'] = df['type'].apply(normalizar_tipo)
    df['ano']  = df['Data'].dt.year
    df[eftp_col] = pd.to_numeric(df[eftp_col], errors='coerce')
    df = df[df['type'].isin(modalidades_sel)].dropna(subset=[eftp_col])
    df = df[df[eftp_col] > 50]

    if len(df) == 0:
        st.warning("Sem dados de eFTP no período.")
        return

    modalidades = [m for m in modalidades_sel if m in df['type'].values]
    anos = sorted(df['ano'].unique())
    CORES_ANO = ['#3498DB','#E74C3C','#2ECC71','#9B59B6','#F39C12']
    mapa_cor  = {a: CORES_ANO[i % len(CORES_ANO)] for i, a in enumerate(anos)}

    # Filtro de anos
    anos_sel = st.multiselect("Filtrar anos", anos, default=list(anos))
    df = df[df['ano'].isin(anos_sel)]

    n_mod = len(modalidades)
    if n_mod == 0:
        st.info("Nenhuma modalidade com dados eFTP.")
        return

    fig, axes = plt.subplots(1, n_mod, figsize=(7*n_mod, 6))
    if n_mod == 1: axes = [axes]

    for ax, mod in zip(axes, modalidades):
        df_mod = df[df['type'] == mod].sort_values('Data')
        cor_mod = get_cor_atividade(mod)

        for ano in anos_sel:
            df_ano = df_mod[df_mod['ano'] == ano]
            if len(df_ano) == 0: continue
            ax.scatter(df_ano['Data'], df_ano[eftp_col],
                       color=mapa_cor[ano], alpha=0.65, s=35, label=str(ano))
            if len(df_ano) >= 3:
                x_n = (df_ano['Data'] - df_ano['Data'].min()).dt.days.values
                coef = np.polyfit(x_n, df_ano[eftp_col].values, 1)
                x_p  = np.array([x_n.min(), x_n.max()])
                y_p  = np.poly1d(coef)(x_p)
                datas_p = [df_ano['Data'].min() + pd.Timedelta(days=int(x)) for x in x_p]
                ax.plot(datas_p, y_p, color=mapa_cor[ano], linewidth=2, linestyle='--', alpha=0.9)
                slope_mes = coef[0] * 30
                ax.annotate(f'{slope_mes:+.1f}W/mês', xy=(datas_p[1], y_p[1]),
                            xytext=(5, 2), textcoords='offset points',
                            fontsize=7.5, color=mapa_cor[ano], fontweight='bold')

        if len(df_mod) >= 5:
            roll = df_mod.set_index('Data')[eftp_col].resample('7D').mean().interpolate()
            ax.plot(roll.index, roll.values, color=cor_mod, linewidth=2, alpha=0.4, label='Média 7d')

        if len(df_mod) > 0:
            eftp_max = df_mod[eftp_col].max()
            ax.axhline(eftp_max, color=cor_mod, linestyle=':', linewidth=1.2, alpha=0.6)
            ax.annotate(f'Máx: {eftp_max:.0f}W',
                        xy=(df_mod.loc[df_mod[eftp_col].idxmax(), 'Data'], eftp_max),
                        xytext=(0, 6), textcoords='offset points',
                        fontsize=8, color=cor_mod, fontweight='bold', ha='center')

        ax.set_title(f'eFTP — {mod}', fontsize=13, fontweight='bold', color=cor_mod)
        ax.set_xlabel('Data'); ax.set_ylabel('eFTP (W)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    plt.suptitle('Evolução do eFTP por Modalidade', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — HR ZONES + RPE ZONES
# ════════════════════════════════════════════════════════════════════════════════

def tab_zones(df_act, modalidades_sel):
    st.header("❤️ HR Zones & RPE Zones")

    df = filtrar_atividades_principais(df_act).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    df['type'] = df['type'].apply(normalizar_tipo)
    df['ano']  = df['Data'].dt.year
    df = df[df['type'].isin(modalidades_sel)]

    if len(df) == 0:
        st.warning("Sem dados no período.")
        return

    anos = sorted(df['ano'].unique())
    ano_sel = st.selectbox("Selecionar ano", anos, index=len(anos)-1)
    df_ano  = df[df['ano'] == ano_sel].copy()

    zonas_hr_raw = [c for c in df.columns if c.lower().startswith('hr_z') and c.lower().endswith('_secs')]
    rpe_col      = next((c for c in ['rpe','RPE','icu_rpe'] if c in df.columns), None)

    CORES_HR  = {'Baixa (Z1+Z2)':'#2ECC71','Moderada (Z3+Z4)':'#F39C12','Alta (Z5+Z6+Z7)':'#E74C3C'}
    CORES_RPE = {'Leve (1–4)':'#3498DB','Moderado (5–6)':'#F39C12','Forte (7–10)':'#E74C3C'}

    def get_zona_num(col):
        m = re.search(r'hr_z(\d+)_secs', col.lower())
        return int(m.group(1)) if m else 0

    col_hr, col_rpe = st.columns(2)

    # ── HR Zones
    with col_hr:
        st.subheader(f"❤️ HR Zones — {ano_sel}")
        if zonas_hr_raw:
            df_hr = df_ano.copy()
            baixa    = [c for c in zonas_hr_raw if get_zona_num(c) in (1,2)]
            moderada = [c for c in zonas_hr_raw if get_zona_num(c) in (3,4)]
            alta     = [c for c in zonas_hr_raw if get_zona_num(c) in (5,6,7)]
            for cols in [baixa, moderada, alta]:
                for c in cols: df_hr[c] = pd.to_numeric(df_hr[c], errors='coerce').fillna(0)
            df_hr['z_b'] = df_hr[baixa].sum(axis=1)    if baixa    else 0
            df_hr['z_m'] = df_hr[moderada].sum(axis=1) if moderada else 0
            df_hr['z_a'] = df_hr[alta].sum(axis=1)     if alta     else 0
            df_hr['z_t'] = df_hr['z_b'] + df_hr['z_m'] + df_hr['z_a']
            df_hr = df_hr[df_hr['z_t'] > 0]
            if len(df_hr) > 0:
                for z, col in zip(['z_b','z_m','z_a'],['pct_b','pct_m','pct_a']):
                    df_hr[col] = df_hr[z] / df_hr['z_t'] * 100
                por_tipo = df_hr.groupby('type')[['pct_b','pct_m','pct_a']].mean()
                por_tipo.columns = list(CORES_HR.keys())
                fig, ax = plt.subplots(figsize=(6, 5))
                bottom = np.zeros(len(por_tipo))
                for zona, cor in CORES_HR.items():
                    vals = por_tipo[zona].values
                    ax.bar(por_tipo.index, vals, bottom=bottom, color=cor, label=zona,
                           edgecolor='white', linewidth=0.5)
                    for i,(v,b) in enumerate(zip(vals,bottom)):
                        if v > 5: ax.text(i, b+v/2, f'{v:.0f}%', ha='center', va='center',
                                          fontsize=9, fontweight='bold', color='white')
                    bottom += vals
                ax.set_ylim(0,100); ax.set_ylabel('% do tempo')
                ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2, axis='y')
                plt.tight_layout()
                st.pyplot(fig); plt.close()
        else:
            st.info("Sem colunas de HR zones nos dados.")

    # ── RPE Zones
    with col_rpe:
        st.subheader(f"🎯 RPE Zones — {ano_sel}")
        if rpe_col:
            df_rpe = df_ano.dropna(subset=[rpe_col]).copy()
            df_rpe[rpe_col] = pd.to_numeric(df_rpe[rpe_col], errors='coerce')
            df_rpe['rpe_zona'] = pd.cut(df_rpe[rpe_col], bins=[0,4.9,6.9,10],
                                         labels=list(CORES_RPE.keys()), right=True)
            df_rpe = df_rpe.dropna(subset=['rpe_zona'])
            if len(df_rpe) > 0:
                pivot = df_rpe.groupby(['type','rpe_zona'], observed=True).size().unstack(fill_value=0)
                for z in CORES_RPE.keys():
                    if z not in pivot.columns: pivot[z] = 0
                pivot = pivot[list(CORES_RPE.keys())]
                pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
                fig, ax = plt.subplots(figsize=(6,5))
                bottom = np.zeros(len(pct))
                for zona, cor in CORES_RPE.items():
                    vals = pct[zona].values
                    ax.bar(pct.index, vals, bottom=bottom, color=cor, label=zona,
                           edgecolor='white', linewidth=0.5)
                    for i,(v,b) in enumerate(zip(vals,bottom)):
                        if v > 5: ax.text(i, b+v/2, f'{v:.0f}%', ha='center', va='center',
                                          fontsize=9, fontweight='bold', color='white')
                    bottom += vals
                ax.set_ylim(0,100); ax.set_ylabel('% de sessões')
                ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2, axis='y')
                plt.tight_layout()
                st.pyplot(fig); plt.close()
        else:
            st.info("Sem coluna RPE nos dados.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — RECOVERY
# ════════════════════════════════════════════════════════════════════════════════

def calcular_recovery(df_well):
    if len(df_well) == 0: return pd.DataFrame()
    df = df_well.copy().sort_values('Data')
    df['hrv_baseline'] = df['hrv'].rolling(14, min_periods=7).mean()
    df['rhr_baseline'] = df['rhr'].rolling(14, min_periods=7).mean()
    df['hrv_cv_7d']    = calcular_cv_rolling(df['hrv'], 7)
    df['hrv_cv_30d']   = calcular_cv_rolling(df['hrv'], 30)

    rows = []
    for _, row in df.iterrows():
        hrv_v = row.get('hrv');    hrv_b = row.get('hrv_baseline'); cv = row.get('hrv_cv_7d')
        rhr_v = row.get('rhr');    rhr_b = row.get('rhr_baseline')

        # HRV score
        hrv_score = 50
        if pd.notna(hrv_v) and pd.notna(hrv_b) and hrv_b > 0 and pd.notna(cv):
            inf, sup = calcular_normal_range(hrv_b, cv)
            if inf and sup:
                band = sup - inf
                if hrv_v >= sup:   hrv_score = min(100, 75 + (hrv_v-sup)/band*25 if band>0 else 75)
                elif hrv_v <= inf: hrv_score = max(0,   40 - (inf-hrv_v)/band*40 if band>0 else 40)
                else:              hrv_score = 50 + ((hrv_v-inf)/band*25 if band>0 else 0)

        # RHR score (invertido)
        rhr_score = 50
        if pd.notna(rhr_v) and pd.notna(rhr_b) and rhr_b > 0:
            pct = (rhr_v - rhr_b) / rhr_b * 100
            rhr_score = 90 if pct<-10 else 75 if pct<-5 else 55 if pct<5 else 35 if pct<10 else 20

        sleep_s   = converter_1_5_para_0_100(row.get('sleep_quality'))
        fatiga_s  = converter_1_5_para_0_100(row.get('fatiga'))
        stress_s  = converter_1_5_para_0_100(row.get('stress'))
        humor_s   = converter_1_5_para_0_100(row.get('humor'))
        soreness_s= converter_1_5_para_0_100(row.get('soreness'))

        score = (hrv_score*0.30 + rhr_score*0.15 + sleep_s*0.20 +
                 fatiga_s*0.10 + stress_s*0.10 + humor_s*0.05 + soreness_s*0.05 + 50*0.05)

        inf_r, sup_r = (calcular_normal_range(hrv_b, cv)
                        if pd.notna(hrv_b) and pd.notna(cv) else (None, None))

        rows.append({'Data': row['Data'], 'recovery_score': score,
                     'hrv': hrv_v, 'hrv_baseline': hrv_b, 'hrv_cv_7d': cv,
                     'hrv_cv_30d': row.get('hrv_cv_30d'),
                     'normal_range_inf': inf_r, 'normal_range_sup': sup_r,
                     'hrv_component': hrv_score, 'rhr_component': rhr_score,
                     'sleep_component': sleep_s, 'fatiga_component': fatiga_s,
                     'stress_component': stress_s})
    return pd.DataFrame(rows)

def tab_recovery(df_well):
    st.header("🔋 Recovery Score")

    if len(df_well) == 0 or 'hrv' not in df_well.columns:
        st.warning("Sem dados de wellness/HRV no período.")
        return

    rec = calcular_recovery(df_well)
    if len(rec) == 0: return

    # KPIs Recovery
    ult = rec.iloc[-1]
    score = ult['recovery_score']
    cat   = ('🟢 Excelente' if score>=80 else '🟡 Bom' if score>=60
             else '🟠 Moderado' if score>=40 else '🔴 Baixo')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recovery Score", f"{score:.0f}/100", delta=cat)
    c2.metric("HRV atual",      f"{ult['hrv']:.0f} ms"      if pd.notna(ult['hrv'])      else "—")
    c3.metric("Baseline HRV",   f"{ult['hrv_baseline']:.0f} ms" if pd.notna(ult['hrv_baseline']) else "—")
    c4.metric("CV% 7d",         f"{ult['hrv_cv_7d']:.1f}%"  if pd.notna(ult['hrv_cv_7d'])  else "—")

    st.markdown("---")

    # Timeline Recovery Score
    st.subheader("📊 Recovery Score — Timeline")
    n_dias = st.slider("Dias a mostrar", 14, min(len(rec), 365), min(90, len(rec)))
    df_tl  = rec.tail(n_dias).copy()

    fig, ax = plt.subplots(figsize=(14,5))
    ax.axhspan(80,100, alpha=0.15, color=CORES['verde'],   label='Excelente (80–100)')
    ax.axhspan(60,80,  alpha=0.15, color=CORES['amarelo'], label='Bom (60–79)')
    ax.axhspan(40,60,  alpha=0.15, color=CORES['laranja'], label='Moderado (40–59)')
    ax.axhspan(0, 40,  alpha=0.15, color=CORES['vermelho'],label='Baixo (0–39)')
    x      = range(len(df_tl))
    scores = df_tl['recovery_score'].values
    cores_pts = [CORES['verde'] if s>=80 else CORES['amarelo'] if s>=60
                 else CORES['laranja'] if s>=40 else CORES['vermelho'] for s in scores]
    ax.plot(x, scores, color=CORES['azul_escuro'], linewidth=2, alpha=0.7)
    ax.scatter(x, scores, c=cores_pts, s=70, edgecolors='white', linewidths=2, zorder=5)
    if len(df_tl) >= 7:
        ax.plot(x, pd.Series(scores).rolling(7,min_periods=3).mean(),
                color=CORES['roxo'], linewidth=2.5, linestyle='--', label='Média 7d', alpha=0.8)
    datas = df_tl['Data'].dt.strftime('%d/%m')
    step  = max(1, len(x)//10)
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels([datas.iloc[i] for i in range(0,len(datas),step)], rotation=45)
    ax.set_ylim(0,105); ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # HRV Normal Range
    st.subheader("📊 HRV com Normal Range")
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,9), height_ratios=[2,1.2])

    df_hrv = df_tl.copy()
    xr = range(len(df_hrv))
    datas_r = df_hrv['Data'].dt.strftime('%d/%m')

    if df_hrv['normal_range_inf'].notna().any():
        ax1.fill_between(xr, df_hrv['normal_range_inf'], df_hrv['normal_range_sup'],
                         alpha=0.25, color=CORES['azul'], label='Normal Range HRV')
    if df_hrv['hrv_baseline'].notna().any():
        ax1.plot(xr, df_hrv['hrv_baseline'], color=CORES['roxo'], linestyle='--',
                 linewidth=2, label='Baseline 14d')

    hrv_vals = df_hrv['hrv'].values
    c_hrv = []
    for _, r in df_hrv.iterrows():
        h, i, s = r.get('hrv'), r.get('normal_range_inf'), r.get('normal_range_sup')
        if pd.isna(h) or pd.isna(i): c_hrv.append(CORES['cinza'])
        elif h > s:  c_hrv.append(CORES['verde'])
        elif h < i:  c_hrv.append(CORES['vermelho'])
        else:        c_hrv.append(CORES['azul'])

    ax1.plot(xr, hrv_vals, color=CORES['preto'], linewidth=2, alpha=0.6)
    ax1.scatter(xr, hrv_vals, c=c_hrv, s=70, edgecolors='white', linewidths=2, zorder=5)
    ax1.legend(loc='upper right', fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('HRV (ms)', fontweight='bold')
    ax1.set_title('HRV com Normal Range (HRV4Training)', fontsize=13, fontweight='bold')

    # CV% Normal Range
    if df_hrv['hrv_cv_7d'].notna().sum() >= 5:
        cv_s = df_hrv['hrv_cv_7d'].copy()
        cv_b = cv_s.rolling(14, min_periods=5).mean()
        cv_std = cv_s.rolling(14, min_periods=5).std()
        cv_inf = cv_b - 0.5 * cv_std
        cv_sup = cv_b + 0.5 * cv_std

        ax2.fill_between(xr, cv_inf, cv_sup, alpha=0.25, color=CORES['laranja'], label='Normal Range CV%')
        ax2.plot(xr, cv_b, color=CORES['roxo'], linestyle='--', linewidth=1.8, alpha=0.8)
        cv_vals = cv_s.values
        c_cv = []
        for i_r in range(len(df_hrv)):
            cv  = cv_vals[i_r]
            ci  = cv_inf.iloc[i_r] if not pd.isna(cv_inf.iloc[i_r]) else np.nan
            cs  = cv_sup.iloc[i_r] if not pd.isna(cv_sup.iloc[i_r]) else np.nan
            if pd.isna(cv) or pd.isna(ci): c_cv.append(CORES['cinza'])
            elif cv > cs: c_cv.append(CORES['verde'])
            elif cv < ci: c_cv.append(CORES['vermelho'])
            else:         c_cv.append(CORES['azul'])
        ax2.plot(xr, cv_vals, color=CORES['preto'], linewidth=1.8, alpha=0.6)
        ax2.scatter(xr, cv_vals, c=c_cv, s=55, edgecolors='white', linewidths=1.5, zorder=5)
        ax2.axhline(3,  color=CORES['cinza'], linestyle=':', alpha=0.5)
        ax2.axhline(10, color=CORES['cinza'], linestyle=':', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=8)

    step_r = max(1, len(xr)//10)
    for ax_r in [ax1, ax2]:
        ax_r.set_xticks(list(xr)[::step_r])
        ax_r.set_xticklabels([datas_r.iloc[i] for i in range(0,len(datas_r),step_r)], rotation=45)
        ax_r.grid(True, alpha=0.3)
    ax2.set_ylabel('CV% HRV', fontweight='bold')
    ax2.set_title('CV% com Normal Range', fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig2); plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — WELLNESS
# ════════════════════════════════════════════════════════════════════════════════

def tab_wellness(df_well):
    st.header("🧘 Wellness")

    if len(df_well) == 0:
        st.warning("Sem dados de wellness no período.")
        return

    metricas = [m for m in ['hrv','rhr','sleep_quality','fatiga','stress','humor','soreness']
                if m in df_well.columns and df_well[m].notna().any()]

    if not metricas:
        st.warning("Nenhuma métrica wellness disponível.")
        return

    # Selector de métricas
    sel = st.multiselect("Métricas a mostrar", metricas, default=metricas[:4])
    if not sel: return

    fig, axes = plt.subplots(len(sel), 1, figsize=(14, 3*len(sel)), sharex=True)
    if len(sel) == 1: axes = [axes]

    x = range(len(df_well))
    datas = pd.to_datetime(df_well['Data']).dt.strftime('%d/%m')
    CORES_MET = {'hrv':CORES['verde'],'rhr':CORES['vermelho'],'sleep_quality':CORES['roxo'],
                 'fatiga':CORES['laranja'],'stress':CORES['vermelho_escuro'],
                 'humor':CORES['verde_escuro'],'soreness':CORES['azul']}

    for ax, met in zip(axes, sel):
        vals = pd.to_numeric(df_well[met], errors='coerce').values
        cor  = CORES_MET.get(met, CORES['azul'])
        ax.plot(x, vals, color=cor, linewidth=2, marker='o', markersize=4)
        if len(vals) >= 7:
            ax.plot(x, pd.Series(vals).rolling(7, min_periods=3).mean(),
                    color=CORES['preto'], linewidth=1.5, linestyle='--', alpha=0.5, label='Média 7d')
        ax.set_ylabel(met.replace('_',' ').title(), fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    step = max(1, len(x)//12)
    axes[-1].set_xticks(list(x)[::step])
    axes[-1].set_xticklabels([datas.iloc[i] for i in range(0,len(datas),step)], rotation=45)
    plt.suptitle('Métricas Wellness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Tabela resumo wellness
    st.subheader("📋 Resumo Wellness (últimos 7 dias)")
    if len(df_well) >= 7:
        ult7 = df_well.tail(7)
        resumo = {}
        for m in metricas:
            col = pd.to_numeric(ult7[m], errors='coerce')
            resumo[m.replace('_',' ').title()] = [f"{col.mean():.1f}", f"{col.min():.0f}", f"{col.max():.0f}"]
        df_res = pd.DataFrame(resumo, index=['Média 7d','Mín','Máx']).T
        st.dataframe(df_res, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════════

def main():
    days_back, data_ini, data_fim, modalidades_sel = render_sidebar()

    st.title("🏃 ATHELTICA Analytics Dashboard")
    st.caption(f"Período: {data_ini.strftime('%d/%m/%Y')} → {data_fim.strftime('%d/%m/%Y')}  |  Modalidades: {', '.join(modalidades_sel)}")

    # Carregar dados
    with st.spinner("A carregar dados..."):
        df_well_raw = carregar_wellness(days_back)
        df_act_raw  = carregar_atividades(days_back)

    if df_well_raw.empty and df_act_raw.empty:
        st.error("Não foi possível carregar dados. Verifica as credenciais e os URLs das Google Sheets.")
        st.stop()

    # Preprocessing — mesmo pipeline do DataPreprocessor original
    df_well_clean = preprocessar_wellness(df_well_raw)
    df_act_clean  = preprocessar_atividades(df_act_raw)

    # Filtrar por período seleccionado
    df_well = filtrar_por_datas(df_well_clean, data_ini, data_fim)
    df_act  = filtrar_por_datas(df_act_clean,  data_ini, data_fim)

    # Filtrar por modalidade
    if len(df_act) > 0 and 'type' in df_act.columns:
        df_act_filt = df_act[df_act['type'].apply(normalizar_tipo).isin(modalidades_sel + ['WeightTraining'])]
    else:
        df_act_filt = df_act

    st.success(f"✅ {len(df_well)} registros wellness (de {len(df_well_raw)} raw)  |  {len(df_act_filt)} atividades (de {len(df_act_raw)} raw)")

    # ── DIAGNÓSTICO (expander)
    with st.expander("🔍 Diagnóstico de dados (clica para ver)", expanded=False):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**📋 Atividades (RAW — antes de filtrar)**")
            st.write(f"Total de linhas: {len(df_act_raw)}")
            if len(df_act_raw) > 0:
                df_act_raw2 = df_act_raw.copy()
                df_act_raw2['Data'] = pd.to_datetime(df_act_raw2['Data'])
                st.write(f"Data mínima: {df_act_raw2['Data'].min().date()}")
                st.write(f"Data máxima: {df_act_raw2['Data'].max().date()}")
                st.write(f"Todas as colunas ({len(df_act_raw2.columns)}): {list(df_act_raw2.columns)}")
                st.write("Últimas 5 linhas:")
                cols_show = [c for c in ['Data','type','name','power_avg','rpe','icu_eftp'] if c in df_act_raw2.columns]
                st.dataframe(df_act_raw2[cols_show].tail(5))
        with col_d2:
            st.markdown("**📋 Wellness (RAW — antes de filtrar)**")
            st.write(f"Total de linhas: {len(df_well_raw)}")
            if len(df_well_raw) > 0:
                df_well_raw2 = df_well_raw.copy()
                df_well_raw2['Data'] = pd.to_datetime(df_well_raw2['Data'])
                st.write(f"Data mínima: {df_well_raw2['Data'].min().date()}")
                st.write(f"Data máxima: {df_well_raw2['Data'].max().date()}")
                st.write(f"Todas as colunas ({len(df_well_raw2.columns)}): {list(df_well_raw2.columns)}")
                st.write("Últimas 5 linhas:")
                cols_w = [c for c in ['Data','hrv','rhr','sleep_quality','fatiga','stress'] if c in df_well_raw2.columns]
                st.dataframe(df_well_raw2[cols_w].tail(5))

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Visão Geral",
        "📈 PMC",
        "⚡ eFTP",
        "❤️ HR & RPE Zones",
        "🔋 Recovery",
        "🧘 Wellness",
    ])

    with tab1: tab_visao_geral(df_well, df_act_filt, data_ini, data_fim)
    with tab2: tab_pmc(df_act_filt)
    with tab3: tab_eftp(df_act_filt, modalidades_sel)
    with tab4: tab_zones(df_act_filt, modalidades_sel)
    with tab5: tab_recovery(df_well)
    with tab6: tab_wellness(df_well)

if __name__ == "__main__":
    main()
