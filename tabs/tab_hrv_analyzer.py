"""
tab_hrv_analyzer.py — Recovery Pattern Analyzer
================================================
Módulo N=1 de análise fisiológica: o que o HRV (rMSSD) diz sobre o treino
em períodos específicos vs períodos anteriores.

Arquitectura:
  1. Construção do sinal HRV normalizado (rMSSD, ln_rMSSD, rMSSD_norm, ratio HRV/RHR)
  2. Detecção automática de períodos (HRV↑ vs HRV↓)
  3. Event window analysis: 14d antes, período, 7d depois
  4. Lag correlation: qual variável de treino antecede as mudanças de HRV
  5. Comparação Before/After: quais variáveis mudaram e quando
  6. Padrões recorrentes: "top 10% HRV days — o que aconteceu antes"
  7. Fingerprints de recovery vs suppression

Métricas HRV usadas:
  rMSSD       — sinal base
  ln_rMSSD    — log-normalizado (literatura padrão)
  AVNN        — 60000 / HR  (espaço temporal por batimento)
  rMSSD_norm  — (rMSSD / AVNN) × 100  = variabilidade relativa à FC
  HRV_RHR_r   — coupling autonómico

Análises estatísticas:
  rolling mean / z-score / EWMA / slope
  cross-correlação com lag 1-14d
  Cohen's d entre períodos
  event windows (alinhamento em torno de mudanças)
"""

from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

MC = {'displayModeBar': False, 'responsive': True, 'scrollZoom': False}
_C = {'primary': '#2980b9', 'hrv_up': '#27ae60', 'hrv_dn': '#e74c3c',
      'neutral': '#7f8c8d', 'load': '#e67e22', 'accent': '#8e44ad',
      'bg': 'white', 'grid': '#eee', 'font': '#111'}


# ── A. Construção do sinal HRV ────────────────────────────────────────────────

def _build_hrv_signal(dw: pd.DataFrame) -> pd.DataFrame:
    """
    A partir do DataFrame de wellness, constrói série diária com:
      hrv, rhr, ln_hrv, avnn, hrv_norm, hrv_rhr_ratio
      rolling baselines 7d / 28d, z-scores, slopes
    """
    df = dw.copy()
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').reset_index(drop=True)

    # Colunas numéricas
    for col in ['hrv', 'rhr', 'sleep_hours', 'sleep_quality',
                'stress', 'fatiga', 'soreness', 'humor']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ln(rMSSD) — sinal padrão na literatura
    if 'hrv' in df.columns:
        df['ln_hrv'] = np.log(df['hrv'].clip(lower=0.01))

        # AVNN = 60000 / HR (ms por batimento)
        if 'rhr' in df.columns:
            df['avnn'] = (60000 / df['rhr'].replace(0, np.nan))
            # rMSSD normalizado: (rMSSD / AVNN) × 100
            # Mede variabilidade relativa ao espaço temporal disponível
            df['hrv_norm'] = (df['hrv'] / df['avnn']) * 100
            # Coupling autonómico: HRV/RHR — inversão esperada em boa adaptação
            df['hrv_rhr_ratio'] = df['hrv'] / df['rhr'].replace(0, np.nan)
        else:
            df['avnn'] = np.nan
            df['hrv_norm'] = np.nan
            df['hrv_rhr_ratio'] = np.nan

        # Rolling stats — 7d e 28d
        for w, sfx in [(7, '7d'), (28, '28d')]:
            df[f'hrv_mean_{sfx}']  = df['hrv'].rolling(w, min_periods=3).mean()
            df[f'hrv_std_{sfx}']   = df['hrv'].rolling(w, min_periods=3).std()
            df[f'ln_hrv_mean_{sfx}'] = df['ln_hrv'].rolling(w, min_periods=3).mean()

        # Z-score vs baseline 28d
        df['hrv_z28'] = ((df['hrv'] - df['hrv_mean_28d']) /
                          df['hrv_std_28d'].replace(0, np.nan))

        # EWMA (alpha=0.1 ≈ span=19)
        df['hrv_ewma'] = df['hrv'].ewm(span=19, adjust=False).mean()
        df['ln_hrv_ewma'] = df['ln_hrv'].ewm(span=19, adjust=False).mean()

        # Slope 7d (via polyfit rolling — simplificado)
        slopes = np.full(len(df), np.nan)
        for i in range(6, len(df)):
            y = df['hrv'].iloc[i-6:i+1].values
            if np.sum(~np.isnan(y)) >= 4:
                x = np.arange(len(y), dtype=float)
                valid = ~np.isnan(y)
                try:
                    z = np.polyfit(x[valid], y[valid], 1)
                    slopes[i] = z[0]
                except Exception:
                    pass
        df['hrv_slope_7d'] = slopes

    if 'rhr' in df.columns:
        df['rhr_mean_28d'] = df['rhr'].rolling(28, min_periods=7).mean()
        df['rhr_z28'] = ((df['rhr'] - df['rhr_mean_28d']) /
                          df['rhr'].rolling(28, min_periods=7).std().replace(0, np.nan))

    return df


# ── B. Detecção automática de períodos ───────────────────────────────────────

def _detect_hrv_periods(sig: pd.DataFrame,
                        min_len: int = 5,
                        z_thresh: float = 0.5) -> list[dict]:
    """
    Detecta períodos de HRV↑ (z28 > z_thresh) e HRV↓ (z28 < -z_thresh)
    com duração mínima min_len dias.
    Retorna lista de dicts {start, end, tipo, mean_z, delta_hrv}
    """
    if 'hrv_z28' not in sig.columns:
        return []

    z = sig['hrv_z28'].fillna(0).values
    dates = pd.to_datetime(sig['Data']).values
    hrv   = sig['hrv'].values

    periods = []
    i = 0
    while i < len(z):
        if z[i] > z_thresh:
            j = i
            while j < len(z) and z[j] > 0:
                j += 1
            if j - i >= min_len:
                periods.append({
                    'start': pd.Timestamp(dates[i]),
                    'end':   pd.Timestamp(dates[j-1]),
                    'tipo':  'HRV↑',
                    'mean_z': float(np.nanmean(z[i:j])),
                    'delta_hrv': float(np.nanmean(hrv[i:j]) -
                                       np.nanmean(hrv[max(0,i-14):i])),
                })
            i = j
        elif z[i] < -z_thresh:
            j = i
            while j < len(z) and z[j] < 0:
                j += 1
            if j - i >= min_len:
                periods.append({
                    'start': pd.Timestamp(dates[i]),
                    'end':   pd.Timestamp(dates[j-1]),
                    'tipo':  'HRV↓',
                    'mean_z': float(np.nanmean(z[i:j])),
                    'delta_hrv': float(np.nanmean(hrv[i:j]) -
                                       np.nanmean(hrv[max(0,i-14):i])),
                })
            i = j
        else:
            i += 1
    return periods


# ── C. Construir DataFrame de treino diário ───────────────────────────────────

def _build_training_signal(da: pd.DataFrame) -> pd.DataFrame:
    """
    Série diária de variáveis de treino: load, kJ, ATL, CTL, TSB,
    freq_sessoes, monotonia, strain, rpe_medio, duracao, dist_z3.
    """
    if da is None or len(da) == 0:
        return pd.DataFrame()

    df = filtrar_principais(da).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    for col in ['icu_training_load', 'moving_time', 'rpe', 'icu_joules',
                'distance', 'icu_atl', 'icu_ctl']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['dur_min'] = df['moving_time'].fillna(0) / 60
    df['rpe_n']   = df['rpe'] if 'rpe' in df.columns else np.nan
    df['load_rpe']= df['dur_min'] * pd.to_numeric(df.get('rpe_n', 0), errors='coerce').fillna(5)

    if 'icu_training_load' in df.columns:
        df['load'] = df['icu_training_load'].fillna(0)
    else:
        df['load'] = df['load_rpe']

    df['kj'] = pd.to_numeric(df.get('icu_joules', pd.Series(dtype=float)),
                              errors='coerce').fillna(0) / 1000
    df['dist_km'] = pd.to_numeric(df.get('distance', pd.Series(dtype=float)),
                                   errors='coerce').fillna(0) / 1000

    # Z3 proxy: RPE ≥ 7
    df['is_z3'] = (pd.to_numeric(df.get('rpe_n', pd.Series(dtype=float)),
                                   errors='coerce').fillna(0) >= 7).astype(float)
    df['load_z3'] = df['load'] * df['is_z3']

    # Agregar por dia
    daily = df.groupby('Data').agg(
        load     = ('load',     'sum'),
        kj       = ('kj',       'sum'),
        dur_min  = ('dur_min',  'sum'),
        n_sess   = ('load',     'count'),
        load_z3  = ('load_z3',  'sum'),
        dist_km  = ('dist_km',  'sum'),
        rpe_med  = ('rpe_n',    'mean'),
    ).reset_index()

    # Reindexar
    date_range = pd.date_range(daily['Data'].min(), pd.Timestamp.now().date())
    daily = daily.set_index('Data').reindex(date_range, fill_value=0).reset_index()
    daily.columns = ['Data'] + list(daily.columns[1:])
    daily['n_sess'] = daily['n_sess'].clip(lower=0)

    # Rolling vars
    daily['atl']   = daily['load'].ewm(span=7,  adjust=False).mean()
    daily['ctl']   = daily['load'].ewm(span=42, adjust=False).mean()
    daily['tsb']   = daily['ctl'] - daily['atl']
    daily['load_7d']  = daily['load'].rolling(7,  min_periods=1).sum()
    daily['load_28d'] = daily['load'].rolling(28, min_periods=7).sum()
    daily['load_z7d_pct'] = (
        (daily['load_7d'] / daily['load_28d'].replace(0, np.nan) * 4 - 1) * 100
    )

    # Monotonia (Banister): media / std da carga 7d
    daily['mono_7d'] = (
        daily['load'].rolling(7, min_periods=3).mean() /
        daily['load'].rolling(7, min_periods=3).std().replace(0, np.nan)
    )
    daily['strain_7d'] = daily['load_7d'] * daily['mono_7d']

    # Pct Z3
    daily['pct_z3'] = (
        daily['load_z3'].rolling(7, min_periods=1).sum() /
        daily['load_7d'].replace(0, np.nan) * 100
    )

    # Freq semanal rolling
    daily['freq_7d'] = daily['n_sess'].rolling(7, min_periods=1).sum()

    return daily


# ── D. Event Window Analysis ──────────────────────────────────────────────────

def _event_window(sig_hrv: pd.DataFrame, sig_train: pd.DataFrame,
                  event_dates: list,
                  pre_days: int = 14, post_days: int = 7,
                  train_vars: list = None) -> pd.DataFrame:
    """
    Para cada evento (data), extrai janela [-pre, +post] dias.
    Normaliza cada série pela sua média no período pré.
    Retorna DataFrame alinhado em torno de lag=0 (dia do evento).
    """
    if train_vars is None:
        train_vars = ['load', 'load_7d', 'kj', 'dur_min', 'n_sess',
                      'pct_z3', 'freq_7d', 'mono_7d', 'atl', 'ctl', 'tsb']

    # HRV vars
    hrv_vars = ['hrv', 'ln_hrv', 'hrv_norm', 'hrv_z28', 'rhr']
    all_vars = hrv_vars + train_vars

    merged = pd.merge(
        sig_hrv[['Data'] + [v for v in hrv_vars if v in sig_hrv.columns]],
        sig_train[['Data'] + [v for v in train_vars if v in sig_train.columns]],
        on='Data', how='outer'
    ).sort_values('Data')
    merged['Data'] = pd.to_datetime(merged['Data'])

    windows = []
    for evt in event_dates:
        evt = pd.Timestamp(evt)
        d0 = evt - pd.Timedelta(days=pre_days)
        d1 = evt + pd.Timedelta(days=post_days)
        sub = merged[(merged['Data'] >= d0) & (merged['Data'] <= d1)].copy()
        sub['lag'] = (sub['Data'] - evt).dt.days
        sub['event'] = evt.strftime('%Y-%m-%d')
        windows.append(sub)

    if not windows:
        return pd.DataFrame()
    return pd.concat(windows, ignore_index=True)


# ── E. Lag Correlation ────────────────────────────────────────────────────────

def _lag_correlations(sig_hrv: pd.DataFrame, sig_train: pd.DataFrame,
                      hrv_var: str = 'hrv',
                      train_vars: list = None,
                      max_lag: int = 14) -> pd.DataFrame:
    """
    Calcula correlação cruzada entre cada variável de treino
    e HRV com lag 0..max_lag dias (treino precede HRV).
    Retorna DataFrame {var, lag, r, p, interpretacao}
    """
    if train_vars is None:
        train_vars = ['load', 'kj', 'dur_min', 'pct_z3',
                      'freq_7d', 'mono_7d', 'strain_7d', 'tsb', 'atl']

    merged = pd.merge(
        sig_hrv[['Data', hrv_var]].rename(columns={hrv_var: 'hrv_tgt'}),
        sig_train[['Data'] + [v for v in train_vars if v in sig_train.columns]],
        on='Data', how='inner'
    ).sort_values('Data')

    hrv_s = merged['hrv_tgt'].values
    rows   = []
    for var in train_vars:
        if var not in merged.columns:
            continue
        x = merged[var].values
        for lag in range(0, max_lag + 1):
            if lag == 0:
                xv = x
                yv = hrv_s
            else:
                xv = x[:-lag]
                yv = hrv_s[lag:]
            valid = ~(np.isnan(xv) | np.isnan(yv))
            if valid.sum() < 20:
                continue
            try:
                r, p = scipy_stats.pearsonr(xv[valid], yv[valid])
            except Exception:
                r, p = np.nan, np.nan
            rows.append({'var': var, 'lag': lag, 'r': r, 'p': p,
                         'r_abs': abs(r)})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['sig'] = df['p'] < 0.05
    df['interp'] = df.apply(lambda row:
        f"{'↑' if row['r'] > 0 else '↓'} HRV com {row['lag']}d de lag"
        if row['sig'] else 'ns', axis=1)
    return df


# ── F. Comparação Before/After ───────────────────────────────────────────────

def _compare_periods(sig_hrv: pd.DataFrame, sig_train: pd.DataFrame,
                     start: pd.Timestamp, end: pd.Timestamp,
                     ref_days: int = 14) -> pd.DataFrame:
    """
    Compara o período [start, end] com os ref_days anteriores.
    Retorna tabela de variáveis com: before_mean, target_mean, delta%, cohen_d
    """
    merged = pd.merge(
        sig_hrv,
        sig_train,
        on='Data', how='outer'
    ).sort_values('Data')
    merged['Data'] = pd.to_datetime(merged['Data'])

    ref_start = start - pd.Timedelta(days=ref_days)
    before = merged[(merged['Data'] >= ref_start) & (merged['Data'] < start)]
    target = merged[(merged['Data'] >= start)     & (merged['Data'] <= end)]

    vars_to_compare = [
        ('load',       'Carga (TSS/dia)'),
        ('kj',         'kJ/dia'),
        ('dur_min',    'Duração (min/dia)'),
        ('n_sess',     'Sessões/dia'),
        ('pct_z3',     '% carga Z3'),
        ('freq_7d',    'Freq. semanal rolling'),
        ('mono_7d',    'Monotonia'),
        ('strain_7d',  'Strain'),
        ('tsb',        'TSB'),
        ('atl',        'ATL'),
        ('ctl',        'CTL'),
        ('hrv',        'HRV (rMSSD)'),
        ('ln_hrv',     'ln(rMSSD)'),
        ('hrv_norm',   'rMSSD norm. (÷AVNN×100)'),
        ('rhr',        'RHR (bpm)'),
        ('hrv_rhr_ratio', 'HRV/RHR coupling'),
    ]

    rows = []
    for col, label in vars_to_compare:
        if col not in merged.columns:
            continue
        b_vals = before[col].dropna().values
        t_vals = target[col].dropna().values
        if len(b_vals) < 3 or len(t_vals) < 3:
            continue
        b_m = float(np.mean(b_vals))
        t_m = float(np.mean(t_vals))
        delta_pct = (t_m - b_m) / abs(b_m) * 100 if b_m != 0 else np.nan
        # Cohen's d
        pooled_std = np.sqrt((np.std(b_vals)**2 + np.std(t_vals)**2) / 2)
        cohen_d = (t_m - b_m) / pooled_std if pooled_std > 0 else np.nan
        _, p_val = scipy_stats.mannwhitneyu(b_vals, t_vals, alternative='two-sided') \
            if len(b_vals) >= 3 and len(t_vals) >= 3 else (np.nan, np.nan)
        rows.append({
            'Variável':    label,
            'col':         col,
            'Antes':       round(b_m, 2),
            'Período':     round(t_m, 2),
            'Δ%':          round(delta_pct, 1),
            "Cohen's d":   round(cohen_d, 2),
            'p-valor':     round(p_val, 3) if not np.isnan(p_val) else '—',
            'sig':         p_val < 0.05 if not np.isnan(p_val) else False,
        })
    return pd.DataFrame(rows)


# ── G. Fingerprint: Top vs Bottom HRV days ───────────────────────────────────

def _hrv_fingerprint(sig_hrv: pd.DataFrame, sig_train: pd.DataFrame,
                     pct: float = 0.10,
                     pre_days: int = 10) -> dict:
    """
    Compara o que aconteceu nos [pre_days] dias antes dos:
      top pct% dias de HRV  vs  bottom pct% dias de HRV
    Retorna dict {top, bottom, diff} com médias de cada variável de treino.
    """
    merged = pd.merge(sig_hrv[['Data','hrv']], sig_train,
                      on='Data', how='inner').sort_values('Data')
    merged['Data'] = pd.to_datetime(merged['Data'])

    hrv_vals = merged['hrv'].dropna()
    q_top = hrv_vals.quantile(1 - pct)
    q_bot = hrv_vals.quantile(pct)

    top_days = merged[merged['hrv'] >= q_top]['Data'].values
    bot_days = merged[merged['hrv'] <= q_bot]['Data'].values

    train_vars = ['load', 'kj', 'dur_min', 'pct_z3', 'freq_7d',
                  'mono_7d', 'strain_7d', 'tsb', 'atl', 'n_sess']

    def _pre_window_mean(days, var):
        vals = []
        for d in days:
            d = pd.Timestamp(d)
            sub = merged[(merged['Data'] >= d - pd.Timedelta(days=pre_days)) &
                         (merged['Data'] < d)][var].dropna()
            if len(sub) >= 3:
                vals.append(float(sub.mean()))
        return np.nanmean(vals) if vals else np.nan

    result = {}
    for var in train_vars:
        if var not in merged.columns:
            continue
        top_m = _pre_window_mean(top_days, var)
        bot_m = _pre_window_mean(bot_days, var)
        diff  = (top_m - bot_m) / abs(bot_m) * 100 if bot_m != 0 else np.nan
        result[var] = {'top': top_m, 'bot': bot_m, 'diff_pct': diff}

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TAB PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def tab_hrv_analyzer(dw: pd.DataFrame, da: pd.DataFrame,
                     wc_full: pd.DataFrame = None, da_full: pd.DataFrame = None):
    """
    Recovery Pattern Analyzer — tab principal.
    dw / wc_full : DataFrame de wellness
    da / da_full : DataFrame de actividades
    """
    st.subheader("🔬 Recovery Pattern Analyzer")
    st.caption(
        "Análise N=1 longitudinal: o que o HRV (rMSSD) diz sobre o treino em "
        "diferentes períodos. Causalidade temporal, lag correlation, event windows, "
        "fingerprints de recovery vs suppression."
    )

    # Dados
    _dw = wc_full if wc_full is not None else dw
    _da = da_full if da_full is not None else da

    if _dw is None or len(_dw) == 0:
        st.warning("Sem dados de wellness. Verifica a ligação à Google Sheet.")
        return
    if 'hrv' not in _dw.columns or _dw['hrv'].notna().sum() < 14:
        st.warning("Sem dados suficientes de HRV (mínimo 14 dias).")
        return

    # ── Filtro de datas ───────────────────────────────────────────────────────
    _dw_all = _dw.copy()
    _dw_all['Data'] = pd.to_datetime(_dw_all['Data'])
    _da_all = _da.copy() if _da is not None else None
    if _da_all is not None:
        _da_all['Data'] = pd.to_datetime(_da_all['Data'])

    _date_min_data = _dw_all['Data'].min().date()
    _date_max_data = _dw_all['Data'].max().date()

    _fc1, _fc2 = st.columns(2)
    with _fc1:
        _filter_from = st.date_input(
            "📅 Analisar a partir de",
            value=max(_date_min_data, pd.Timestamp('2023-01-01').date()),
            min_value=_date_min_data,
            max_value=_date_max_data,
            key="hrv_filter_from",
            help="Exclui dados anteriores a esta data de TODAS as análises. "
                 "Útil para ignorar períodos com dados incompletos ou de treino muito diferente."
        )
    with _fc2:
        _filter_to = st.date_input(
            "Até",
            value=_date_max_data,
            min_value=_date_min_data,
            max_value=_date_max_data,
            key="hrv_filter_to",
            help="Data final da análise."
        )

    # Aplicar filtro
    _dw = _dw_all[
        (_dw_all['Data'].dt.date >= _filter_from) &
        (_dw_all['Data'].dt.date <= _filter_to)
    ].reset_index(drop=True)
    if _da_all is not None:
        _da = _da_all[
            (_da_all['Data'].dt.date >= _filter_from) &
            (_da_all['Data'].dt.date <= _filter_to)
        ].reset_index(drop=True)

    _n_hrv = _dw['hrv'].notna().sum()
    if _n_hrv < 14:
        st.warning(f"Apenas {_n_hrv} dias de HRV no período seleccionado (mínimo 14). "
                   "Alarga o intervalo de datas.")
        return

    st.caption(
        f"📅 Período de análise: **{_filter_from}** → **{_filter_to}** "
        f"({(_filter_to - _filter_from).days} dias | {_n_hrv} medições HRV)"
    )

    # ── Construir sinais ──────────────────────────────────────────────────────
    with st.spinner("A construir sinais HRV e treino..."):
        sig_hrv   = _build_hrv_signal(_dw)
        sig_train = _build_training_signal(_da) if _da is not None else pd.DataFrame()

    # ── Selector de análise (botões principais) ───────────────────────────────
    st.markdown("---")
    _btns = st.columns(4)
    _analyses = ["📅 Período Manual", "🔍 Detecção Automática",
                 "🔗 Lag Correlation", "🧬 Fingerprint HRV"]
    _mode = None
    for i, (col, lbl) in enumerate(zip(_btns, _analyses)):
        if col.button(lbl, use_container_width=True, key=f"hrv_mode_{i}"):
            st.session_state['hrv_mode'] = i

    _mode = st.session_state.get('hrv_mode', 0)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. PAINEL SUPERIOR — Sinal HRV completo (sempre visível)
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("📊 Sinal HRV — visão geral completa", expanded=True):
        _metric_choice = st.radio(
            "Métrica HRV a visualizar",
            ["rMSSD absoluto", "ln(rMSSD)", "rMSSD normalizado (÷AVNN×100)",
             "HRV/RHR coupling"],
            horizontal=True, key="hrv_metric_choice"
        )
        _col_map = {
            "rMSSD absoluto":                  "hrv",
            "ln(rMSSD)":                       "ln_hrv",
            "rMSSD normalizado (÷AVNN×100)":   "hrv_norm",
            "HRV/RHR coupling":                "hrv_rhr_ratio",
        }
        _yvar = _col_map[_metric_choice]

        if _yvar not in sig_hrv.columns:
            st.warning(f"Coluna {_yvar} não disponível (falta RHR?).")
        else:
            _fig_hrv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     row_heights=[0.65, 0.35],
                                     vertical_spacing=0.04)

            # Sinal principal
            _fig_hrv.add_trace(go.Scatter(
                x=sig_hrv['Data'], y=sig_hrv[_yvar],
                mode='lines', name=_metric_choice,
                line=dict(color=_C['hrv_up'], width=1.5),
                hovertemplate='%{x|%d/%m/%Y}<br>HRV: <b>%{y:.1f}</b><extra></extra>'
            ), row=1, col=1)

            # EWMA
            _ewma_col = 'ln_hrv_ewma' if 'ln' in _yvar else 'hrv_ewma'
            if _ewma_col in sig_hrv.columns:
                _fig_hrv.add_trace(go.Scatter(
                    x=sig_hrv['Data'], y=sig_hrv[_ewma_col],
                    mode='lines', name='EWMA (span=19)',
                    line=dict(color=_C['accent'], width=2, dash='dash'),
                    hovertemplate='%{x|%d/%m/%Y}<br>EWMA: <b>%{y:.1f}</b><extra></extra>'
                ), row=1, col=1)

            # Banda baseline ±1 std (28d)
            if 'hrv_mean_28d' in sig_hrv.columns and _yvar == 'hrv':
                _fig_hrv.add_trace(go.Scatter(
                    x=sig_hrv['Data'],
                    y=sig_hrv['hrv_mean_28d'] + sig_hrv['hrv_std_28d'],
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ), row=1, col=1)
                _fig_hrv.add_trace(go.Scatter(
                    x=sig_hrv['Data'],
                    y=sig_hrv['hrv_mean_28d'] - sig_hrv['hrv_std_28d'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(39,174,96,0.12)',
                    name='Banda ±1σ (28d)', hoverinfo='skip'
                ), row=1, col=1)

            # RHR no painel inferior (se disponível)
            if 'rhr' in sig_hrv.columns:
                _fig_hrv.add_trace(go.Scatter(
                    x=sig_hrv['Data'], y=sig_hrv['rhr'],
                    mode='lines', name='RHR (bpm)',
                    line=dict(color=_C['hrv_dn'], width=1.5),
                    hovertemplate='%{x|%d/%m/%Y}<br>RHR: <b>%{y:.0f}</b> bpm<extra></extra>'
                ), row=2, col=1)

            # Z-score overlay (eixo secundário seria complexo — usar cor de fundo)
            # Marcar zonas z>0.5 a verde e z<-0.5 a vermelho
            if 'hrv_z28' in sig_hrv.columns:
                _hv = sig_hrv[sig_hrv['hrv_z28'] > 0.5]
                _hd = sig_hrv[sig_hrv['hrv_z28'] < -0.5]
                for _row_mark, _df_mark, _col_mark in [
                    (1, _hv, 'rgba(39,174,96,0.15)'),
                    (1, _hd, 'rgba(231,76,60,0.15)'),
                ]:
                    if len(_df_mark) > 0:
                        _dates_m = _df_mark['Data']
                        _y_m     = _df_mark[_yvar]
                        _fig_hrv.add_trace(go.Scatter(
                            x=_dates_m, y=_y_m,
                            mode='markers',
                            marker=dict(size=5, color=_col_mark.replace('0.15', '0.6'),
                                        line=dict(width=0)),
                            name='HRV↑ (z>0.5)' if 'verde' in _col_mark or '174' in _col_mark
                                 else 'HRV↓ (z<-0.5)',
                            showlegend=True, hoverinfo='skip'
                        ), row=_row_mark, col=1)

            _fig_hrv.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#111', size=11),
                height=420, hovermode='x unified',
                margin=dict(t=20, b=50, l=55, r=20),
                legend=dict(orientation='h', y=-0.16,
                            font=dict(color='#111', size=10)),
            )
            _fig_hrv.update_xaxes(showgrid=True, gridcolor='#eee',
                                   tickfont=dict(color='#111'))
            _fig_hrv.update_yaxes(showgrid=True, gridcolor='#eee',
                                   tickfont=dict(color='#111'))
            st.plotly_chart(_fig_hrv, use_container_width=True,
                            config=MC, key='hrv_main_plot')

            # Cards de resumo
            _c1, _c2, _c3, _c4, _c5 = st.columns(5)
            _hrv_now  = sig_hrv['hrv'].dropna().iloc[-1] if sig_hrv['hrv'].notna().any() else np.nan
            _hrv_b28  = sig_hrv['hrv_mean_28d'].dropna().iloc[-1] if 'hrv_mean_28d' in sig_hrv.columns else np.nan
            _hrv_z    = sig_hrv['hrv_z28'].dropna().iloc[-1] if 'hrv_z28' in sig_hrv.columns else np.nan
            _hrv_slp  = sig_hrv['hrv_slope_7d'].dropna().iloc[-1] if 'hrv_slope_7d' in sig_hrv.columns else np.nan
            _ln_now   = sig_hrv['ln_hrv'].dropna().iloc[-1] if 'ln_hrv' in sig_hrv.columns else np.nan
            _norm_now = sig_hrv['hrv_norm'].dropna().iloc[-1] if 'hrv_norm' in sig_hrv.columns else np.nan
            _rhr_now  = sig_hrv['rhr'].dropna().iloc[-1] if 'rhr' in sig_hrv.columns else np.nan

            _c1.metric("rMSSD hoje", f"{_hrv_now:.0f} ms" if not np.isnan(_hrv_now) else "—",
                       delta=f"base: {_hrv_b28:.0f}" if not np.isnan(_hrv_b28) else None,
                       help="rMSSD absoluto. Baseline = média 28d.")
            _c2.metric("ln(rMSSD)", f"{_ln_now:.2f}" if not np.isnan(_ln_now) else "—",
                       help="Logaritmo natural do rMSSD — distribuição mais normal.")
            _c3.metric("rMSSD norm.", f"{_norm_now:.2f}" if not np.isnan(_norm_now) else "—",
                       help="(rMSSD÷AVNN)×100 — variabilidade relativa à FC de repouso.")
            _c4.metric("z-score 28d", f"{_hrv_z:+.2f}" if not np.isnan(_hrv_z) else "—",
                       delta="acima baseline" if (not np.isnan(_hrv_z) and _hrv_z > 0) else "abaixo baseline",
                       delta_color="normal" if (not np.isnan(_hrv_z) and _hrv_z > 0) else "inverse",
                       help="Desvios-padrão acima/abaixo da média 28d.")
            _c5.metric("Slope 7d", f"{_hrv_slp:+.1f} ms/d" if not np.isnan(_hrv_slp) else "—",
                       delta="→ melhorando" if (not np.isnan(_hrv_slp) and _hrv_slp > 0.3) else
                             ("→ estável" if not np.isnan(_hrv_slp) and abs(_hrv_slp) <= 0.3 else "→ a cair"),
                       delta_color="normal" if (not np.isnan(_hrv_slp) and _hrv_slp > 0.3) else
                                   "off" if not np.isnan(_hrv_slp) and abs(_hrv_slp) <= 0.3 else "inverse",
                       help="Slope da regressão linear dos últimos 7 dias de HRV.")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. MODO: PERÍODO MANUAL
    # ══════════════════════════════════════════════════════════════════════════
    if _mode == 0:
        st.markdown("### 📅 Análise por período manual")
        st.caption("Selecciona o período alvo e compara com o período anterior.")

        _hrv_dates = sig_hrv['Data'].dropna()
        _date_min  = _hrv_dates.min().date()
        _date_max  = _hrv_dates.max().date()

        _col_d1, _col_d2, _col_d3 = st.columns(3)
        with _col_d1:
            _p_start = st.date_input("Início do período", value=_date_max - pd.Timedelta(days=21),
                                     min_value=_date_min, max_value=_date_max, key="hrv_pstart")
        with _col_d2:
            _p_end = st.date_input("Fim do período", value=_date_max,
                                   min_value=_date_min, max_value=_date_max, key="hrv_pend")
        with _col_d3:
            _ref_days = st.number_input("Dias de referência (anterior)", value=21,
                                        min_value=7, max_value=90, step=7, key="hrv_refdays")

        if st.button("▶ Analisar período", type="primary", key="hrv_run_manual"):
            _ts = pd.Timestamp(_p_start)
            _te = pd.Timestamp(_p_end)

            if len(sig_train) == 0:
                st.warning("Sem dados de treino para comparar.")
            else:
                cmp = _compare_periods(sig_hrv, sig_train, _ts, _te, _ref_days)
                if cmp.empty:
                    st.warning("Sem dados suficientes no período.")
                else:
                    st.markdown(f"#### Comparação: {_ref_days}d antes vs [{_p_start} → {_p_end}]")

                    # Separar HRV de treino
                    _hrv_rows   = cmp[cmp['col'].isin(['hrv','ln_hrv','hrv_norm',
                                                        'hrv_rhr_ratio','rhr'])]
                    _train_rows = cmp[~cmp['col'].isin(['hrv','ln_hrv','hrv_norm',
                                                         'hrv_rhr_ratio','rhr'])]

                    # ── HRV: o que mudou ────────────────────────────────────
                    st.markdown("**❤️ HRV — o que mudou neste período**")
                    _hrv_display = _hrv_rows[['Variável','Antes','Período','Δ%',"Cohen's d",'sig']].copy()
                    _hrv_display['Δ%'] = _hrv_display['Δ%'].apply(
                        lambda x: f"{x:+.1f}%" if not pd.isna(x) else '—')
                    _hrv_display['sig'] = _hrv_display['sig'].map({True: '✅', False: ''})
                    st.dataframe(_hrv_display.rename(columns={'sig': 'Sig.'}),
                                 hide_index=True, use_container_width=True)

                    # ── Treino: o que mudou ─────────────────────────────────
                    st.markdown("**🏋️ Treino — o que antecedeu / acompanhou**")
                    _train_rows_s = _train_rows.sort_values('Δ%', key=lambda x:
                        pd.to_numeric(x, errors='coerce').abs(), ascending=False)
                    _tr_display = _train_rows_s[['Variável','Antes','Período','Δ%',
                                                  "Cohen's d",'sig']].copy()
                    _tr_display['Δ%'] = _tr_display['Δ%'].apply(
                        lambda x: f"{x:+.1f}%" if not pd.isna(x) else '—')
                    _tr_display['sig'] = _tr_display['sig'].map({True: '✅', False: ''})
                    st.dataframe(_tr_display.rename(columns={'sig': 'Sig.'}),
                                 hide_index=True, use_container_width=True)

                    # ── Narrativa automática ────────────────────────────────
                    st.markdown("**💡 Interpretação automática**")
                    _sig_changes = cmp[cmp['sig'] & (cmp['Δ%'].abs() > 5)].copy()
                    _sig_changes['Δ%_num'] = pd.to_numeric(_sig_changes['Δ%'], errors='coerce')

                    _hrv_delta = cmp[cmp['col']=='hrv']['Δ%'].values
                    _hrv_delta = float(_hrv_delta[0]) if len(_hrv_delta) > 0 else 0
                    _hrv_dir   = "subiu" if _hrv_delta > 0 else "desceu"
                    _hrv_mag   = "significativamente" if abs(_hrv_delta) > 10 else "ligeiramente"

                    _narrativa = [f"**HRV {_hrv_dir} {abs(_hrv_delta):.1f}%** "
                                  f"({_hrv_mag}) neste período."]

                    _top_pos = _sig_changes[_sig_changes['Δ%_num'] > 10] \
                        .nlargest(3, 'Δ%_num')
                    _top_neg = _sig_changes[_sig_changes['Δ%_num'] < -10] \
                        .nsmallest(3, 'Δ%_num')

                    if len(_top_pos) > 0:
                        _items = ', '.join(
                            f"**{r['Variável']}** ({r['Δ%']:+.1f}%)"
                            for _, r in _top_pos.iterrows()
                        )
                        _narrativa.append(f"Variáveis que subiram: {_items}.")
                    if len(_top_neg) > 0:
                        _items = ', '.join(
                            f"**{r['Variável']}** ({r['Δ%']:+.1f}%)"
                            for _, r in _top_neg.iterrows()
                        )
                        _narrativa.append(f"Variáveis que desceram: {_items}.")

                    for _n in _narrativa:
                        st.markdown(f"→ {_n}")

                    # ── Radar chart Before vs After ─────────────────────────
                    _radar_vars = ['load', 'dur_min', 'pct_z3', 'freq_7d',
                                   'mono_7d', 'tsb', 'atl']
                    _radar_rows = cmp[cmp['col'].isin(_radar_vars)].copy()
                    if len(_radar_rows) >= 4:
                        _before_n = (_radar_rows['Antes'] /
                                     _radar_rows['Antes'].replace(0, np.nan)).fillna(1)
                        _target_n = (_radar_rows['Período'] /
                                     _radar_rows['Antes'].replace(0, np.nan)).fillna(1)
                        _labels   = _radar_rows['Variável'].tolist()

                        _fig_r = go.Figure()
                        _fig_r.add_trace(go.Scatterpolar(
                            r=list(_before_n) + [_before_n.iloc[0]],
                            theta=_labels + [_labels[0]],
                            fill='toself', name='Antes',
                            line_color=_C['neutral'],
                            fillcolor='rgba(127,140,141,0.2)'
                        ))
                        _fig_r.add_trace(go.Scatterpolar(
                            r=list(_target_n) + [_target_n.iloc[0]],
                            theta=_labels + [_labels[0]],
                            fill='toself', name='Período',
                            line_color=_C['primary'],
                            fillcolor='rgba(41,128,185,0.2)'
                        ))
                        _fig_r.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 2],
                                                tickfont=dict(color='#111', size=9))
                            ),
                            paper_bgcolor='white', font=dict(color='#111', size=11),
                            height=380, margin=dict(t=30, b=30, l=40, r=40),
                            legend=dict(orientation='h', y=-0.08,
                        font=dict(color='#111', size=10))
                        )
                        st.plotly_chart(_fig_r, use_container_width=True,
                                        config=MC, key='hrv_radar')

                    # ── Download ────────────────────────────────────────────
                    st.download_button(
                        "📥 Download comparação período",
                        cmp[['Variável','Antes','Período','Δ%',"Cohen's d",'p-valor']].to_csv(
                            index=False, sep=';', decimal=','
                        ).encode('utf-8'),
                        f"hrv_comparacao_{_p_start}_{_p_end}.csv",
                        "text/csv", key="hrv_dl_manual"
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. MODO: DETECÇÃO AUTOMÁTICA
    # ══════════════════════════════════════════════════════════════════════════
    elif _mode == 1:
        st.markdown("### 🔍 Detecção automática de períodos HRV")
        st.caption("Detecta automaticamente períodos de HRV↑ e HRV↓ com base no z-score 28d.")

        _cz1, _cz2, _cz3 = st.columns(3)
        _z_thresh = _cz1.slider("Threshold z-score", 0.3, 1.5, 0.5, 0.1, key="hrv_zthresh")
        _min_len  = _cz2.slider("Duração mínima (dias)", 3, 14, 5, 1, key="hrv_minlen")
        _show_n   = _cz3.number_input("Mostrar últimos N períodos", 3, 20, 6, 1, key="hrv_shown")

        periods = _detect_hrv_periods(sig_hrv, _min_len, _z_thresh)

        if not periods:
            st.info("Sem períodos detectados com os critérios actuais.")
        else:
            _periods_df = pd.DataFrame(periods)
            _periods_df['duração'] = (_periods_df['end'] - _periods_df['start']).dt.days + 1
            _periods_df['start'] = _periods_df['start'].dt.strftime('%Y-%m-%d')
            _periods_df['end']   = _periods_df['end'].dt.strftime('%Y-%m-%d')
            _periods_df['mean_z'] = _periods_df['mean_z'].round(2)
            _periods_df['delta_hrv'] = _periods_df['delta_hrv'].round(1)

            st.markdown(f"**{len(periods)} períodos detectados** "
                        f"({sum(1 for p in periods if p['tipo']=='HRV↑')} ↑ | "
                        f"{sum(1 for p in periods if p['tipo']=='HRV↓')} ↓)")

            _pu = _periods_df[_periods_df['tipo']=='HRV↑'].tail(int(_show_n//2))
            _pd = _periods_df[_periods_df['tipo']=='HRV↓'].tail(int(_show_n//2))

            _ca, _cb = st.columns(2)
            with _ca:
                st.markdown("**HRV↑ — períodos de recuperação/adaptação**")
                if len(_pu) > 0:
                    st.dataframe(_pu[['start','end','duração','mean_z','delta_hrv']],
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("Sem períodos HRV↑")
            with _cb:
                st.markdown("**HRV↓ — períodos de supressão/fadiga**")
                if len(_pd) > 0:
                    st.dataframe(_pd[['start','end','duração','mean_z','delta_hrv']],
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("Sem períodos HRV↓")

            # Seleccionar um período para analisar
            st.markdown("---")
            st.markdown("**Analisar em detalhe um período detectado:**")

            # Construir lista local dos períodos a mostrar (mesmos que nas tabelas)
            _n_show   = int(_show_n)
            _show_list = periods[-_n_show:] if len(periods) >= _n_show else periods[:]
            _sel_opts  = [
                f"{p['tipo']} {p['start'].strftime('%d/%m/%Y')} → {p['end'].strftime('%d/%m/%Y')}"
                for p in _show_list
            ]
            _sel = st.selectbox("Seleccionar período", _sel_opts,
                                key="hrv_auto_sel")
            if _sel and len(sig_train) > 0:
                _idx  = _sel_opts.index(_sel)
                # Índice directo na lista local — sem aritmética negativa
                _per  = _show_list[_idx]
                _ts   = pd.Timestamp(_per['start'])
                _te   = pd.Timestamp(_per['end'])
                cmp   = _compare_periods(sig_hrv, sig_train, _ts, _te, 14)
                if not cmp.empty:
                    _sig_rows = cmp[cmp['sig']].sort_values(
                        'Δ%', key=lambda x: pd.to_numeric(x, errors='coerce').abs(),
                        ascending=False).head(10)
                    if len(_sig_rows) > 0:
                        st.markdown(f"**Top mudanças significativas — {_sel}:**")
                        _disp = _sig_rows[['Variável','Antes','Período','Δ%',"Cohen's d"]].copy()
                        _disp['Δ%'] = _disp['Δ%'].apply(
                            lambda x: f"{float(x):+.1f}%" if pd.notna(x) else '—')
                        st.dataframe(_disp, hide_index=True, use_container_width=True)
                    else:
                        st.info("Sem mudanças estatisticamente significativas neste período.")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. MODO: LAG CORRELATION
    # ══════════════════════════════════════════════════════════════════════════
    elif _mode == 2:
        st.markdown("### 🔗 Lag Correlation")
        st.caption(
            "Qual variável de treino antecede as mudanças de HRV e com quantos dias? "
            "Lag positivo = variável de treino precede HRV."
        )

        if len(sig_train) == 0:
            st.warning("Sem dados de treino.")
        else:
            _lc1, _lc2 = st.columns(2)
            _hrv_target = _lc1.selectbox(
                "Variável HRV alvo",
                [v for v in ['hrv','ln_hrv','hrv_norm','hrv_z28'] if v in sig_hrv.columns],
                key="hrv_lag_tgt"
            )
            _max_lag = _lc2.slider("Lag máximo (dias)", 3, 21, 14, 1, key="hrv_lag_max")

            if st.button("▶ Calcular lag correlations", type="primary", key="hrv_run_lag"):
                with st.spinner("A calcular correlações com lag..."):
                    lag_df = _lag_correlations(sig_hrv, sig_train,
                                               hrv_var=_hrv_target,
                                               max_lag=_max_lag)

                if lag_df.empty:
                    st.warning("Sem dados suficientes.")
                else:
                    # Melhor lag por variável — groupby().apply() com idxmax()
                    # perde colunas no Pandas 2.x → usar merge explícito
                    _sig_df = lag_df[lag_df['sig']].copy()
                    if len(_sig_df) > 0:
                        _best_idx = _sig_df.groupby('var')['r_abs'].idxmax()
                        best = _sig_df.loc[_best_idx].reset_index(drop=True) \
                                      .sort_values('r_abs', ascending=False)
                    else:
                        best = pd.DataFrame(columns=lag_df.columns)

                    if len(best) > 0:
                        st.markdown("**Top correlações significativas por variável:**")
                        _best_disp = best[['var','lag','r','p']].copy()
                        _best_disp['r']   = _best_disp['r'].round(3)
                        _best_disp['p']   = _best_disp['p'].apply(lambda x: f"{x:.3f}")
                        _best_disp['lag'] = _best_disp['lag'].apply(lambda x: f"{x}d")
                        _best_disp['direcção'] = best['r'].apply(
                            lambda x: '↑ HRV com ↑ variável' if x > 0
                                      else '↑ HRV com ↓ variável')
                        st.dataframe(_best_disp.rename(columns={'var': 'Variável treino',
                                                                  'lag': 'Lag óptimo',
                                                                  'r':   'r Pearson',
                                                                  'p':   'p-valor'}),
                                     hide_index=True, use_container_width=True)

                    # Heatmap lag × variável
                    _lag_pivot = lag_df.pivot(index='var', columns='lag', values='r')
                    if not _lag_pivot.empty:
                        _fig_heat = go.Figure(go.Heatmap(
                            z=_lag_pivot.values,
                            x=[f"lag {l}d" for l in _lag_pivot.columns],
                            y=_lag_pivot.index.tolist(),
                            colorscale='RdBu', zmid=0,
                            zmin=-1, zmax=1,
                            colorbar=dict(title='r', tickfont=dict(color='#111')),
                            hovertemplate='%{y} @ lag %{x}<br>r = <b>%{z:.2f}</b><extra></extra>'
                        ))
                        _fig_heat.update_layout(
                            paper_bgcolor='white', plot_bgcolor='white',
                            font=dict(color='#111', size=10),
                            height=max(300, len(_lag_pivot) * 28 + 80),
                            margin=dict(t=20, b=60, l=120, r=30),
                            xaxis_tickangle=-45,
                        )
                        st.plotly_chart(_fig_heat, use_container_width=True,
                                        config=MC, key='hrv_lag_heat')

                        st.caption(
                            "Azul escuro = correlação positiva forte (↑ variável → ↑ HRV). "
                            "Vermelho escuro = correlação negativa forte (↑ variável → ↓ HRV). "
                            "Lag Xd = variável X dias antes do HRV."
                        )

                    # Download
                    st.download_button(
                        "📥 Download lag correlations",
                        lag_df[lag_df['sig']].round(3).to_csv(
                            index=False, sep=';', decimal=','
                        ).encode('utf-8'),
                        "hrv_lag_correlations.csv", "text/csv",
                        key="hrv_dl_lag"
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. MODO: FINGERPRINT HRV
    # ══════════════════════════════════════════════════════════════════════════
    elif _mode == 3:
        st.markdown("### 🧬 Fingerprint — top vs bottom HRV days")
        st.caption(
            "O que aconteceu nos X dias antes dos melhores e piores dias de HRV? "
            "Identifica o padrão de treino que antecede a boa forma autonómica."
        )

        if len(sig_train) == 0:
            st.warning("Sem dados de treino.")
        else:
            _fp1, _fp2 = st.columns(2)
            _fp_pct  = _fp1.slider("Percentil top/bottom (%)", 5, 25, 10, 5,
                                    key="hrv_fp_pct")
            _fp_pre  = _fp2.slider("Dias antes a analisar", 3, 14, 7, 1,
                                    key="hrv_fp_pre")

            if st.button("▶ Calcular fingerprint", type="primary", key="hrv_run_fp"):
                with st.spinner("A calcular fingerprints..."):
                    fp = _hrv_fingerprint(sig_hrv, sig_train,
                                          pct=_fp_pct/100, pre_days=_fp_pre)

                if not fp:
                    st.warning("Sem dados suficientes.")
                else:
                    _var_labels = {
                        'load': 'Carga (TSS)',
                        'kj': 'kJ',
                        'dur_min': 'Duração (min)',
                        'n_sess': 'Nº sessões',
                        'pct_z3': '% Z3',
                        'freq_7d': 'Freq. semanal',
                        'mono_7d': 'Monotonia',
                        'strain_7d': 'Strain',
                        'tsb': 'TSB',
                        'atl': 'ATL',
                    }

                    _fp_rows = []
                    for var, vals in fp.items():
                        _fp_rows.append({
                            'Variável': _var_labels.get(var, var),
                            f'Top {_fp_pct}% HRV': round(vals['top'], 2) if not np.isnan(vals['top']) else '—',
                            f'Bottom {_fp_pct}% HRV': round(vals['bot'], 2) if not np.isnan(vals['bot']) else '—',
                            'Diferença %': f"{vals['diff_pct']:+.1f}%" if not np.isnan(vals['diff_pct']) else '—',
                            '_diff': vals['diff_pct'],
                        })

                    _fp_df = pd.DataFrame(_fp_rows)
                    _fp_df_s = _fp_df.dropna(subset=['_diff']).sort_values('_diff', ascending=False)

                    # Interpretação
                    _pos_patterns = _fp_df_s[_fp_df_s['_diff'] > 15]
                    _neg_patterns = _fp_df_s[_fp_df_s['_diff'] < -15]

                    st.markdown(f"#### Nos {_fp_pre} dias antes dos top {_fp_pct}% HRV:")
                    _fcp1, _fcp2 = st.columns(2)
                    with _fcp1:
                        st.markdown("🟢 **Mais alto nos dias de bom HRV:**")
                        for _, r in _pos_patterns.iterrows():
                            st.markdown(f"→ **{r['Variável']}**: {r['Diferença %']}")
                    with _fcp2:
                        st.markdown("🔴 **Mais baixo nos dias de bom HRV:**")
                        for _, r in _neg_patterns.iterrows():
                            st.markdown(f"→ **{r['Variável']}**: {r['Diferença %']}")

                    # Tabela
                    _fp_disp = _fp_df_s.drop(columns=['_diff'])
                    st.dataframe(_fp_disp, hide_index=True, use_container_width=True)

                    # Barras horizontais
                    _fig_bar = go.Figure()
                    _colors  = [_C['hrv_up'] if d >= 0 else _C['hrv_dn']
                                 for d in _fp_df_s['_diff']]
                    _fig_bar.add_trace(go.Bar(
                        y=_fp_df_s['Variável'],
                        x=_fp_df_s['_diff'],
                        orientation='h',
                        marker_color=_colors,
                        text=[f"{v:+.1f}%" for v in _fp_df_s['_diff']],
                        textposition='outside',
                        hovertemplate='%{y}<br>Diferença: <b>%{x:+.1f}%</b><extra></extra>'
                    ))
                    _fig_bar.add_vline(x=0, line_color='#aaa', line_width=1)
                    _fig_bar.update_layout(
                        paper_bgcolor='white', plot_bgcolor='white',
                        font=dict(color='#111', size=11),
                        height=max(280, len(_fp_df_s) * 32 + 60),
                        margin=dict(t=20, b=40, l=120, r=60),
                        xaxis_title=f"Diferença % (top vs bottom {_fp_pct}% HRV)",
                        yaxis_title=None,
                    )
                    st.plotly_chart(_fig_bar, use_container_width=True,
                                    config=MC, key='hrv_fp_bar')

                    st.caption(
                        f"Verde = variável mais alta nos {_fp_pre}d antes de dias de HRV alto. "
                        f"Vermelho = variável mais baixa antes de dias de HRV alto. "
                        f"Interpretação: os gatilhos positivos são as variáveis verdes."
                    )

                    st.download_button(
                        "📥 Download fingerprint HRV",
                        _fp_disp.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                        f"hrv_fingerprint_top{_fp_pct}.csv", "text/csv",
                        key="hrv_dl_fp"
                    )

    # ── Análises avançadas ────────────────────────────────────────────────────
    if len(sig_train) > 0:
        tab_hrv_advanced(sig_hrv, sig_train, da_full=_da)
    else:
        st.info("Conecta os dados de actividade para aceder às análises avançadas "
                "(ARI, Estados, Elasticidade, Lag Avançado, etc.).")

    # ── Nota metodológica ─────────────────────────────────────────────────────
    with st.expander("ℹ️ Metodologia — Recovery Pattern Analyzer"):
        st.markdown(f"""
**Métricas HRV utilizadas:**

| Métrica | Fórmula | Interpretação |
|---|---|---|
| rMSSD | directo | Actividade parassimpática absoluta |
| ln(rMSSD) | log(rMSSD) | Distribuição normal; padrão na literatura |
| AVNN | 60000 / RHR | Espaço temporal por batimento (ms) |
| rMSSD norm. | (rMSSD / AVNN) × 100 | Variabilidade relativa à FC de repouso |
| HRV/RHR ratio | rMSSD / RHR | Coupling autonómico |
| z-score 28d | (HRV - média28d) / std28d | Desvio relativo ao baseline |

**rMSSD normalizado — porquê importa:**
Um rMSSD de 60ms com RHR=60bpm (AVNN=1000ms) dá norm=6.
O mesmo rMSSD=60ms com RHR=40bpm (AVNN=1500ms) dá norm=4.
A variabilidade relativa piorou mesmo com rMSSD estável.

**Lag correlation:**
Correlação de Pearson entre variável de treino (dia t-lag) e HRV (dia t).
Lag óptimo = o lag com maior |r| significativo (p<0.05).

**Event windows:**
Para cada evento detectado, alinha os dados em torno do dia 0 e calcula
a média normalizada de cada variável na janela [-14d, +7d].

**Fingerprint HRV:**
Compara a média de cada variável de treino nos X dias antes dos top 10% HRV days
vs os X dias antes dos bottom 10% HRV days.
Diferença positiva = esta variável está associada a melhor HRV.

**Referências:** Kiviniemi et al. (2007), Hautala et al. (2010),
Plews et al. (2013), Buchheit (2014), Flatt & Esco (2016).
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULOS AVANÇADOS — adicionados após revisão de arquitectura
# ══════════════════════════════════════════════════════════════════════════════

# ── H. Estados Fisiológicos Heurísticos ──────────────────────────────────────

_STATES = {
    'autonomic_suppression': {
        'label': '🔴 Autonomic Suppression',
        'color': '#c0392b',
        'desc': 'HRV colapsado + RHR elevada + strain alto + slope negativo.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) < -1.0 and
            r.get('rhr_z', 0) > 0.8 and
            r.get('strain_7d', 0) > 0 and
            r.get('hrv_slope_7d', 0) < -0.3
        ),
    },
    'accumulated_fatigue': {
        'label': '🟠 Accumulated Fatigue',
        'color': '#e67e22',
        'desc': 'HRV abaixo baseline + ATL elevada + coupling deteriorando.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) < -0.5 and
            r.get('rhr_z', 0) > 0.3 and
            r.get('atl', 0) > r.get('ctl', 1) * 1.1
        ),
    },
    'functional_overreach': {
        'label': '🟡 Functional Overreach',
        'color': '#f39c12',
        'desc': 'HRV variável + monotonia alta + strain elevado. Precisa de unload.',
        'rules': lambda r: (
            r.get('mono_7d', 0) > 2.0 and
            r.get('strain_7d', 0) > 0 and
            abs(r.get('ln_hrv_z', 0)) < 1.0
        ),
    },
    'taper_response': {
        'label': '🟢 Taper Response',
        'color': '#27ae60',
        'desc': 'HRV a subir + RHR a cair + carga reduzida. Forma a emergir.',
        'rules': lambda r: (
            r.get('hrv_slope_7d', 0) > 0.3 and
            r.get('ln_hrv_z', 0) > 0.0 and
            r.get('load_7d', 1) < r.get('load_28d', 1) / 4 * 0.85
        ),
    },
    'parasympathetic_rebound': {
        'label': '💚 Parasympathetic Rebound',
        'color': '#1abc9c',
        'desc': 'HRV bem acima baseline + slope positivo + RHR baixa. Óptimo.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) > 1.0 and
            r.get('hrv_slope_7d', 0) > 0.2 and
            r.get('rhr_z', 0) < 0.0
        ),
    },
    'resilient_state': {
        'label': '🔵 Resilient State',
        'color': '#2980b9',
        'desc': 'HRV estável e acima baseline com carga normal. Adaptado.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) > 0.3 and
            abs(r.get('hrv_slope_7d', 0)) < 0.3 and
            r.get('rhr_z', 0) < 0.3
        ),
    },
    'maladaptation': {
        'label': '⚫ Maladaptation Risk',
        'color': '#2c3e50',
        'desc': 'HRV cronicamente baixo + RHR alta + strain persistente.',
        'rules': lambda r: (
            r.get('ln_hrv_z', 0) < -0.8 and
            r.get('rhr_z', 0) > 0.5 and
            r.get('mono_7d', 0) > 1.5
        ),
    },
    'baseline': {
        'label': '⚪ Baseline',
        'color': '#95a5a6',
        'desc': 'Estado neutro — sem padrão fisiológico dominante.',
        'rules': lambda r: True,   # fallback
    },
}


def _classify_states(sig_hrv: pd.DataFrame,
                     sig_train: pd.DataFrame) -> pd.DataFrame:
    """
    Classifica cada dia num dos 8 estados fisiológicos heurísticos.
    Retorna sig_hrv enriquecido com coluna 'state' e 'state_label'.
    """
    merged = pd.merge(
        sig_hrv,
        sig_train[['Data', 'load_7d', 'load_28d', 'atl', 'ctl',
                    'mono_7d', 'strain_7d', 'n_sess']] if len(sig_train) > 0
        else pd.DataFrame(columns=['Data']),
        on='Data', how='left'
    ).sort_values('Data').reset_index(drop=True)

    # Pré-calcular z-scores necessários
    if 'ln_hrv' in merged.columns:
        merged['ln_hrv_z'] = (
            (merged['ln_hrv'] - merged['ln_hrv'].rolling(28, min_periods=7).mean()) /
            merged['ln_hrv'].rolling(28, min_periods=7).std().replace(0, np.nan)
        )
    else:
        merged['ln_hrv_z'] = 0.0

    merged['rhr_z'] = merged.get('rhr_z28', pd.Series(np.zeros(len(merged))))

    states = []
    for _, row in merged.iterrows():
        r = row.to_dict()
        assigned = 'baseline'
        # Ordem de prioridade: estados mais graves primeiro
        for s_key in ['autonomic_suppression', 'maladaptation',
                       'accumulated_fatigue', 'functional_overreach',
                       'parasympathetic_rebound', 'taper_response',
                       'resilient_state', 'baseline']:
            try:
                if _STATES[s_key]['rules'](r):
                    assigned = s_key
                    break
            except Exception:
                pass
        states.append(assigned)

    merged['state']       = states
    merged['state_label'] = merged['state'].map(
        {k: v['label'] for k, v in _STATES.items()})
    merged['state_color'] = merged['state'].map(
        {k: v['color'] for k, v in _STATES.items()})
    return merged


# ── I. ARI — Autonomic Readiness Index ───────────────────────────────────────

_ARI_WEIGHTS = {
    'ln_hrv_z':       +0.35,   # HRV logarítmico normalizado
    'rhr_z':          -0.30,   # RHR (negativo: RHR alta = ARI baixo)
    'hrv_norm_z':     +0.20,   # rMSSD norm (variabilidade relativa)
    'instability_z':  -0.10,   # instabilidade HRV 7d (negativo)
    'slope_z':        +0.05,   # slope positivo = melhorando
}

def _compute_ari(sig_hrv: pd.DataFrame) -> pd.DataFrame:
    """
    Autonomic Readiness Index (ARI):
      ARI = 0.35×z(ln_rMSSD) - 0.30×z(RHR) + 0.20×z(rMSSD_norm)
            - 0.10×z(instability_7d) + 0.05×z(slope_7d)

    Escalado para 0-100 (média histórica = 50).
    Confidence = nº de sinais disponíveis e alinhados (0-5).
    """
    df = sig_hrv.copy()

    def _z28(col):
        s = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        mu = s.rolling(28, min_periods=7).mean()
        sd = s.rolling(28, min_periods=7).std().replace(0, np.nan)
        return (s - mu) / sd

    # z-scores de cada componente
    df['_z_ln_hrv']    = _z28('ln_hrv')
    df['_z_rhr']       = _z28('rhr')
    df['_z_hrv_norm']  = _z28('hrv_norm') if 'hrv_norm' in df.columns \
                          else pd.Series(0.0, index=df.index)
    # Instabilidade = std rolling 7d do HRV (alta instabilidade = mau sinal)
    df['_instab']      = df['hrv'].rolling(7, min_periods=3).std() \
                          if 'hrv' in df.columns else pd.Series(np.nan, index=df.index)
    df['_z_instab']    = _z28('_instab')
    df['_z_slope']     = _z28('hrv_slope_7d') if 'hrv_slope_7d' in df.columns \
                          else pd.Series(0.0, index=df.index)

    # Score composto (soma ponderada)
    components = [
        ('_z_ln_hrv',   +0.35),
        ('_z_rhr',      -0.30),
        ('_z_hrv_norm', +0.20),
        ('_z_instab',   -0.10),
        ('_z_slope',    +0.05),
    ]

    ari_raw    = pd.Series(0.0, index=df.index)
    n_avail    = pd.Series(0,   index=df.index)
    n_aligned  = pd.Series(0,   index=df.index)  # sinais apontando na direcção correcta

    for col, w in components:
        valid = df[col].notna()
        ari_raw = ari_raw.where(~valid, ari_raw + df[col].fillna(0) * w)
        n_avail = n_avail + valid.astype(int)
        # "Alinhado" = sinal positivo com peso positivo OU negativo com peso negativo
        n_aligned = n_aligned + (
            ((df[col].fillna(0) > 0) & (w > 0)) |
            ((df[col].fillna(0) < 0) & (w < 0))
        ).astype(int)

    # Escalar para 0-100: média histórica → 50, ±2 std → ±30
    mu_ari  = ari_raw.rolling(90, min_periods=14).mean()
    sd_ari  = ari_raw.rolling(90, min_periods=14).std().replace(0, np.nan)
    df['ARI'] = (50 + 15 * (ari_raw - mu_ari) / sd_ari.fillna(1)).clip(0, 100)

    # Confidence: baseado no nº de sinais disponíveis E alinhados
    df['ARI_n_signals']  = n_avail
    df['ARI_n_aligned']  = n_aligned
    df['ARI_confidence'] = pd.cut(
        n_aligned,
        bins=[-1, 1, 2, 3, 4, 10],
        labels=['Muito baixa', 'Baixa', 'Moderada', 'Alta', 'Muito alta']
    )
    return df


# ── J. Recovery Elasticity ────────────────────────────────────────────────────

def _recovery_elasticity(sig_hrv: pd.DataFrame,
                          sig_train: pd.DataFrame,
                          z_suppress: float = -1.0,
                          z_recover: float = -0.3,
                          max_days: int = 21) -> dict:
    """
    Para cada evento de supressão de HRV (z28 < z_suppress),
    mede quantos dias demora até z28 > z_recover.

    Retorna:
      {
        events: list of {date, days_to_recovery, recovered, suppression_depth},
        tau_median: float,
        tau_mean: float,
        by_modality: {mod: tau_median} (modalidade dominante no evento),
        n_events: int,
        n_recovered: int,
      }
    """
    if 'hrv_z28' not in sig_hrv.columns:
        return {'n_events': 0, 'error': 'Sem z-score 28d'}

    df = sig_hrv.sort_values('Data').reset_index(drop=True)
    z  = df['hrv_z28'].values
    dt = pd.to_datetime(df['Data']).values

    events      = []
    i           = 0
    in_suppress = False
    event_start = None

    while i < len(z):
        v = z[i] if not np.isnan(z[i]) else 0
        if not in_suppress and v < z_suppress:
            in_suppress = True
            event_start = i
        elif in_suppress and v >= z_recover:
            # Evento completo
            days_to_rec = i - event_start
            depth       = float(np.nanmin(z[event_start:i]))
            events.append({
                'date':               pd.Timestamp(dt[event_start]).date(),
                'days_to_recovery':   days_to_rec,
                'suppression_depth':  round(depth, 2),
                'recovered':          True,
            })
            in_suppress = False
        elif in_suppress and (i - event_start) > max_days:
            # Não recuperou dentro da janela
            depth = float(np.nanmin(z[event_start:i]))
            events.append({
                'date':               pd.Timestamp(dt[event_start]).date(),
                'days_to_recovery':   max_days,
                'suppression_depth':  round(depth, 2),
                'recovered':          False,
            })
            in_suppress = False
        i += 1

    if not events:
        return {'n_events': 0, 'tau_median': np.nan, 'tau_mean': np.nan,
                'events': [], 'n_recovered': 0}

    recovered_days = [e['days_to_recovery'] for e in events if e['recovered']]
    tau_median     = float(np.median(recovered_days)) if recovered_days else np.nan
    tau_mean       = float(np.mean(recovered_days))   if recovered_days else np.nan

    # Por modalidade dominante no evento (modalidade com mais carga nos 7d antes)
    by_mod = {}
    if len(sig_train) > 0:
        for e in events:
            edate = pd.Timestamp(e['date'])
            pre   = sig_train[
                (sig_train['Data'] >= edate - pd.Timedelta(days=7)) &
                (sig_train['Data'] < edate)
            ]
            # Proxy: usar atl por modalidade se disponível, senão skip
            by_mod.setdefault('Todos', []).append(e['days_to_recovery'])

    by_mod_summary = {m: round(float(np.median(v)), 1)
                       for m, v in by_mod.items() if v}

    return {
        'n_events':   len(events),
        'n_recovered': len(recovered_days),
        'tau_median': round(tau_median, 1) if not np.isnan(tau_median) else None,
        'tau_mean':   round(tau_mean, 1)   if not np.isnan(tau_mean)   else None,
        'events':     events,
        'by_modality': by_mod_summary,
    }


# ── K. Lag Correlation Avançada (Pearson + Spearman + MI) ─────────────────────

def _normalized_mi(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Mutual Information normalizada: MI / sqrt(H(x)×H(y))
    Valores em [0, 1]. Detecta relações não-lineares.
    Usa permutation baseline para corrigir viés de N pequeno.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y  = x[valid], y[valid]
    if len(x) < 20:
        return np.nan

    # Discretizar
    def _entropy(arr, bins):
        h, _ = np.histogram(arr, bins=bins)
        p    = h / h.sum()
        p    = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _joint_entropy(a, b, bins):
        h, _, _ = np.histogram2d(a, b, bins=bins)
        p       = h / h.sum()
        p       = p[p > 0]
        return -np.sum(p * np.log2(p))

    hx  = _entropy(x, n_bins)
    hy  = _entropy(y, n_bins)
    hxy = _joint_entropy(x, y, n_bins)
    mi  = hx + hy - hxy

    # Permutation baseline: MI esperado por acaso
    mi_perm = []
    rng = np.random.default_rng(42)
    for _ in range(20):
        yp  = rng.permutation(y)
        hxyp = _joint_entropy(x, yp, n_bins)
        mi_perm.append(hx + hy - hxyp)
    mi_baseline = float(np.mean(mi_perm))
    mi_corrected = max(0.0, mi - mi_baseline)

    # Normalizar
    denom = np.sqrt(hx * hy)
    return float(mi_corrected / denom) if denom > 0 else 0.0


def _lag_correlations_advanced(sig_hrv: pd.DataFrame,
                                sig_train: pd.DataFrame,
                                hrv_var: str = 'hrv',
                                train_vars: list = None,
                                max_lag: int = 14) -> pd.DataFrame:
    """
    Lag correlation com 3 métodos:
      Pearson  — magnitude linear
      Spearman — robusto a outliers e monotónico
      MI_norm  — detecta relações não-lineares (HIIT dose-response em U)
    """
    if train_vars is None:
        train_vars = ['load', 'kj', 'dur_min', 'pct_z3',
                       'freq_7d', 'mono_7d', 'strain_7d', 'tsb', 'atl', 'n_sess']

    merged = pd.merge(
        sig_hrv[['Data', hrv_var]].rename(columns={hrv_var: 'hrv_tgt'}),
        sig_train[['Data'] + [v for v in train_vars if v in sig_train.columns]],
        on='Data', how='inner'
    ).sort_values('Data')

    hrv_s = merged['hrv_tgt'].values
    rows  = []

    for var in train_vars:
        if var not in merged.columns:
            continue
        x = merged[var].values

        for lag in range(0, max_lag + 1):
            xv = x[:-lag] if lag > 0 else x
            yv = hrv_s[lag:] if lag > 0 else hrv_s
            valid = ~(np.isnan(xv) | np.isnan(yv))

            if valid.sum() < 20:
                continue

            xvv, yvv = xv[valid], yv[valid]

            # Pearson
            try:
                r_p, p_p = scipy_stats.pearsonr(xvv, yvv)
            except Exception:
                r_p, p_p = np.nan, np.nan

            # Spearman
            try:
                r_s, p_s = scipy_stats.spearmanr(xvv, yvv)
            except Exception:
                r_s, p_s = np.nan, np.nan

            # MI normalizada
            mi = _normalized_mi(xvv, yvv)

            rows.append({
                'var':          var,
                'lag':          lag,
                'r_pearson':    round(r_p, 3) if not np.isnan(r_p) else np.nan,
                'p_pearson':    round(p_p, 3) if not np.isnan(p_p) else np.nan,
                'r_spearman':   round(r_s, 3) if not np.isnan(r_s) else np.nan,
                'p_spearman':   round(p_s, 3) if not np.isnan(p_s) else np.nan,
                'mi_norm':      round(mi, 3)  if not np.isnan(mi)  else np.nan,
                'r_abs':        abs(r_p) if not np.isnan(r_p) else 0.0,
                'sig_pearson':  p_p < 0.05 if not np.isnan(p_p) else False,
                'sig_spearman': p_s < 0.05 if not np.isnan(p_s) else False,
                'sig_any':      (p_p < 0.05 or p_s < 0.05) if not (np.isnan(p_p) and np.isnan(p_s)) else False,
            })

    return pd.DataFrame(rows)


# ── L. Directional Analysis ───────────────────────────────────────────────────

def _directional_analysis(sig_hrv: pd.DataFrame,
                            sig_train: pd.DataFrame,
                            patterns: list[dict],
                            outcome_lag: int = 5,
                            hrv_improve_z: float = 0.3) -> list[dict]:
    """
    Para cada padrão em patterns (lista de condições sobre variáveis de treino),
    conta quantas vezes ocorreu e quantas vezes foi seguido por HRV melhorado.

    pattern = {
        'name': 'Monotonia↓ + Z2↑',
        'conditions': [
            {'var': 'mono_7d_delta', 'op': '<', 'val': -0.15},
            {'var': 'pct_z3', 'op': '<', 'val': 30},
        ]
    }
    """
    merged = pd.merge(
        sig_hrv[['Data', 'hrv_z28', 'hrv_slope_7d']],
        sig_train,
        on='Data', how='inner'
    ).sort_values('Data').reset_index(drop=True)

    # Calcular deltas rolling
    for var in ['mono_7d', 'strain_7d', 'load_7d', 'pct_z3', 'freq_7d']:
        if var in merged.columns:
            merged[f'{var}_delta'] = merged[var].pct_change(periods=7).fillna(0)

    results = []
    for pat in patterns:
        n_occur   = 0
        n_improve = 0
        dates_ok  = []

        for i in range(len(merged) - outcome_lag):
            row = merged.iloc[i]
            # Avaliar condições
            cond_met = True
            for c in pat.get('conditions', []):
                val = row.get(c['var'], np.nan)
                if np.isnan(val):
                    cond_met = False
                    break
                if c['op'] == '<'  and not (val < c['val']):   cond_met = False; break
                if c['op'] == '>'  and not (val > c['val']):   cond_met = False; break
                if c['op'] == '<=' and not (val <= c['val']):  cond_met = False; break
                if c['op'] == '>=' and not (val >= c['val']):  cond_met = False; break

            if cond_met:
                n_occur += 1
                dates_ok.append(merged.iloc[i]['Data'])
                # Verificar outcome: HRV sobe nos próximos outcome_lag dias?
                future_z = merged['hrv_z28'].iloc[i+1:i+1+outcome_lag]
                if future_z.max() > hrv_improve_z:
                    n_improve += 1

        consistency = n_improve / n_occur if n_occur > 0 else 0.0
        confidence  = ('Alto (N≥20)'       if n_occur >= 20 else
                       'Moderado (N=10-19)' if n_occur >= 10 else
                       'Baixo (N<10)')

        results.append({
            'pattern':     pat['name'],
            'n_occur':     n_occur,
            'n_improve':   n_improve,
            'consistency': round(consistency * 100, 1),
            'confidence':  confidence,
            'dates':       dates_ok,
        })

    return results


# ── M. Dose-Response Curves (LOWESS) ─────────────────────────────────────────

def _dose_response(sig_hrv: pd.DataFrame,
                    sig_train: pd.DataFrame,
                    x_var: str,
                    y_var: str = 'hrv',
                    lag: int = 3,
                    frac: float = 0.4) -> pd.DataFrame:
    """
    Relação entre variável de treino (x_var, dia t) e HRV (y_var, dia t+lag).
    Usa LOWESS smoothing para capturar relações não-lineares (U-shape).
    """
    from scipy.stats import pearsonr

    merged = pd.merge(
        sig_hrv[['Data', y_var]].rename(columns={y_var: 'hrv_out'}),
        sig_train[['Data', x_var]] if x_var in sig_train.columns
        else pd.DataFrame(columns=['Data', x_var]),
        on='Data', how='inner'
    ).sort_values('Data').reset_index(drop=True)

    if len(merged) < 20 or x_var not in merged.columns:
        return pd.DataFrame()

    x = merged[x_var].values
    if lag > 0:
        # Alinhar: x[i] → hrv[i+lag]
        xv = x[:-lag]
        yv = merged['hrv_out'].values[lag:]
    else:
        xv, yv = x, merged['hrv_out'].values

    valid = ~(np.isnan(xv) | np.isnan(yv))
    xv, yv = xv[valid], yv[valid]

    if len(xv) < 10:
        return pd.DataFrame()

    # Ordenar por x para LOWESS
    order = np.argsort(xv)
    xo, yo = xv[order], yv[order]

    # LOWESS manual (scipy não tem, usar statsmodels se disponível, senão rolling)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(yo, xo, frac=frac, return_sorted=True)
        xs, ys = smooth[:, 0], smooth[:, 1]
    except ImportError:
        # Fallback: rolling mean com janela proporcional
        w = max(3, int(len(xo) * frac))
        ys = pd.Series(yo).rolling(w, center=True, min_periods=3).mean().values
        xs = xo

    return pd.DataFrame({'x': xs, 'y_smooth': ys,
                         'x_raw': xo, 'y_raw': yo})


# ── N. K-means de semanas ─────────────────────────────────────────────────────

def _cluster_weeks(sig_hrv: pd.DataFrame,
                    sig_train: pd.DataFrame,
                    n_clusters: int = 4) -> pd.DataFrame:
    """
    Clusturiza semanas por variáveis de TREINO (sem HRV no clustering).
    Depois colore os clusters pelo outcome HRV médio da semana seguinte.

    Features: load_total, mono_mean, freq, pct_z3, strain_mean
    Target (coloring): hrv_next_week_mean
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Agregar por semana
    merged = pd.merge(sig_hrv[['Data','hrv']], sig_train, on='Data', how='inner')
    merged['Data'] = pd.to_datetime(merged['Data'])
    merged['week'] = merged['Data'].dt.to_period('W')

    wk = merged.groupby('week').agg(
        load_total = ('load',     'sum'),
        mono_mean  = ('mono_7d',  'mean'),
        freq       = ('n_sess',   'sum'),
        pct_z3     = ('pct_z3',   'mean'),
        strain_mean= ('strain_7d','mean'),
        hrv_mean   = ('hrv',      'mean'),
        n_days     = ('hrv',      'count'),
    ).reset_index()

    wk = wk[wk['n_days'] >= 4].dropna(subset=['load_total','mono_mean'])
    if len(wk) < n_clusters * 3:
        return pd.DataFrame()

    # HRV da semana SEGUINTE como outcome
    wk = wk.sort_values('week').reset_index(drop=True)
    wk['hrv_next'] = wk['hrv_mean'].shift(-1)

    features = ['load_total', 'mono_mean', 'freq', 'pct_z3', 'strain_mean']
    X = wk[features].fillna(wk[features].median())

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    wk['cluster'] = km.fit_predict(X_scaled)

    # Label dos clusters por HRV outcome
    cluster_hrv = wk.groupby('cluster')['hrv_next'].mean().sort_values(ascending=False)
    rank_map     = {c: i+1 for i, c in enumerate(cluster_hrv.index)}
    wk['cluster_rank'] = wk['cluster'].map(rank_map)

    labels = {1: '🟢 Semana Óptima', 2: '🟡 Semana Boa',
              3: '🟠 Semana de Atenção', 4: '🔴 Semana Difícil'}
    wk['cluster_label'] = wk['cluster_rank'].map(labels)

    return wk


# ── O. Transition Matrix ──────────────────────────────────────────────────────

def _transition_matrix(state_series: pd.Series) -> pd.DataFrame:
    """
    Probabilistic transition matrix entre estados fisiológicos.
    P(estado_t+1 | estado_t)
    """
    states = state_series.dropna().values
    unique = sorted(set(states))

    mat = pd.DataFrame(0, index=unique, columns=unique, dtype=float)
    for i in range(len(states) - 1):
        mat.loc[states[i], states[i+1]] += 1

    # Normalizar por linha
    row_sums = mat.sum(axis=1).replace(0, np.nan)
    mat = mat.div(row_sums, axis=0).fillna(0)
    return mat.round(3)


# ══════════════════════════════════════════════════════════════════════════════
# TAB AVANÇADA — adicionar à função tab_hrv_analyzer existente
# ══════════════════════════════════════════════════════════════════════════════

def tab_hrv_advanced(sig_hrv: pd.DataFrame,
                      sig_train: pd.DataFrame,
                      da_full: pd.DataFrame = None):
    """
    Secção avançada da tab HRV — chamada dentro de tab_hrv_analyzer.
    Contém: ARI, Estados, Elasticidade, Lag Avançado, Directional,
            Dose-Response, K-means, Transition Matrix.
    """
    st.markdown("---")
    st.subheader("🧠 Análises Avançadas")

    _adv_tabs = st.tabs([
        "🎯 ARI",
        "🏷️ Estados",
        "⚡ Elasticidade",
        "🔗 Lag Avançado",
        "➡️ Directional",
        "📈 Dose-Response",
        "🗂️ Semanas",
        "🔄 Transições",
    ])

    # ── ARI ──────────────────────────────────────────────────────────────────
    with _adv_tabs[0]:
        st.markdown("#### 🎯 Autonomic Readiness Index (ARI)")
        st.caption(
            "Score composto 0-100 que integra 5 sinais autonómicos. "
            "Média histórica = 50. ARI>60 = boa readiness. ARI<40 = atenção."
        )
        _ari_df = _compute_ari(sig_hrv)

        # Cards actuais
        _ari_now   = _ari_df['ARI'].dropna().iloc[-1] if _ari_df['ARI'].notna().any() else np.nan
        _ari_conf  = _ari_df['ARI_confidence'].dropna().iloc[-1] if len(_ari_df) > 0 else '—'
        _ari_nalign= int(_ari_df['ARI_n_aligned'].dropna().iloc[-1]) if _ari_df['ARI_n_aligned'].notna().any() else 0
        _ari_navail= int(_ari_df['ARI_n_signals'].dropna().iloc[-1]) if _ari_df['ARI_n_signals'].notna().any() else 0

        _ac1, _ac2, _ac3 = st.columns(3)
        _ari_color = ('🟢' if not np.isnan(_ari_now) and _ari_now > 60 else
                      '🟡' if not np.isnan(_ari_now) and _ari_now > 40 else '🔴')
        _ac1.metric("ARI hoje",
                    f"{_ari_color} {_ari_now:.0f}/100" if not np.isnan(_ari_now) else "—",
                    help="0-100. Média histórica=50. >60=boa readiness. <40=atenção.")
        _ac2.metric("Confidence",
                    str(_ari_conf),
                    delta=f"Sinais alinhados: {_ari_nalign}/{_ari_navail}",
                    delta_color="normal" if _ari_nalign >= 3 else "off",
                    help="Quantos dos 5 sinais estão alinhados na mesma direcção.")
        _ac3.metric("Pesos ARI",
                    "ln(HRV)×0.35 | RHR×0.30",
                    help="ARI = 0.35z(ln_rMSSD) - 0.30z(RHR) + 0.20z(norm) - 0.10z(instab) + 0.05z(slope)")

        # Série ARI
        _fig_ari = go.Figure()
        _fig_ari.add_hrect(y0=60, y1=100, fillcolor='rgba(39,174,96,0.08)',
                            line_width=0, name='Zona óptima')
        _fig_ari.add_hrect(y0=0, y1=40, fillcolor='rgba(231,76,60,0.08)',
                            line_width=0, name='Zona de atenção')
        _fig_ari.add_hline(y=50, line_dash='dot', line_color='#aaa', line_width=1)
        _fig_ari.add_trace(go.Scatter(
            x=_ari_df['Data'], y=_ari_df['ARI'],
            mode='lines', name='ARI',
            line=dict(color=_C['primary'], width=2.5),
            fill='tozeroy', fillcolor='rgba(41,128,185,0.08)',
            hovertemplate='%{x|%d/%m/%Y}<br>ARI: <b>%{y:.0f}</b><extra></extra>'
        ))
        _fig_ari.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=11), height=320,
            margin=dict(t=20, b=50, l=50, r=20),
            yaxis=dict(range=[0, 100], title='ARI', tickfont=dict(color='#111'),
                       showgrid=True, gridcolor='#eee'),
            xaxis=dict(tickfont=dict(color='#111'), showgrid=True, gridcolor='#eee'),
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.18,
                        font=dict(color='#111', size=10)),
        )
        st.plotly_chart(_fig_ari, use_container_width=True, config=MC, key='ari_series')

        # Download ARI
        _ari_dl = _ari_df[['Data','hrv','ln_hrv','rhr','ARI',
                              'ARI_n_signals','ARI_n_aligned','ARI_confidence']].copy()
        _ari_dl['Data'] = _ari_dl['Data'].astype(str)
        st.download_button("📥 Download ARI diário",
                           _ari_dl.round(3).to_csv(index=False,sep=';',decimal=',').encode(),
                           "atheltica_ari.csv","text/csv", key="ari_dl")

        with st.expander("ℹ️ Fórmula ARI"):
            st.markdown("""
| Componente | Peso | Interpretação |
|---|---|---|
| z(ln_rMSSD) | +0.35 | HRV logarítmico — sinal principal |
| z(RHR) | **-0.30** | RHR alta = ARI baixo |
| z(rMSSD_norm) | +0.20 | Variabilidade relativa à FC |
| z(instabilidade_7d) | **-0.10** | Instabilidade HRV = stress |
| z(slope_7d) | +0.05 | Tendência positiva = melhorando |

Escalado: média 90d = 50, ±2σ ≈ ±30 pontos.
Confidence = nº de sinais alinhados na mesma direcção (0-5).
            """)

    # ── ESTADOS ──────────────────────────────────────────────────────────────
    with _adv_tabs[1]:
        st.markdown("#### 🏷️ Estados Fisiológicos Heurísticos")
        st.caption(
            "7 estados detectados por regras fisiológicas. "
            "Interpretáveis e accionáveis sem modelo probabilístico."
        )

        _state_df = _classify_states(sig_hrv, sig_train)

        # Timeline de estados
        _fig_st = go.Figure()
        for state_key, state_info in _STATES.items():
            _mask = _state_df['state'] == state_key
            if _mask.sum() == 0:
                continue
            _sub = _state_df[_mask]
            _fig_st.add_trace(go.Scatter(
                x=_sub['Data'], y=[state_info['label']] * len(_sub),
                mode='markers',
                marker=dict(size=8, color=state_info['color'],
                            line=dict(width=1, color='white')),
                name=state_info['label'],
                hovertemplate='%{x|%d/%m/%Y}<br>%{y}<extra></extra>'
            ))

        _fig_st.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='#111', size=10), height=320,
            margin=dict(t=20, b=50, l=200, r=20),
            xaxis=dict(tickfont=dict(color='#111'), showgrid=True, gridcolor='#eee'),
            yaxis=dict(tickfont=dict(color='#111')),
            legend=dict(orientation='h', y=-0.22,
                        font=dict(color='#111', size=9)),
            hovermode='closest',
        )
        st.plotly_chart(_fig_st, use_container_width=True, config=MC, key='state_timeline')

        # Distribuição de estados
        _st_counts = _state_df['state'].value_counts().reset_index()
        _st_counts.columns = ['state','n']
        _st_counts['label'] = _st_counts['state'].map(
            {k: v['label'] for k, v in _STATES.items()})
        _st_counts['color'] = _st_counts['state'].map(
            {k: v['color'] for k, v in _STATES.items()})
        _st_counts['pct']   = (_st_counts['n'] / len(_state_df) * 100).round(1)

        _fc1, _fc2 = st.columns([2, 3])
        with _fc1:
            st.markdown("**Distribuição**")
            st.dataframe(
                _st_counts[['label','n','pct']].rename(
                    columns={'label':'Estado','n':'Dias','pct':'%'}),
                hide_index=True, use_container_width=True)

        with _fc2:
            st.markdown("**Definições**")
            for k, v in _STATES.items():
                if k == 'baseline': continue
                st.markdown(f"**{v['label']}** — {v['desc']}")

        # Estado actual
        _today_state = _state_df['state_label'].dropna().iloc[-1] if len(_state_df) > 0 else '—'
        _today_desc  = _STATES.get(_state_df['state'].iloc[-1], {}).get('desc', '')
        st.info(f"**Estado actual:** {_today_state}\n\n{_today_desc}")

        # Download estados
        _st_dl = _state_df[['Data','state_label','state',
                              'hrv','ln_hrv','hrv_norm','rhr',
                              'hrv_z28','hrv_slope_7d']].copy()
        _st_dl['Data'] = _st_dl['Data'].astype(str)
        st.download_button(
            "📥 Download estados fisiológicos diários",
            _st_dl.round(3).to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
            "atheltica_hrv_estados.csv", "text/csv", key="adv_estados_dl"
        )

    # ── ELASTICIDADE ─────────────────────────────────────────────────────────
    with _adv_tabs[2]:
        st.markdown("#### ⚡ Recovery Elasticity")
        st.caption(
            "τ_recovery = dias até HRV voltar ao baseline após supressão. "
            "Assinatura individual: τ baixo = recuperas rápido."
        )

        _ez1, _ez2 = st.columns(2)
        _z_supp = _ez1.slider("z supressão (trigger)", -2.0, -0.5, -1.0, 0.1,
                               key="elast_z_supp")
        _z_rec  = _ez2.slider("z recuperação (target)", -0.5, 0.5, -0.3, 0.1,
                               key="elast_z_rec")

        elast = _recovery_elasticity(sig_hrv, sig_train,
                                      z_suppress=_z_supp, z_recover=_z_rec)

        if elast['n_events'] == 0:
            st.info("Sem eventos de supressão detectados com estes critérios.")
        else:
            _ec1, _ec2, _ec3, _ec4 = st.columns(4)
            _ec1.metric("τ mediana",
                        f"{elast['tau_median']}d" if elast['tau_median'] else "—",
                        help="Mediana de dias para recuperar após supressão.")
            _ec2.metric("τ média",
                        f"{elast['tau_mean']}d" if elast['tau_mean'] else "—")
            _ec3.metric("Eventos",
                        f"{elast['n_events']}",
                        delta=f"{elast['n_recovered']} recuperados",
                        delta_color="normal")
            _ec4.metric("Taxa recuperação",
                        f"{elast['n_recovered']/elast['n_events']*100:.0f}%"
                        if elast['n_events'] > 0 else "—")

            # Tabela de eventos
            if elast['events']:
                _ev_df = pd.DataFrame(elast['events'])
                _ev_df['date'] = _ev_df['date'].astype(str)
                _ev_df['recovered'] = _ev_df['recovered'].map({True: '✅', False: '❌'})
                st.dataframe(
                    _ev_df.rename(columns={
                        'date': 'Data',
                        'days_to_recovery': 'Dias até recuperar',
                        'suppression_depth': 'Profundidade z',
                        'recovered': 'Recuperou',
                    }),
                    hide_index=True, use_container_width=True)

            # Histogram de τ
            _rec_days = [e['days_to_recovery'] for e in elast['events'] if e['recovered']]
            if _rec_days:
                _fig_hist = go.Figure(go.Histogram(
                    x=_rec_days, nbinsx=min(10, len(_rec_days)),
                    marker_color=_C['primary'],
                    marker_line_color='white', marker_line_width=1,
                    hovertemplate='%{x}d: <b>%{y}</b> eventos<extra></extra>'
                ))
                if elast['tau_median']:
                    _fig_hist.add_vline(x=elast['tau_median'],
                                        line_dash='dash', line_color='#e74c3c',
                                        line_width=2,
                                        annotation_text=f"τ={elast['tau_median']}d",
                                        annotation_font=dict(color='#e74c3c'))
                _fig_hist.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='#111', size=11), height=260,
                    margin=dict(t=20, b=50, l=50, r=20),
                    xaxis=dict(title='Dias até recuperar',
                               tickfont=dict(color='#111'), showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='Nº eventos', tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#eee'),
                )
                st.plotly_chart(_fig_hist, use_container_width=True,
                                config=MC, key='elast_hist')
                st.caption(
                    f"τ mediana = {elast['tau_median']}d. "
                    "Este é o teu tempo típico de recuperação HRV. "
                    "Usa-o para planear o intervalo entre blocos de carga intensos."
                )
                # Download elasticidade
                _el_dl = pd.DataFrame(elast['events'])
                if not _el_dl.empty:
                    _el_dl['date'] = _el_dl['date'].astype(str)
                    _el_meta = pd.DataFrame([{
                        'metrica': 'tau_mediana_dias', 'valor': elast['tau_median']},
                        {'metrica': 'tau_media_dias',   'valor': elast['tau_mean']},
                        {'metrica': 'n_eventos',         'valor': elast['n_events']},
                        {'metrica': 'n_recuperados',     'valor': elast['n_recovered']},
                        {'metrica': 'taxa_recuperacao_pct',
                         'valor': round(elast['n_recovered']/elast['n_events']*100,1)
                                  if elast['n_events'] > 0 else 0},
                    ])
                    st.download_button(
                        "📥 Download Recovery Elasticity",
                        (_el_dl.to_csv(index=False, sep=';', decimal=',') +
                         '\n--- Métricas ---\n' +
                         _el_meta.to_csv(index=False, sep=';', decimal=',')
                        ).encode('utf-8'),
                        "atheltica_hrv_elasticidade.csv", "text/csv",
                        key="adv_elast_dl"
                    )

    # ── LAG AVANÇADO ─────────────────────────────────────────────────────────
    with _adv_tabs[3]:
        st.markdown("#### 🔗 Lag Correlation Avançada (Pearson + Spearman + MI)")
        st.caption(
            "Pearson: magnitude linear. "
            "Spearman: robusto a outliers. "
            "MI normalizada: detecta relações não-lineares (U-shape dose-resposta)."
        )

        _lv1, _lv2 = st.columns(2)
        _lv_tgt  = _lv1.selectbox(
            "HRV alvo", [v for v in ['hrv','ln_hrv','hrv_norm'] if v in sig_hrv.columns],
            key="adv_lag_tgt")
        _lv_max  = _lv2.slider("Lag máximo (d)", 3, 21, 10, 1, key="adv_lag_max")

        if st.button("▶ Calcular (pode demorar 10-20s)", type="primary", key="adv_lag_run"):
            with st.spinner("Pearson + Spearman + MI..."):
                adv_lag = _lag_correlations_advanced(
                    sig_hrv, sig_train, hrv_var=_lv_tgt, max_lag=_lv_max)

            if adv_lag.empty:
                st.warning("Sem dados suficientes.")
            else:
                # Top por cada método
                st.markdown("**Melhor lag por variável — comparação dos 3 métodos:**")

                def _best_per_var(df, r_col, sig_col):
                    sub = df[df[sig_col]].copy() if sig_col in df.columns else df.copy()
                    if sub.empty: return pd.DataFrame()
                    idx = sub.groupby('var')[r_col].apply(lambda x: x.abs().idxmax())
                    return sub.loc[idx].reset_index(drop=True)

                _bp = _best_per_var(adv_lag, 'r_pearson',  'sig_pearson')
                _bs = _best_per_var(adv_lag, 'r_spearman', 'sig_spearman')

                _comp_rows = []
                for var in adv_lag['var'].unique():
                    _rp  = _bp[_bp['var']==var]['r_pearson'].values
                    _lp  = _bp[_bp['var']==var]['lag'].values
                    _rs  = _bs[_bs['var']==var]['r_spearman'].values
                    _ls  = _bs[_bs['var']==var]['lag'].values
                    _mi  = adv_lag[adv_lag['var']==var]['mi_norm'].max()
                    _comp_rows.append({
                        'Variável': var,
                        'r Pearson': f"{_rp[0]:+.3f} @{_lp[0]}d" if len(_rp) else '—',
                        'r Spearman': f"{_rs[0]:+.3f} @{_ls[0]}d" if len(_rs) else '—',
                        'MI norm max': f"{_mi:.3f}" if not np.isnan(_mi) else '—',
                        'MI>Pearson?': ('✅' if not np.isnan(_mi) and
                                         _mi > (abs(_rp[0]) if len(_rp) else 0) else ''),
                    })

                st.dataframe(pd.DataFrame(_comp_rows), hide_index=True,
                             use_container_width=True)
                st.caption(
                    "MI>Pearson? = a MI detectou relação mais forte que Pearson "
                    "(possível não-linearidade / dose-resposta em U)."
                )

                st.download_button(
                    "📥 Download lag correlation avançada",
                    adv_lag.to_csv(index=False, sep=';', decimal=',').encode(),
                    "hrv_lag_advanced.csv", "text/csv", key="adv_lag_dl")

    # ── DIRECTIONAL ───────────────────────────────────────────────────────────
    with _adv_tabs[4]:
        st.markdown("#### ➡️ Directional Analysis")
        st.caption(
            "Padrões de treino → probabilidade de HRV melhorar nos X dias seguintes. "
            "⚠️ Ferramenta de geração de hipóteses, não inferência causal. "
            "N pequeno → confidence baixa."
        )

        # Padrões pré-definidos fisiologicamente
        _DEFAULT_PATTERNS = [
            {'name': 'Monotonia↓ >15%',
             'conditions': [{'var': 'mono_7d_delta', 'op': '<', 'val': -0.15}]},
            {'name': 'Carga↓ + Freq.↓',
             'conditions': [{'var': 'load_7d_delta', 'op': '<', 'val': -0.20},
                             {'var': 'freq_7d', 'op': '<', 'val': 4}]},
            {'name': 'Alta Monotonia (>2.0)',
             'conditions': [{'var': 'mono_7d', 'op': '>', 'val': 2.0}]},
            {'name': 'Strain↓ + Z2↑ (pct_z3<30%)',
             'conditions': [{'var': 'strain_7d_delta', 'op': '<', 'val': -0.15},
                             {'var': 'pct_z3', 'op': '<', 'val': 30}]},
            {'name': 'TSB positivo (>+5)',
             'conditions': [{'var': 'tsb', 'op': '>', 'val': 5}]},
            {'name': 'Carga muito elevada (ATL>CTL×1.2)',
             'conditions': [{'var': 'atl', 'op': '>', 'val': 0}]},  # proxy
        ]

        _dp_lag = st.slider("Janela de outcome (dias)", 3, 10, 5, 1, key="dir_lag")

        if st.button("▶ Analisar padrões", type="primary", key="dir_run"):
            _dir_res = _directional_analysis(
                sig_hrv, sig_train, _DEFAULT_PATTERNS, outcome_lag=_dp_lag)

            _dir_df = pd.DataFrame([{
                'Padrão':       r['pattern'],
                'Ocorrências':  r['n_occur'],
                'HRV melhorou': r['n_improve'],
                'Consistência': f"{r['consistency']}%",
                'Confidence':   r['confidence'],
            } for r in _dir_res])

            if not _dir_df.empty:
                st.dataframe(_dir_df, hide_index=True, use_container_width=True)
                st.caption(
                    "Consistência = % das ocorrências seguidas de HRV melhorado "
                    f"nos {_dp_lag} dias seguintes. "
                    "Confidence indica o tamanho amostral — N<10 é exploratório."
                )
                st.download_button(
                    "📥 Download Directional Analysis",
                    _dir_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                    f"atheltica_hrv_directional_lag{_dp_lag}d.csv", "text/csv",
                    key="adv_dir_dl"
                )

    # ── DOSE-RESPONSE ─────────────────────────────────────────────────────────
    with _adv_tabs[5]:
        st.markdown("#### 📈 Dose-Response Curves (LOWESS)")
        st.caption(
            "Relação não-linear entre variável de treino e HRV. "
            "LOWESS detecta U-shapes (ex: carga baixa/alta → HRV ruim, carga moderada → HRV óptimo)."
        )

        _dr1, _dr2, _dr3 = st.columns(3)
        _dr_xvar = _dr1.selectbox(
            "Variável de treino (X)",
            [v for v in ['mono_7d','strain_7d','load_7d','pct_z3','freq_7d','atl','tsb']
             if v in sig_train.columns],
            key="dr_xvar")
        _dr_lag  = _dr2.slider("Lag (dias)", 0, 10, 3, 1, key="dr_lag")
        _dr_yvar = _dr3.selectbox(
            "HRV alvo (Y)",
            [v for v in ['hrv','ln_hrv','hrv_norm'] if v in sig_hrv.columns],
            key="dr_yvar")

        if st.button("▶ Calcular dose-response", type="primary", key="dr_run"):
            dr = _dose_response(sig_hrv, sig_train, _dr_xvar, _dr_yvar, _dr_lag)

            if dr.empty:
                st.warning("Dados insuficientes.")
            else:
                _fig_dr = go.Figure()
                # Scatter pontos reais
                _fig_dr.add_trace(go.Scatter(
                    x=dr['x_raw'], y=dr['y_raw'],
                    mode='markers', name='Dados',
                    marker=dict(size=5, color='rgba(41,128,185,0.3)',
                                line=dict(width=0)),
                    hovertemplate=f'{_dr_xvar}: %{{x:.2f}}<br>{_dr_yvar}: %{{y:.1f}}<extra></extra>'
                ))
                # Curva LOWESS
                _fig_dr.add_trace(go.Scatter(
                    x=dr['x'], y=dr['y_smooth'],
                    mode='lines', name='LOWESS',
                    line=dict(color='#e74c3c', width=3),
                    hovertemplate=f'{_dr_xvar}: %{{x:.2f}}<br>HRV smooth: %{{y:.1f}}<extra></extra>'
                ))
                _fig_dr.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='#111', size=11), height=360,
                    margin=dict(t=20, b=60, l=60, r=20),
                    xaxis=dict(title=_dr_xvar, tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title=f'{_dr_yvar} (lag={_dr_lag}d)',
                               tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#eee'),
                    legend=dict(orientation='h', y=-0.18,
                                font=dict(color='#111', size=10)),
                )
                st.plotly_chart(_fig_dr, use_container_width=True,
                                config=MC, key='dr_plot')
                st.caption(
                    f"X = {_dr_xvar} no dia t. "
                    f"Y = {_dr_yvar} no dia t+{_dr_lag}. "
                    "Curva LOWESS capta relações não-lineares — pico = zona óptima."
                )
                st.download_button(
                    "📥 Download Dose-Response (dados + curva LOWESS)",
                    dr.round(3).to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                    f"atheltica_hrv_dose_response_{_dr_xvar}_lag{_dr_lag}d.csv",
                    "text/csv", key="adv_dr_dl"
                )

    # ── K-MEANS ───────────────────────────────────────────────────────────────
    with _adv_tabs[6]:
        st.markdown("#### 🗂️ Clustering de Semanas")
        st.caption(
            "K-means sobre variáveis de TREINO (sem HRV). "
            "Clusters coloridos pelo HRV médio da semana seguinte. "
            "⚠️ N pequeno (~100 semanas) — interpretar com cautela."
        )

        _kk1, _kk2 = st.columns(2)
        _n_clust = _kk1.slider("Nº clusters", 2, 6, 4, 1, key="km_n")

        try:
            import sklearn
            _has_sklearn = True
        except ImportError:
            _has_sklearn = False

        if not _has_sklearn:
            st.warning("sklearn não disponível. Instala scikit-learn para usar esta análise.")
        elif st.button("▶ Clusturizar semanas", type="primary", key="km_run"):
            with st.spinner("K-means..."):
                wk_df = _cluster_weeks(sig_hrv, sig_train, n_clusters=_n_clust)

            if wk_df.empty:
                st.warning("Dados insuficientes (mínimo 12 semanas completas).")
            else:
                # Tabela de características por cluster
                _feat_cols = ['load_total','mono_mean','freq','pct_z3','strain_mean','hrv_next']
                _clust_summary = wk_df.groupby('cluster_label')[_feat_cols].mean().round(2)
                st.markdown("**Características médias por cluster:**")
                st.dataframe(_clust_summary, use_container_width=True)

                # Scatter semanas ao longo do tempo
                _cmap = {'🟢 Semana Óptima': '#27ae60', '🟡 Semana Boa': '#f39c12',
                          '🟠 Semana de Atenção': '#e67e22', '🔴 Semana Difícil': '#e74c3c'}
                _fig_km = go.Figure()
                for lbl, color in _cmap.items():
                    _sub = wk_df[wk_df['cluster_label'] == lbl]
                    if len(_sub) == 0: continue
                    _fig_km.add_trace(go.Scatter(
                        x=_sub['week'].astype(str), y=_sub['hrv_next'],
                        mode='markers', name=lbl,
                        marker=dict(size=9, color=color,
                                    line=dict(width=1, color='white')),
                        hovertemplate='Semana %{x}<br>HRV seguinte: <b>%{y:.1f}</b><extra></extra>'
                    ))
                _fig_km.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='#111', size=10), height=320,
                    margin=dict(t=20, b=80, l=60, r=20),
                    xaxis=dict(tickangle=-45, tickfont=dict(color='#111', size=8),
                               showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title='HRV semana seguinte', tickfont=dict(color='#111'),
                               showgrid=True, gridcolor='#eee'),
                    legend=dict(orientation='h', y=-0.30,
                                font=dict(color='#111', size=9)),
                )
                st.plotly_chart(_fig_km, use_container_width=True,
                                config=MC, key='km_scatter')

                _current_week = pd.Timestamp.now().to_period('W')
                _current_row  = wk_df[wk_df['week'] == _current_week]
                if len(_current_row) > 0:
                    _clbl = _current_row['cluster_label'].values[0]
                    st.info(f"**Semana actual:** {_clbl}")

                # Download K-means
                _km_dl = wk_df[['week','cluster_label','load_total','mono_mean',
                                  'freq','pct_z3','strain_mean',
                                  'hrv_mean','hrv_next']].copy()
                _km_dl['week'] = _km_dl['week'].astype(str)
                st.download_button(
                    "📥 Download clustering de semanas",
                    _km_dl.round(3).to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
                    f"atheltica_hrv_clusters_{_n_clust}k.csv", "text/csv",
                    key="adv_km_dl"
                )

    # ── TRANSIÇÕES ───────────────────────────────────────────────────────────
    with _adv_tabs[7]:
        st.markdown("#### 🔄 Probabilistic Transition Matrix")
        st.caption(
            "P(estado_amanhã | estado_hoje). "
            "Alternativa ao Sankey — mostra probabilidades reais entre estados."
        )

        _state_df2 = _classify_states(sig_hrv, sig_train)
        _state_labels = _state_df2['state_label'].dropna()

        if len(_state_labels) < 10:
            st.warning("Dados insuficientes para transition matrix.")
        else:
            _tm = _transition_matrix(_state_df2['state_label'])

            if not _tm.empty:
                # Heatmap da transition matrix
                _fig_tm = go.Figure(go.Heatmap(
                    z=_tm.values,
                    x=list(_tm.columns),
                    y=list(_tm.index),
                    colorscale='Blues',
                    zmin=0, zmax=1,
                    text=_tm.round(2).values,
                    texttemplate='%{text}',
                    colorbar=dict(title='P', tickfont=dict(color='#111')),
                    hovertemplate='De: %{y}<br>Para: %{x}<br>P = <b>%{z:.2f}</b><extra></extra>'
                ))
                _fig_tm.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='#111', size=9),
                    height=max(320, len(_tm) * 40 + 100),
                    margin=dict(t=20, b=120, l=220, r=20),
                    xaxis=dict(tickangle=-35, tickfont=dict(color='#111', size=8)),
                    yaxis=dict(tickfont=dict(color='#111', size=8)),
                )
                st.plotly_chart(_fig_tm, use_container_width=True,
                                config=MC, key='tm_heat')

                # Insights das transições mais prováveis
                st.markdown("**Transições mais prováveis (P > 0.40):**")
                _trans_rows = []
                for frm in _tm.index:
                    for to in _tm.columns:
                        p = _tm.loc[frm, to]
                        if p > 0.40 and frm != to:
                            _trans_rows.append({
                                'De': frm, 'Para': to, 'P': f"{p:.2f}"})
                if _trans_rows:
                    st.dataframe(pd.DataFrame(_trans_rows), hide_index=True,
                                 use_container_width=True)
                else:
                    st.info("Sem transições com P>0.40 (estados muito distribuídos).")

                st.caption(
                    "Lê-se por linha: dado que hoje estás em estado X, "
                    "qual a probabilidade de amanhã estar em Y? "
                    "Diagonal = auto-persistência do estado."
                )
                st.download_button(
                    "📥 Download Transition Matrix",
                    _tm.to_csv(sep=';', decimal=',').encode('utf-8'),
                    "atheltica_hrv_transition_matrix.csv", "text/csv",
                    key="adv_tm_dl"
                )
