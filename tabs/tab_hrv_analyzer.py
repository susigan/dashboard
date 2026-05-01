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
                            legend=dict(orientation='h', y=-0.08)
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
            _sel_opts = [f"{p['tipo']} {p['start'].strftime('%d/%m/%Y')} → {p['end'].strftime('%d/%m/%Y')}"
                         for p in periods[-int(_show_n):]]
            _sel = st.selectbox("Seleccionar período", _sel_opts,
                                key="hrv_auto_sel")
            if _sel and len(sig_train) > 0:
                _idx   = _sel_opts.index(_sel)
                _pidx  = -(int(_show_n) - _idx)
                _per   = periods[_pidx]
                _ts    = pd.Timestamp(_per['start'])
                _te    = pd.Timestamp(_per['end'])
                cmp    = _compare_periods(sig_hrv, sig_train, _ts, _te, 14)
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
