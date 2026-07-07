# ══════════════════════════════════════════════════════════════════════════════
# utils/hrv_analyzer.py — ATHELTICA
# Módulo central de análise HRV avançada. Fonte ÚNICA de verdade para:
#   • Construção de sinais HRV/treino    → _build_hrv_signal, _build_training_signal
#   • Detecção de períodos               → _detect_hrv_periods
#   • Event window / comparação          → _event_window, _compare_periods
#   • Lag correlations (simples+avançado) → _lag_correlations, _lag_correlations_advanced
#   • Fingerprint / classificação estados → _hrv_fingerprint, _classify_states
#   • ARI / elasticidade recuperação     → _compute_ari, _recovery_elasticity
#   • Informação mútua / direccional     → _normalized_mi, _directional_analysis
#   • Dose-response / clusters / transições→ _dose_response, _cluster_weeks, _transition_matrix
#   • AUTO-RUNNER (varredura períodos)    → run_autorunner()
#
# Todas as funções de análise são PURAS (sem Streamlit). O auto-runner CHAMA
# estas funções (não reimplementa) — lógica num só sítio, sem duplicação.
# Extraído de tab_hrv_analyzer.py. Reutilizável por qualquer tab.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import davies_bouldin_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# filtrar_principais vem de utils.data (filtra modalidades principais Bike/Row/Ski/Run)
try:
    from utils.data import filtrar_principais
except Exception:
    def filtrar_principais(df):
        """Fallback: se utils.data não disponível, devolve o df tal como está."""
        return df



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
# AUTO-RUNNER — varredura de parâmetros óptimos por período
# ══════════════════════════════════════════════════════════════════════════════

def _noop(*args, **kwargs):
    """Absorve chamadas de display (st.markdown/dataframe/etc) no modo módulo."""
    return None


def run_autorunner(sig_hrv, sig_train, da_full=None, hoje_ar=None, on_progress=None):
    """
    Corre o Auto-Runner completo: varre múltiplos períodos (180d/1ano/2anos/
    3anos/tudo) × grids de parâmetros, detectando os valores óptimos por
    variável/análise/período. Chama as funções puras deste módulo onde aplicável
    e replica a lógica de optimização original.

    Parâmetros:
      sig_hrv    : DataFrame de _build_hrv_signal (coluna 'Data')
      sig_train  : DataFrame de _build_training_signal (coluna 'Data')
      da_full    : DataFrame de actividades completas (para CTLg por modalidade)
      hoje_ar    : pd.Timestamp de referência (default: hoje)
      on_progress: callback opcional (pct:int, label:str) para progresso

    Devolve dict:
      {'runner_results': [...], 'summary_rows': [...]}
      — mesmas estruturas do auto-runner original, prontas a exportar/mostrar.
    """
    import numpy as np
    import pandas as pd
    import warnings as _warnings
    from scipy.stats import ConstantInputWarning as _CIW
    from scipy.stats import pearsonr as _pr, spearmanr as _sr
    # Segmentos constantes (ex.: períodos sem variação) geram ConstantInputWarning
    # inofensivos na varredura — silenciamos para não poluir os logs.
    _warnings.filterwarnings('ignore', category=_CIW)
    try:
        from sklearn.cluster import KMeans as _KM
        from sklearn.metrics import davies_bouldin_score as _dbs
    except Exception:
        _KM = None; _dbs = None

    if hoje_ar is None:
        hoje_ar = pd.Timestamp.now().normalize()
    _hoje_ar = hoje_ar

    def _report_progress(pct, label=''):
        if on_progress is not None:
            try: on_progress(pct, label)
            except Exception: pass

    _runner_results = []
    _summary_rows   = []

    _periodos_run = [
        (180,   "180 dias"),
        (365,   "1 ano"),
        (730,   "2 anos"),
        (1095,  "3 anos"),
        (99999, "Todo historico"),
    ]

    _LAG_GRID      = [7, 10, 14, 21, 28, 35]
    _LAG_MAX_GRID  = [14, 21, 28, 35]
    _CLUSTER_GRID  = [3, 4, 5, 6, 7]
    _DIR_GRID      = [5, 7, 10, 14]
    _Z_GRID        = [1.0, 1.5, 2.0]
    _FP_GRID       = [3, 5, 7, 14]
    _DR_LAG_GRID   = [0, 3, 5, 7, 10, 14, 21, 28]
    _DR_VARS       = ['load','kj','atl','ctl','pct_z3','mono_7d',
                      'strain_7d','load_28d','freq_7d',
                      'CTLg_Bike','CTLg_Run','CTLg_Ski','CTLg_Row',
                      'ctl_poly2']
    _HRV_ALVO_GRID = [v for v in ['hrv','hrv_norm','hrv_z28']
                      if v in sig_hrv.columns]
    _INCLUDE_CV    = True  # CV% como alvo adicional

    _hoje_ar = pd.Timestamp.now().normalize()

    # ── Construir variaveis novas: CTLg modal, polynomial, CV% ──
    _st_extra = None
    if True:  # (era st.spinner)
        if da_full is not None and 'type' in da_full.columns:
            _da_e = da_full.copy()
            _da_e['Data'] = pd.to_datetime(_da_e['Data'])
            _load_col_e = next((c for c in ['icu_training_load','load']
                               if c in _da_e.columns), None)
            _da_e['_load'] = (pd.to_numeric(_da_e[_load_col_e], errors='coerce').fillna(0)
                              if _load_col_e else 0.0)
            _date_all_e = pd.date_range(_da_e['Data'].min(), _hoje_ar, freq='D')
            _st_extra = pd.DataFrame({'Data': _date_all_e})
            for _mod_e in ['Bike','Run','Row','Ski']:
                _mask_m = _da_e['type'].str.contains(_mod_e, na=False, case=False)
                _dm = (_da_e[_mask_m].groupby('Data')['_load'].sum()
                       .reindex(_date_all_e, fill_value=0))
                _st_extra[f'CTLg_{_mod_e}'] = _dm.ewm(span=42, adjust=False).mean().values
                _st_extra[f'ATLg_{_mod_e}'] = _dm.ewm(span=7,  adjust=False).mean().values

        _st_main_t = sig_train.copy()
        _st_main_t['Data'] = pd.to_datetime(_st_main_t['Data'])
        if 'ctl' in _st_main_t.columns:
            _ctl_s_t = (pd.to_numeric(_st_main_t.set_index('Data')['ctl'], errors='coerce'))
            _WIN_POLY = 60
            _p2_vals = []
            for _i_p in range(len(_ctl_s_t)):
                _seg = _ctl_s_t.iloc[max(0,_i_p-_WIN_POLY+1):_i_p+1].dropna().values
                if len(_seg) >= 10:
                    try:
                        _p2_vals.append(float(np.polyfit(np.arange(len(_seg)), _seg, 2)[0]))
                    except Exception:
                        _p2_vals.append(np.nan)
                else:
                    _p2_vals.append(np.nan)
            _poly2_df = pd.DataFrame({'Data': _ctl_s_t.index, 'ctl_poly2': _p2_vals})
            _st_extra = _poly2_df if _st_extra is None else _st_extra.merge(_poly2_df, on='Data', how='left')

        _hrv_main_e = sig_hrv.copy()
        _hrv_main_e['Data'] = pd.to_datetime(_hrv_main_e['Data'])
        _hrv_col_e  = 'hrv' if 'hrv' in _hrv_main_e.columns else _hrv_main_e.columns[0]
        _hrv_s_cv   = pd.to_numeric(_hrv_main_e[_hrv_col_e], errors='coerce').values
        _hrv_s_cv_s = pd.Series(_hrv_s_cv, index=pd.to_datetime(_hrv_main_e['Data']))
        _cv28_s     = (_hrv_s_cv_s.rolling(28,min_periods=14).std() /
                       _hrv_s_cv_s.rolling(28,min_periods=14).mean() * 100)
        _cv_df_e    = pd.DataFrame({'Data': _cv28_s.index, 'hrv_cv28': _cv28_s.values})
        _st_extra   = _cv_df_e if _st_extra is None else _st_extra.merge(_cv_df_e, on='Data', how='left')

    _n_extra = len([c for c in (_st_extra.columns if _st_extra is not None else [])
                    if c != 'Data'])
    _noop(f"Variaveis adicionais calculadas: {_n_extra} "
               f"(CTLg x4 modalidades, ATLg x4, ctl_poly2, hrv_cv28)")


    for _ndias, _plabel in _periodos_run:
        _noop(f"**▶ Período: {_plabel}**")
        _report_progress(0, _plabel)

        # Filtrar dados para este período
        _cutoff = (_hoje_ar - pd.Timedelta(days=_ndias)
                   if _ndias < 99999 else pd.Timestamp('2000-01-01'))

        # sig_hrv e sig_train têm 'Data' como coluna normal
        def _filtrar_por_data(df_in, cutoff):
            df_c = df_in.copy()
            if 'Data' in df_c.columns:
                df_c['Data'] = pd.to_datetime(df_c['Data'])
                return df_c[df_c['Data'] >= cutoff].reset_index(drop=True)
            # Fallback: tentar índice
            try:
                idx = pd.to_datetime(df_c.index)
                return df_c[idx >= cutoff]
            except Exception:
                return df_c

        _hrv_p  = _filtrar_por_data(sig_hrv, _cutoff)
        _trn_p  = _filtrar_por_data(sig_train, _cutoff)

        if len(_hrv_p) < 30:
            _noop(f"  Dados insuficientes para {_plabel} (N={len(_hrv_p)}).")
            continue

        # Construir série HRV indexada por Data para correlações com shift
        _hrv_col = 'hrv' if 'hrv' in _hrv_p.columns else _hrv_p.columns[0]
        _hrv_series = pd.to_numeric(_hrv_p[_hrv_col], errors='coerce')
        if 'Data' in _hrv_p.columns:
            _hrv_series.index = pd.to_datetime(_hrv_p['Data'])
        _hrv_vals = _hrv_series.dropna()
        _N_hrv    = len(_hrv_vals)

        # Construir série de treino indexada por Data
        if 'Data' in _trn_p.columns:
            _trn_idx = _trn_p.set_index(pd.to_datetime(_trn_p['Data'])).drop(columns=['Data'])
        else:
            _trn_idx = _trn_p.copy()
        # Alinhar índice ao mesmo range que hrv
        _date_range_p = pd.date_range(_cutoff, _hoje_ar, freq='D')
        _hrv_vals_ri  = _hrv_vals.reindex(_date_range_p)
        _trn_idx_ri   = _trn_idx.reindex(_date_range_p)

        # Fundir variáveis extra (CTLg modal, polynomial, CV%) se disponíveis
        if _st_extra is not None:
            _extra_idx = (_st_extra.set_index(pd.to_datetime(_st_extra['Data']))
                          .drop(columns=['Data'])
                          .reindex(_date_range_p))
            # Adicionar colunas que ainda não existem no _trn_idx_ri
            for _ec in _extra_idx.columns:
                if _ec not in _trn_idx_ri.columns:
                    _trn_idx_ri[_ec] = _extra_idx[_ec]

        # CV% HRV como série alvo extra (além dos HRV alvos normais)
        _cv28_ri = None
        if _INCLUDE_CV and 'hrv_cv28' in _trn_idx_ri.columns:
            _cv28_ri = _trn_idx_ri['hrv_cv28'].copy()
            _trn_idx_ri = _trn_idx_ri.drop(columns=['hrv_cv28'])  # remover de preditores

        # ── A. Lag correlations: HRV alvo × lag máximo × variáveis treino
        # Testa todas as variáveis HRV alvo disponíveis ×
        # todos os lag máximos do grid × todas as variáveis de treino
        _lag_vars = [c for c in _trn_idx_ri.columns
                     if _trn_idx_ri[c].notna().sum() >= 20]
        _best_lag_por_var = {}  # sumário — hrv principal, lag máx maior

        # Construir séries HRV alvo filtradas para este período
        _hrv_p_df = _filtrar_por_data(sig_hrv, _cutoff)
        _hrv_alvo_series = {}
        for _hrv_alvo in _HRV_ALVO_GRID:
            if _hrv_alvo not in _hrv_p_df.columns: continue
            if 'Data' in _hrv_p_df.columns:
                _s = pd.Series(
                    pd.to_numeric(_hrv_p_df[_hrv_alvo].values, errors='coerce'),
                    index=pd.to_datetime(_hrv_p_df['Data'])
                )
            else:
                _s = pd.to_numeric(_hrv_p_df[_hrv_alvo], errors='coerce')
            _hrv_alvo_series[_hrv_alvo] = _s.reindex(_date_range_p)

        for _hrv_alvo, _hrv_alvo_ri in _hrv_alvo_series.items():
            for _lmax in _LAG_MAX_GRID:
                for _var in _lag_vars:
                    x_full = pd.to_numeric(_trn_idx_ri[_var], errors='coerce')
                    best_v = {'lag': 0, 'r_pearson': 0, 'r_spearman': 0,
                              'r_abs': 0, 'n': 0}
                    for _lag in range(0, _lmax + 1):
                        df_xy = pd.DataFrame(
                            {'x': x_full.shift(_lag), 'y': _hrv_alvo_ri}
                        ).dropna()
                        if len(df_xy) < 15: continue
                        try:
                            rp, pp = _pr(df_xy['x'].values, df_xy['y'].values)
                            rs, ps = _sr(df_xy['x'].values, df_xy['y'].values)
                            if abs(rp) > abs(best_v['r_abs']):
                                best_v = {
                                    'lag':        _lag,
                                    'r_pearson':  round(rp, 4),
                                    'r_spearman': round(rs, 4),
                                    'p_pearson':  round(pp, 4),
                                    'r_abs':      round(abs(rp), 4),
                                    'n':          len(df_xy),
                                }
                        except Exception:
                            pass
                    if best_v['r_abs'] > 0:
                        # Guardar para sumário (só hrv principal + lag_max maior)
                        if (_hrv_alvo == _HRV_ALVO_GRID[0]
                                and _lmax == max(_LAG_MAX_GRID)):
                            _best_lag_por_var[_var] = best_v
                        _runner_results.append({
                            'periodo':          _plabel,
                            'analise':          'lag_correlation',
                            'variavel':         _var,
                            'hrv_alvo':         _hrv_alvo,
                            'lag_max_testado':  _lmax,
                            'param_nome':       'lag_optimo_dias',
                            'param_val':        best_v['lag'],
                            'r_pearson':        best_v['r_pearson'],
                            'r_spearman':       best_v['r_spearman'],
                            'p_pearson':        best_v.get('p_pearson'),
                            'n':                best_v.get('n'),
                            'r_abs':            best_v['r_abs'],
                            'nota':             '',
                        })

        _report_progress(20, _plabel)

        # ── B. Lag máximo óptimo por HRV alvo ─────────────────────────
        # Para cada HRV alvo, qual lag máximo maximiza o r² médio
        for _hrv_alvo_b, _hrv_ri_b in _hrv_alvo_series.items():
            _lag_max_scores_b = {}
            for _lmax in _LAG_MAX_GRID:
                _scores_b = []
                for _var in list(_best_lag_por_var.keys())[:6]:
                    x_full = pd.to_numeric(
                        _trn_idx_ri.get(_var, pd.Series(dtype=float)), errors='coerce')
                    best_r = 0
                    for _lag in range(0, _lmax + 1):
                        df_xy = pd.DataFrame(
                            {'x': x_full.shift(_lag), 'y': _hrv_ri_b}).dropna()
                        if len(df_xy) < 15: continue
                        try:
                            rp, _ = _pr(df_xy['x'].values, df_xy['y'].values)
                            best_r = max(best_r, abs(rp))
                        except Exception:
                            pass
                    _scores_b.append(best_r)
                _lag_max_scores_b[_lmax] = float(np.mean(_scores_b)) if _scores_b else 0
            _best_lmax_b = max(_lag_max_scores_b, key=_lag_max_scores_b.get)
            _runner_results.append({
                'periodo':    _plabel,
                'analise':    'lag_max_optimo',
                'variavel':   'global',
                'hrv_alvo':   _hrv_alvo_b,
                'lag_max_testado': None,
                'param_nome': 'lag_max_dias',
                'param_val':  _best_lmax_b,
                'r_pearson':  round(_lag_max_scores_b[_best_lmax_b], 4),
                'r_spearman': None, 'p_pearson': None, 'n': _N_hrv,
                'r_abs':      round(_lag_max_scores_b[_best_lmax_b], 4),
                'nota':       f"Scores por lag_max: {_lag_max_scores_b}",
            })
        # Para sumário: usar hrv principal
        _hrv_ri_principal = _hrv_alvo_series.get(_HRV_ALVO_GRID[0], _hrv_vals_ri)
        _lag_max_scores_p = {k: v for k, v in _lag_max_scores_b.items()
                             if _hrv_alvo_b == _HRV_ALVO_GRID[0]} \
            if _HRV_ALVO_GRID else {28: 0}
        _best_lmax = _best_lmax_b if _hrv_alvo_b == _HRV_ALVO_GRID[0] else 28

        _report_progress(40, _plabel)

        # ── C. Clustering: qual n_clusters tem menor Davies-Bouldin ──
        _wk_cols = [c for c in _trn_idx_ri.columns if _trn_idx_ri[c].notna().sum() >= 10]
        _wk_data = _trn_idx_ri[_wk_cols].resample('W').mean().dropna(how='all')
        _wk_data = _wk_data.fillna(_wk_data.median())
        _best_nc = 4; _best_db = 9999
        _cluster_scores = {}
        if len(_wk_data) >= 15:
            from sklearn.preprocessing import StandardScaler as _SS
            _X_wk = _SS().fit_transform(_wk_data.values)
            for _nc in _CLUSTER_GRID:
                if _nc >= len(_wk_data): continue
                try:
                    _km = _KM(n_clusters=_nc, random_state=42, n_init=10)
                    _labels = _km.fit_predict(_X_wk)
                    _db = _dbs(_X_wk, _labels)
                    _cluster_scores[_nc] = round(_db, 4)
                    if _db < _best_db:
                        _best_db = _db; _best_nc = _nc
                except Exception:
                    pass
        _runner_results.append({
            'periodo': _plabel, 'analise': 'clustering_semanas',
            'variavel': 'global', 'param_nome': 'n_clusters',
            'param_val': _best_nc,
            'r_pearson': None, 'r_spearman': None, 'p_pearson': None,
            'n': len(_wk_data),
            'r_abs': round(1 / _best_db, 4) if _best_db > 0 else 0,
            'nota': f"Davies-Bouldin por n_clusters: {_cluster_scores} — menor=melhor",
        })

        _report_progress(55, _plabel)

        # ── D. Directional: qual janela maximiza consistência ─────────
        # Critério correcto: HRV médio dos N dias APÓS evento
        #                   vs HRV médio dos N dias ANTES do evento (baseline)
        # Responde a: "após carga elevada, o HRV dos dias seguintes foi
        # acima ou abaixo do baseline pré-evento?"
        # N por janela pode diferir: eventos muito próximos podem ser excluídos
        # se a janela pós-evento se sobrepõe ao próximo evento
        _dir_scores    = {}
        _dir_n_eventos = {}
        if 'atl' in _trn_idx_ri.columns:
            _atl  = pd.to_numeric(_trn_idx_ri.get('atl', pd.Series(dtype=float)), errors='coerce')
            _ctl  = (pd.to_numeric(_trn_idx_ri.get('ctl', pd.Series(dtype=float)), errors='coerce')
                     if 'ctl' in _trn_idx_ri.columns else None)
            _hrv_s = _hrv_vals_ri.reindex(_atl.index, method='nearest')

            # Identificar todos os eventos (ATL > CTL×1.2)
            _event_dates = []
            for _dt in _atl.index:
                if _ctl is not None:
                    _cv = float(_ctl.get(_dt, 0) or 0)
                    _av = float(_atl.get(_dt, 0) or 0)
                    if _cv > 0 and _av / _cv > 1.2:
                        _event_dates.append(_dt)
                else:
                    if float(_atl.get(_dt, 0) or 0) >= 25:
                        _event_dates.append(_dt)

            for _jdir in _DIR_GRID:
                _ok = 0; _total = 0
                for _i_ev, _dt in enumerate(_event_dates):
                    # Baseline = média HRV dos _jdir dias ANTES do evento
                    _pre_vals = [
                        float(_hrv_s.get(_dt - pd.Timedelta(days=k), np.nan) or np.nan)
                        for k in range(1, _jdir + 1)
                    ]
                    _pre_vals = [v for v in _pre_vals if np.isfinite(v)]
                    if len(_pre_vals) < max(1, _jdir // 2): continue  # mín. metade da janela

                    # Outcome = média HRV dos _jdir dias APÓS o evento
                    _fut_vals = [
                        float(_hrv_s.get(_dt + pd.Timedelta(days=k), np.nan) or np.nan)
                        for k in range(1, _jdir + 1)
                    ]
                    _fut_vals = [v for v in _fut_vals if np.isfinite(v)]
                    if len(_fut_vals) < max(1, _jdir // 2): continue

                    _hrv_baseline = float(np.mean(_pre_vals))
                    _hrv_outcome  = float(np.mean(_fut_vals))
                    _total += 1
                    if _hrv_outcome > _hrv_baseline: _ok += 1

                _dir_scores[_jdir]    = round(_ok / _total, 4) if _total >= 5 else None
                _dir_n_eventos[_jdir] = _total

        _valid_dir  = {k: v for k, v in _dir_scores.items() if v is not None}
        _best_jdir  = (max(_valid_dir, key=lambda k: abs(_valid_dir[k] - 0.5))
                       if _valid_dir else 10)
        _best_n_dir = _dir_n_eventos.get(_best_jdir, 0)
        _runner_results.append({
            'periodo':    _plabel,
            'analise':    'directional_janela',
            'variavel':   'Carga muito elevada (ATL>CTL×1.2)',
            'param_nome': 'janela_outcome_dias',
            'param_val':  _best_jdir,
            'r_pearson':  None, 'r_spearman': None, 'p_pearson': None,
            'n':          _best_n_dir,
            'r_abs':      round(abs(_valid_dir.get(_best_jdir, 0.5) - 0.5), 4),
            'nota':       (f"Critério: HRV(pós N dias) vs HRV(pré N dias baseline) | "
                           f"Consistência: {_valid_dir} | "
                           f"N eventos por janela: {_dir_n_eventos}"),
        })

        _report_progress(65, _plabel)

        # ── E. Fingerprint — grid de dias antes ────────────────────────
        _fp_vars = [c for c in _trn_idx_ri.columns
                    if _trn_idx_ri[c].notna().sum() >= 20]
        _hrv_q10 = float(_hrv_vals_ri.dropna().quantile(0.10))
        _hrv_q90 = float(_hrv_vals_ri.dropna().quantile(0.90))
        for _fp_dias in _FP_GRID:
            for _fp_var in _fp_vars:
                x_fp = pd.to_numeric(_trn_idx_ri[_fp_var], errors='coerce')
                _vals_high = []; _vals_low = []
                for _dt in _hrv_vals_ri.dropna().index:
                    _hv = float(_hrv_vals_ri.get(_dt, np.nan) or np.nan)
                    if not np.isfinite(_hv): continue
                    _prev = [float(x_fp.get(
                        _dt - pd.Timedelta(days=k), np.nan) or np.nan)
                        for k in range(1, _fp_dias+1)]
                    _prev = [v for v in _prev if np.isfinite(v)]
                    if not _prev: continue
                    _mp = float(np.mean(_prev))
                    if _hv >= _hrv_q90:
                        _vals_high.append(_mp)
                    elif _hv <= _hrv_q10:
                        _vals_low.append(_mp)
                if not _vals_high or not _vals_low: continue
                _mh = float(np.mean(_vals_high))
                _ml = float(np.mean(_vals_low))
                _dp = ((_mh - _ml) / abs(_ml) * 100) if _ml != 0 else None
                _runner_results.append({
                    'periodo': _plabel, 'analise': 'fingerprint',
                    'variavel': _fp_var,
                    'param_nome': 'dias_antes',
                    'param_val': _fp_dias,
                    'r_pearson': round(_dp, 2) if _dp else None,
                    'r_spearman': None, 'p_pearson': None,
                    'n': len(_vals_high) + len(_vals_low),
                    'r_abs': round(abs(_dp), 2) if _dp else None,
                    'nota': (f"HRV alto: {_mh:.2f} | HRV baixo: {_ml:.2f} | "
                             f"Diff%: {_dp:.1f}%") if _dp else "sem dados",
                })

        _report_progress(80, _plabel)

        # ── F. Dose-Response — Spearman + quartil óptimo ───────────────
        _DR_VARS = ['load','kj','atl','pct_z3','mono_7d',
                    'strain_7d','load_28d','freq_7d']
        _DR_LAG_GRID = [0, 3, 5, 7, 10, 14, 21, 28]
        for _dr_var in _DR_VARS:
            if _dr_var not in _trn_idx_ri.columns: continue
            x_dr = pd.to_numeric(_trn_idx_ri[_dr_var], errors='coerce')
            _dr_best = {'lag': 0, 'r_sp': 0, 'r_abs': 0,
                        'hrv_max_q': None, 'n': 0}
            for _dr_lag in _DR_LAG_GRID:
                _df_dr = pd.DataFrame(
                    {'x': x_dr.shift(_dr_lag), 'y': _hrv_vals_ri}).dropna()
                if len(_df_dr) < 20: continue
                try:
                    from scipy.stats import spearmanr as _sr_dr
                    rs, _ = _sr_dr(_df_dr['x'].values, _df_dr['y'].values)
                    _df_dr['_xq'] = pd.qcut(_df_dr['x'], q=4,
                        labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
                    _hq = _df_dr.groupby('_xq', observed=True)['y'].mean()
                    _bq = str(_hq.idxmax()) if len(_hq) > 0 else None
                    if abs(rs) > abs(_dr_best['r_abs']):
                        _dr_best = {'lag': _dr_lag, 'r_sp': round(rs,4),
                                    'r_abs': round(abs(rs),4),
                                    'hrv_max_q': _bq, 'n': len(_df_dr)}
                except Exception:
                    pass
            if _dr_best['r_abs'] > 0:
                _runner_results.append({
                    'periodo': _plabel, 'analise': 'dose_response',
                    'variavel': _dr_var, 'param_nome': 'lag_optimo_dias',
                    'param_val': _dr_best['lag'],
                    'r_pearson': None, 'r_spearman': _dr_best['r_sp'],
                    'p_pearson': None, 'n': _dr_best['n'],
                    'r_abs': _dr_best['r_abs'],
                    'nota': (f"HRV max em {_dr_best['hrv_max_q']} "
                             f"de {_dr_var} @lag{_dr_best['lag']}d"),
                })

        _report_progress(70, _plabel)

        # ── E. Elasticidade: qual target_z detecta mais eventos ────────
        _ela_scores = {}
        _hrv_roll = _hrv_vals_ri.rolling(14, min_periods=7).mean()
        _hrv_std  = _hrv_vals_ri.rolling(14, min_periods=7).std()
        for _tz in _Z_GRID:
            _n_events = 0
            _taus     = []
            for _i in range(14, len(_hrv_vals_ri)):
                _dt  = _hrv_vals_ri.index[_i]
                _mu  = float(_hrv_roll.iloc[_i - 1] if not pd.isna(_hrv_roll.iloc[_i - 1]) else np.nan)
                _sd  = float(_hrv_std.iloc[_i - 1]  if not pd.isna(_hrv_std.iloc[_i - 1])  else np.nan)
                _val = float(_hrv_vals_ri.iloc[_i]  if not pd.isna(_hrv_vals_ri.iloc[_i])   else np.nan)
                if not all(np.isfinite([_mu, _sd, _val])): continue
                if _sd <= 0: continue
                if (_mu - _val) / _sd >= _tz:
                    _n_events += 1
                    for _k in range(1, 15):
                        if _i + _k >= len(_hrv_vals_ri): break
                        _v_next = _hrv_vals_ri.iloc[_i + _k]
                        if pd.isna(_v_next): continue
                        if float(_v_next) >= _mu - _sd * 0.5:
                            _taus.append(_k); break
            _ela_scores[_tz] = {
                'n_events': _n_events,
                'tau_med':  round(float(np.median(_taus)), 1) if _taus else None,
            }
        # Óptimo: target_z que tem N razoável (≥10) e tau estável
        _best_tz = min(
            [tz for tz, v in _ela_scores.items() if v['n_events'] >= 10],
            key=lambda tz: abs(_ela_scores[tz].get('tau_med', 99) - 2.0),
            default=1.5
        )
        _runner_results.append({
            'periodo': _plabel, 'analise': 'elasticidade_target_z',
            'variavel': 'hrv', 'param_nome': 'target_z',
            'param_val': _best_tz,
            'r_pearson': None, 'r_spearman': None, 'p_pearson': None,
            'n': _ela_scores[_best_tz]['n_events'],
            'r_abs': None,
            'nota': str(_ela_scores),
        })

        _report_progress(85, _plabel)

        # ── G. CV% HRV como alvo — preditores de estabilidade ─────
        # CV% baixo = HRV estável = melhor. r negativo com CV% = bom.
        if _cv28_ri is not None and _cv28_ri.notna().sum() >= 20:
            for _var_cv in _lag_vars:
                if _var_cv not in _trn_idx_ri.columns: continue
                x_cv = pd.to_numeric(_trn_idx_ri[_var_cv], errors='coerce')
                best_cv = {'lag':0,'r':0,'r_abs':0,'n':0}
                for _lag_cv in range(0, max(_LAG_MAX_GRID)+1):
                    df_cv = pd.DataFrame({'x':x_cv.shift(_lag_cv),
                                          'y':_cv28_ri}).dropna()
                    if len(df_cv) < 15: continue
                    try:
                        rp_cv, _ = _pr(df_cv['x'].values, df_cv['y'].values)
                        if abs(rp_cv) > best_cv['r_abs']:
                            best_cv = {'lag':_lag_cv,'r':round(rp_cv,4),
                                       'r_abs':round(abs(rp_cv),4),
                                       'n':len(df_cv)}
                    except Exception:
                        pass
                if best_cv['r_abs'] > 0.08:
                    _runner_results.append({
                        'periodo':         _plabel,
                        'analise':         'cv_pct_lag_correlation',
                        'variavel':        _var_cv,
                        'hrv_alvo':        'hrv_cv28',
                        'lag_max_testado': max(_LAG_MAX_GRID),
                        'param_nome':      'lag_optimo_dias',
                        'param_val':       best_cv['lag'],
                        'r_pearson':       best_cv['r'],
                        'r_spearman':      None,
                        'p_pearson':       None,
                        'n':               best_cv['n'],
                        'r_abs':           best_cv['r_abs'],
                        'nota':            ('r negativo = mais variavel → CV baixo = HRV estavel'
                                            if best_cv['r'] < 0 else
                                            'r positivo = mais variavel → CV alto = instavel'),
                    })

        # ── H. Range óptimo calibrado ─────────────────────────────
        # Estado óptimo = CV% < Q33 + HRV > média 28d
        if _cv28_ri is not None and _hrv_vals_ri.notna().sum() >= 20:
            _cv_q33  = float(_cv28_ri.dropna().quantile(0.33))
            _hrv_mu  = float(_hrv_vals_ri.dropna().mean())
            _otimo   = (_cv28_ri < _cv_q33) & (_hrv_vals_ri > _hrv_mu)
            _mau     = (_cv28_ri >= _cv28_ri.dropna().quantile(0.67)) & (_hrv_vals_ri < _hrv_mu)
            if _otimo.sum() >= 10 and 'ctl' in _trn_idx_ri.columns:
                for _metric_r, _col_r in [
                    ('atl','atl'), ('ctl','ctl'), ('tsb','tsb'),
                    ('CTLg_Bike','CTLg_Bike'), ('CTLg_Run','CTLg_Run'),
                    ('CTLg_Ski','CTLg_Ski'), ('ctl_poly2','ctl_poly2'),
                ]:
                    if _col_r not in _trn_idx_ri.columns: continue
                    _s_r = _trn_idx_ri[_col_r]
                    _s_ot = _s_r[_otimo].dropna()
                    _s_mau= _s_r[_mau].dropna()
                    if len(_s_ot) < 5: continue
                    _runner_results.append({
                        'periodo':         _plabel,
                        'analise':         'range_otimo_calibrado',
                        'variavel':        _metric_r,
                        'hrv_alvo':        'hrv_cv28',
                        'lag_max_testado': None,
                        'param_nome':      'iqr_estado_otimo',
                        'param_val':       round(float(_s_ot.median()), 2),
                        'r_pearson':       round(float(_s_ot.quantile(0.25)), 2),
                        'r_spearman':      round(float(_s_ot.quantile(0.75)), 2),
                        'p_pearson':       round(float(_s_mau.median()), 2) if len(_s_mau)>3 else None,
                        'n':               int(_otimo.sum()),
                        'r_abs':           None,
                        'nota':            (f"otimo: med={_s_ot.median():.1f} "
                                            f"IQR=[{_s_ot.quantile(0.25):.1f}-{_s_ot.quantile(0.75):.1f}] "
                                            f"| mau: med={_s_mau.median():.1f} (N={len(_s_mau)})"),
                    })

        _report_progress(90, _plabel)

        # ── I. Sumário do período ──────────────────────────────────────
        _top_neg = sorted(
            [(v,d) for v,d in _best_lag_por_var.items() if d['r_pearson']<0],
            key=lambda x: x[1]['r_abs'], reverse=True)[:2]
        _top_pos = sorted(
            [(v,d) for v,d in _best_lag_por_var.items() if d['r_pearson']>0],
            key=lambda x: x[1]['r_abs'], reverse=True)[:2]

        _fp_this = [r for r in _runner_results
                    if r['periodo']==_plabel and r['analise']=='fingerprint'
                    and r.get('r_abs') is not None]
        _fp_top = (max(_fp_this, key=lambda r: r['r_abs'] or 0)
                   if _fp_this else None)

        _summary_rows.append({
            'Período':              _plabel,
            'N dias HRV':           _N_hrv,
            'Lag máx óptimo':       _best_lmax,
            'N clusters óptimo':    _best_nc,
            'Janela directional (d)': _best_jdir,
            'Consist. directional': f"{_valid_dir.get(_best_jdir,0):.1%}" if _valid_dir else '—',
            'N eventos directional': _best_n_dir,
            'Target Z':             _best_tz,
            'Tau elast. (d)':       _ela_scores[_best_tz].get('tau_med','—'),
            'Melhor preditor ↘ HRV': (
                f"{_top_neg[0][0]} r={_top_neg[0][1]['r_pearson']:+.3f} "
                f"@{_top_neg[0][1]['lag']}d") if _top_neg else '—',
            'FP: var mais discriminante': (
                f"{_fp_top['variavel']} {_fp_top['r_pearson']:+.1f}% "
                f"@{int(_fp_top['param_val'])}d ant.") if _fp_top else '—',
            'N lags sig p<0.05':    sum(
                1 for d in _best_lag_por_var.values()
                if d.get('p_pearson',1) < 0.05),
        })
        _report_progress(100, _plabel)

    # ── Display resumo ────────────────────────────────────────────────
    _noop("---")
    _noop("### 📊 Resumo — parâmetros óptimos por período")
    _noop(
        "📌 **'Todo histórico'** reproduz as condições do CSV antigo (83-96%). "
        "Compara N eventos directional: se alto com todo histórico mas ~52% com 1 ano "
        "→ confirma efeito de N grande, não sinal causal real."
    )

    if _summary_rows:
        _df_sum = pd.DataFrame(_summary_rows)
        _noop(_df_sum, hide_index=True, use_container_width=True)

        # ── Fingerprint top por período ────────────────────────────────
        _fp_all = [r for r in _runner_results
                   if r['analise']=='fingerprint' and r.get('r_abs')]
        if _fp_all:
            _noop("### 👆 Fingerprint — variáveis mais discriminantes (1 ano)")
            _df_fp = pd.DataFrame(_fp_all)
            _fp_1a = (_df_fp[_df_fp['periodo']=='1 ano']
                      .nlargest(10,'r_abs')
                      [['variavel','param_val','r_pearson','n','nota']]
                      .rename(columns={
                          'variavel':'Variável','param_val':'Dias antes',
                          'r_pearson':'Diff% HRV alto vs baixo','n':'N dias'}))
            _noop(_fp_1a, hide_index=True, use_container_width=True)

        # ── Dose-Response por período ──────────────────────────────────
        _dr_all = [r for r in _runner_results if r['analise']=='dose_response']
        if _dr_all:
            _noop("### 📈 Dose-Response — quartil óptimo de carga (1 ano)")
            _df_dr = pd.DataFrame(_dr_all)
            _df_dr_1a = _df_dr[_df_dr['periodo']=='1 ano']
            if len(_df_dr_1a) > 0:
                _noop(
                    _df_dr_1a[['variavel','param_val','r_spearman','n','nota']]
                    .rename(columns={
                        'variavel':'Variável','param_val':'Lag óptimo (d)',
                        'r_spearman':'r Spearman','n':'N pares'}),
                    hide_index=True, use_container_width=True)

        # ── Análise de divergências automática ────────────────────────
        _noop("### 🔍 Divergências entre períodos")
        _div_rows = []

        # Lag máximo
        _lag_vals = [r['Lag máx óptimo'] for r in _summary_rows]
        if max(_lag_vals) - min(_lag_vals) >= 7:
            _div_rows.append({
                'Parâmetro': 'Lag máximo óptimo',
                'Min': f"{min(_lag_vals)}d",
                'Max': f"{max(_lag_vals)}d",
                'Divergência': '⚠️ Alta — lag de resposta ao HRV mudou ao longo do tempo',
            })
        else:
            _div_rows.append({
                'Parâmetro': 'Lag máximo óptimo',
                'Min': f"{min(_lag_vals)}d",
                'Max': f"{max(_lag_vals)}d",
                'Divergência': '✅ Estável entre períodos',
            })

        # Target Z — usar chave correcta 'Target Z'
        _tz_vals = [r['Target Z'] for r in _summary_rows
                    if r['Período'] != 'Todo histórico']
        if _tz_vals and len(set(_tz_vals)) > 1:
            _div_rows.append({
                'Parâmetro': 'Target Z (limiar supressão HRV)',
                'Min': str(min(_tz_vals)),
                'Max': str(max(_tz_vals)),
                'Divergência': '⚠️ Sensibilidade HRV mudou — atleta mais/menos resiliente recentemente',
            })
        elif _tz_vals:
            _div_rows.append({
                'Parâmetro': 'Target Z',
                'Min': str(_tz_vals[0]),
                'Max': str(_tz_vals[0]),
                'Divergência': '✅ Consistente',
            })

        # N clusters
        _nc_vals = [r['N clusters óptimo'] for r in _summary_rows
                    if r['Período'] != 'Todo histórico']
        if _nc_vals and max(_nc_vals) - min(_nc_vals) >= 2:
            _div_rows.append({
                'Parâmetro': 'N clusters óptimo',
                'Min': str(min(_nc_vals)),
                'Max': str(max(_nc_vals)),
                'Divergência': '⚠️ Complexidade dos padrões de treino mudou',
            })
        elif _nc_vals:
            _div_rows.append({
                'Parâmetro': 'N clusters',
                'Min': str(min(_nc_vals)),
                'Max': str(max(_nc_vals)),
                'Divergência': '✅ Estável',
            })

        # Directional: todo histórico vs 1 ano
        _dir_hist = next((r for r in _summary_rows
                          if r['Período'] == 'Todo histórico'), None)
        _dir_1a   = next((r for r in _summary_rows
                          if r['Período'] == '1 ano'), None)
        if _dir_hist and _dir_1a:
            _div_rows.append({
                'Parâmetro': 'Directional consistência',
                'Min': _dir_1a['Consist. directional'],
                'Max': _dir_hist['Consist. directional'],
                'Divergência': (
                    f"⚠️ Efeito N: histór. "
                    f"{_dir_hist['Consist. directional']} "
                    f"(N={_dir_hist['N eventos directional']}) vs "
                    f"1ano {_dir_1a['Consist. directional']} "
                    f"(N={_dir_1a['N eventos directional']})")
            })

        _noop(pd.DataFrame(_div_rows), hide_index=True,
                     use_container_width=True)

        # ── Insights síntese — 180 dias ───────────────────────────────
        _noop("### 💡 Insights — período mais recente (180 dias)")
        _rec = next((r for r in _summary_rows if r['Período']=='180 dias'), {})
        if _rec:
            _noop(f"""
- **Lag de resposta HRV**: {_rec.get('Lag máx óptimo','—')}d — carga hoje afecta HRV daqui a **{_rec.get('Lag máx óptimo','?')} dias**
- **Preditor que mais suprime HRV**: {_rec.get('Melhor preditor ↘ HRV','—')}
- **Fingerprint (var mais discriminante)**: {_rec.get('FP: var mais discriminante','—')}
- **Limiar de supressão (Z)**: {_rec.get('Target Z','—')} | Tau recuperação: {_rec.get('Tau elast. (d)','—')}d
- **Directional (180d)**: {_rec.get('Consist. directional','—')} (N={_rec.get('N eventos directional','—')})
- **Clusters óptimos**: {_rec.get('N clusters óptimo','—')} tipos de semana nos últimos 180d
- **N variáveis sig. (p<0.05)**: {_rec.get('N lags sig p<0.05','—')}
            """)

    return {'runner_results': _runner_results, 'summary_rows': _summary_rows}
