from utils.config import *
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


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

# ════════════════════════════════════════════════════════════════════════════════
# FRACTIONAL TRAINING LOAD MEMORY (FTLM)
# Della Mattia (2025) — Parts I & II + FMT 2019
#
# Classical PMC uses exponential kernel: K(τ) = e^(−τ/τc)  → saturates quickly
# Fractional FTLM uses power-law kernel: K(τ) = τ^(γ−1)   → long memory, no saturation
#
# CTLγ(t) = (1/Γ(γ)) · Σ  Load(t−k) · k^(γ−1)
#                          k=1..t
#
# γ ∈ (0,1): controls memory depth
#   γ → 1.0 : exponential decay, short memory (≈ classical PMC)
#   γ → 0.3 : power-law decay, long memory (optimal for endurance base)
#   γ → 0.1 : very long memory (months of training persist)
#
# Two γ values are fitted independently (Dual-Gamma, Part II §3):
#   γ_perf    → performance proxy (icu_pm_cp or MMP PR efforts)
#   γ_recovery → HRV trend via moving-window regression (Part II §2)
# ════════════════════════════════════════════════════════════════════════════════

def _ftlm_kernel_weights(n, gamma_val):
    """
    Compute Riemann-Liouville fractional kernel weights for n steps back.
    w(k) = k^(gamma-1) / Γ(gamma)   for k = 1..n

    Vectorised with NumPy — O(n) instead of the naive O(n²) loop in the paper.
    """
    from scipy.special import gamma as _gamma_fn
    k = np.arange(1, n + 1, dtype=np.float64)
    w = np.power(k, gamma_val - 1.0)
    return w / _gamma_fn(gamma_val)


def ftlm_fractional(load_arr, gamma_val, max_lag=None):
    """
    Discrete Riemann-Liouville fractional integral of training load.

    CTLγ(t) = Σ_{k=1}^{t} Load(t-k) · k^(γ-1) / Γ(γ)

    Parameters
    ----------
    load_arr  : array-like, daily load values (chronological order)
    gamma_val : float in (0, 1) — memory depth parameter
    max_lag   : int or None — max days to look back (None = full history)
                For speed with long series, 365 is a good practical limit.

    Returns
    -------
    np.ndarray of same length as load_arr
    """
    load = np.array(load_arr, dtype=np.float64)
    n    = len(load)
    ctl  = np.zeros(n)

    for t in range(n):
        lag   = t if max_lag is None else min(t, max_lag)
        if lag == 0:
            continue
        # Load values at t-k for k=1..lag (reversed so oldest is last)
        past  = load[t-lag:t][::-1]           # past[0] = t-1, past[1] = t-2, …
        k     = np.arange(1, lag + 1, dtype=np.float64)
        from scipy.special import gamma as _gamma_fn
        w     = np.power(k, gamma_val - 1.0) / _gamma_fn(gamma_val)
        ctl[t]= float(np.dot(past, w))

    return ctl


def _hrv_trend(hrv_arr, window=7):
    """
    Extract local HRV trend via moving-window linear regression (Part II §2).
    For each day t, fits HRV(τ) = a(t)·τ + b(t) over [t-window+1, t].
    Returns b(t) — the intercept — as the smoothed HRV trend value.
    NaN for the first (window-1) days.
    """
    from scipy.stats import linregress as _lr
    hrv  = np.array(hrv_arr, dtype=np.float64)
    n    = len(hrv)
    trend= np.full(n, np.nan)
    x    = np.arange(window, dtype=np.float64)
    for t in range(window - 1, n):
        y = hrv[t - window + 1:t + 1]
        if np.isnan(y).any():
            # Use only valid points if enough remain
            mask = ~np.isnan(y)
            if mask.sum() < 4:
                continue
            _, b, *_ = _lr(x[mask], y[mask])
        else:
            _, b, *_ = _lr(x, y)
        trend[t] = b
    return trend


def fit_gamma_performance(load_arr, perf_arr, gamma_range=(0.10, 0.90),
                          step=0.01, lag=0, max_lag=365):
    """
    Fit γ_performance: find γ that maximises R² between CTLγ and a
    performance proxy (icu_pm_cp or mmp_pr_w values).

    Based on Part II §1: Power_test = β₀ + β₁ · CTLγ

    Parameters
    ----------
    load_arr  : array, daily training load (full history)
    perf_arr  : array, daily performance proxy (NaN on non-test days)
    lag       : int, days to lag CTLγ before correlating (default 0)
    max_lag   : int, max history for kernel (365 = ~1 year)

    Returns
    -------
    best_gamma : float
    best_r2    : float
    n_points   : int — number of non-NaN pairs used
    """
    from scipy.stats import pearsonr as _pr
    load  = np.array(load_arr,  dtype=np.float64)
    perf  = np.array(perf_arr,  dtype=np.float64)
    gammas= np.arange(gamma_range[0], gamma_range[1] + step/2, step)

    best_gamma, best_r2 = 0.35, -np.inf
    perf_valid = ~np.isnan(perf)

    for g in gammas:
        ctl_g = ftlm_fractional(load, g, max_lag=max_lag)
        if lag > 0:
            ctl_g = np.roll(ctl_g, lag)
            ctl_g[:lag] = np.nan
        valid = perf_valid & ~np.isnan(ctl_g) & (ctl_g > 0)
        if valid.sum() < 5:
            continue
        try:
            r, _ = _pr(ctl_g[valid], perf[valid])
            r2   = r ** 2
            if r2 > best_r2:
                best_r2, best_gamma = r2, float(g)
        except Exception:
            continue

    n_pts = int(perf_valid.sum())
    return best_gamma, best_r2, n_pts


def fit_gamma_recovery(load_arr, hrv_arr, gamma_range=(0.10, 0.90),
                       step=0.01, hrv_window=7, lag=1, max_lag=365):
    """
    Fit γ_recovery: find γ that maximises R² between CTLγ(t-1) and
    HRV_trend(t), using moving-window regression to extract local HRV
    trend (Part II §2).

    The lag=1 is physiologically motivated: yesterday's load state
    predicts today's autonomic recovery.

    Returns
    -------
    best_gamma : float
    best_r2    : float
    hrv_trend  : np.ndarray — smoothed HRV trend series (for plotting)
    """
    from scipy.stats import pearsonr as _pr
    load  = np.array(load_arr, dtype=np.float64)
    hrv   = np.array(hrv_arr,  dtype=np.float64)
    trend = _hrv_trend(hrv, window=hrv_window)
    gammas= np.arange(gamma_range[0], gamma_range[1] + step/2, step)

    best_gamma, best_r2 = 0.35, -np.inf

    for g in gammas:
        ctl_g = ftlm_fractional(load, g, max_lag=max_lag)
        # Apply lag: CTLγ(t-lag) vs HRV_trend(t)
        if lag > 0:
            ctl_lagged = np.roll(ctl_g, lag)
            ctl_lagged[:lag] = np.nan
        else:
            ctl_lagged = ctl_g
        valid = (~np.isnan(ctl_lagged)) & (~np.isnan(trend)) & (ctl_lagged > 0)
        if valid.sum() < 14:
            continue
        try:
            r, _ = _pr(ctl_lagged[valid], trend[valid])
            r2   = r ** 2
            if r2 > best_r2:
                best_r2, best_gamma = r2, float(g)
        except Exception:
            continue

    return best_gamma, best_r2, trend


def calcular_series_carga(df_act, df_wellness=None, ate_hoje=True):
    """
    Calcula CTL/ATL/TSB clássico (EWM) + FTLM fraccionário Della Mattia (2025).

    FONTE DE CARGA (prioridade):
        1. icu_training_load  (Intervals.icu, escala ~20-200/dia)
        2. session_rpe = (moving_time_min × RPE)   — fallback

    FTLM FRACCIONÁRIO — kernel Riemann-Liouville:
        CTLγ(t) = (1/Γ(γ)) · Σ_{k=1}^{t} Load(t-k) · k^(γ-1)
        Memória em lei de potência — treinos antigos persistem (sem saturação).

    FITTING DUAL DE γ POR MODALIDADE (Della Mattia Part II §1-3):
        Para cada modalidade (Bike, Row, Ski, Run):
            γ_perf_mod → max R²(CTLγ_mod ↔ icu_pm_cp_mod)    lag=0
            γ_mmp_mod  → max R²(CTLγ_mod ↔ MMP_PR_mod)       lag=0  (só sessões is_pr=True)
        γ_recovery (global) → max R²(CTLγ_overall(t-1) ↔ LnRMSSD_trend(t))  lag=1
            HRV trend via regressão janela 7d sobre LnRMSSD (não RMSSD bruto)

    RETORNA:
        ld   : DataFrame — Data, load_val, CTL, ATL, TSB,
                           CTLg_perf, CTLg_rec, FTLM,
                           CTLg_{mod}  para cada modalidade (load fraccionário modal)
        info : dict — gammas, R², fontes, n_pontos por modalidade
    """
    df = filtrar_principais(df_act).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    # ── Fonte de carga ────────────────────────────────────────────────────────
    if 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        df['_load'] = pd.to_numeric(df['icu_training_load'], errors='coerce').fillna(0)
        fonte = 'icu_training_load'
    elif 'moving_time' in df.columns and 'rpe' in df.columns:
        _rpe = pd.to_numeric(df['rpe'], errors='coerce')
        df['_load'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * _rpe.fillna(_rpe.median())
        df['_load'] = df['_load'].fillna(0)
        fonte = 'session_rpe'
    else:
        return pd.DataFrame(), {'fonte': 'none'}

    # ── Série diária OVERALL ──────────────────────────────────────────────────
    ld = df.groupby('Data')['_load'].sum().reset_index().sort_values('Data')
    _idx = pd.date_range(ld['Data'].min(),
                          datetime.now().date() if ate_hoje else ld['Data'].max())
    ld = ld.set_index('Data').reindex(_idx, fill_value=0).reset_index()
    ld.columns = ['Data', 'load_val']

    # ── CTL / ATL / TSB clássicos ──────────────────────────────────────────────
    ld['CTL'] = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL'] = ld['load_val'].ewm(span=7,  adjust=False).mean()
    ld['TSB'] = ld['CTL'] - ld['ATL']

    load_overall = ld['load_val'].values
    MAX_LAG = min(365, len(load_overall))
    _date_idx = pd.DatetimeIndex(ld['Data'])

    # ── γ_recovery via LnRMSSD trend (GLOBAL — HRV é modalidade-agnóstico) ──
    # Part II §2: HRV trend via regressão janela 7d sobre LnRMSSD
    # LnRMSSD (não RMSSD bruto): distribuição normal, compatível com tab_recovery
    gamma_rec, r2_rec = 0.35, 0.0
    hrv_trend_arr = None

    if df_wellness is not None and len(df_wellness) > 0 and 'hrv' in df_wellness.columns:
        _wc = df_wellness.copy()
        _wc['Data'] = pd.to_datetime(_wc['Data'])
        _hrv_raw = (_wc.groupby('Data')['hrv']
                    .mean()
                    .reindex(_date_idx, fill_value=np.nan).values)
        # LnRMSSD: log do RMSSD bruto — distribuição normal, sensibilidade linear
        # Exclui zeros/negativos antes de log
        _hrv_ln = np.where(_hrv_raw > 0, np.log(_hrv_raw), np.nan)
        if int(np.isfinite(_hrv_ln).sum()) >= 21:
            gamma_rec, r2_rec, hrv_trend_arr = fit_gamma_recovery(
                load_overall, _hrv_ln, hrv_window=7, lag=1, max_lag=MAX_LAG)

    # ── γ_perf POR MODALIDADE (Della Mattia Part II §1 + §3) ─────────────────
    # Cada modalidade tem a sua carga, CP e MMP — escalas de potência diferentes
    # Bike MMP20 ≠ Row MMP20 — misturar seria erro sistemático
    # γ reflecte a memória fisiológica específica de cada desporto:
    #   Ski: alta sazonalidade → γ baixo (memória muito longa)
    #   Row: sessões longas, poucas/semana → γ médio
    #   Bike: treino frequente, intensidade variável → depende dos dados
    #   Run: adaptações neuromusculares + aeróbicas → depende dos dados

    MODS = ['Bike', 'Row', 'Ski', 'Run']
    mod_info = {}      # γ, R², n por modalidade
    mod_loads = {}     # série diária de carga por modalidade

    if 'type' not in df.columns:
        df['type'] = 'Bike'

    for mod in MODS:
        df_mod = df[df['type'] == mod].copy()
        if len(df_mod) < 5:
            mod_info[mod] = {'gamma_perf': 0.35, 'r2_perf': 0.0,
                             'gamma_mmp': 0.35, 'r2_mmp': 0.0,
                             'n_cp': 0, 'n_mmp': 0, 'n_sessions': 0}
            mod_loads[mod] = np.zeros(len(ld))
            continue

        # Carga diária desta modalidade (alinhada com ld)
        _ld_mod = (df_mod.groupby('Data')['_load']
                   .sum()
                   .reindex(_date_idx, fill_value=0).values)
        mod_loads[mod] = _ld_mod

        # Adiciona ao ld para visualização
        ld[f'load_{mod}'] = _ld_mod
        ld[f'CTL_{mod}']  = pd.Series(_ld_mod).ewm(span=42, adjust=False).mean().values
        ld[f'ATL_{mod}']  = pd.Series(_ld_mod).ewm(span=7,  adjust=False).mean().values

        # γ via icu_pm_cp desta modalidade
        gamma_m, r2_m, n_cp_m = 0.35, 0.0, 0
        if 'icu_pm_cp' in df_mod.columns and df_mod['icu_pm_cp'].notna().sum() >= 5:
            _cp_mod = (df_mod.groupby('Data')['icu_pm_cp']
                       .mean()
                       .reindex(_date_idx, fill_value=np.nan).values)
            gamma_m, r2_m, n_cp_m = fit_gamma_performance(
                _ld_mod, _cp_mod, lag=0, max_lag=MAX_LAG)

        # γ via MMP PR desta modalidade — Della Mattia Part II §1
        # Bike/Run: MMP20 (potência em ~20 min — paper §1, Imbach 2021)
        # Ski/Row:  MMP5  (esforços máximos curtos mais frequentes/realistas nestes desportos)
        # Apenas sessões is_pr=True: data exacta, esforço máximo confirmado
        gamma_mmp_m, r2_mmp_m, n_mmp_m = 0.35, 0.0, 0
        _mmp_col = 'mmp5_pr_w' if mod in ('Ski', 'Row') else 'mmp20_pr_w'
        if _mmp_col in df_mod.columns and df_mod[_mmp_col].notna().any():
            _mmp_s = (df_mod[df_mod[_mmp_col].notna()]
                      .groupby('Data')[_mmp_col].mean()
                      .reindex(_date_idx))
            if _mmp_s.notna().sum() >= 5:
                _gm, _rm, _nm = fit_gamma_performance(
                    _ld_mod, _mmp_s.values, lag=0, max_lag=MAX_LAG)
                if _rm > r2_m:
                    gamma_m, r2_m = _gm, _rm
                gamma_mmp_m, r2_mmp_m, n_mmp_m = _gm, _rm, _nm

        # CTLγ fraccionário para esta modalidade
        ld[f'CTLg_{mod}'] = ftlm_fractional(_ld_mod, gamma_m, max_lag=MAX_LAG)

        mod_info[mod] = {
            'gamma_perf': round(gamma_m,     3),
            'r2_perf':    round(r2_m,        3),
            'gamma_mmp':  round(gamma_mmp_m, 3),
            'r2_mmp':     round(r2_mmp_m,    3),
            'n_cp':       n_cp_m,
            'n_mmp':      n_mmp_m,
            'mmp_col':    _mmp_col,   # MMP5 or MMP20 depending on sport
            'n_sessions': len(df_mod),
        }

    # ── CTLγ overall (γ_perf = média ponderada por R² das modalidades) ────────
    # Usa o γ da modalidade com mais sessões como γ_perf global
    best_mod = max(MODS, key=lambda m: (mod_info[m]['r2_perf'], mod_info[m]['n_sessions']))
    gamma_perf  = mod_info[best_mod]['gamma_perf']
    r2_perf     = mod_info[best_mod]['r2_perf']

    ld['CTLg_perf'] = ftlm_fractional(load_overall, gamma_perf, max_lag=MAX_LAG)
    ld['CTLg_rec']  = ftlm_fractional(load_overall, gamma_rec,  max_lag=MAX_LAG)

    # FTLM activo = melhor R²
    best_gamma  = gamma_perf if r2_perf >= r2_rec else gamma_rec
    best_source = 'perf'     if r2_perf >= r2_rec else 'recovery'
    ld['FTLM']  = ld['CTLg_perf'] if best_source == 'perf' else ld['CTLg_rec']

    if hrv_trend_arr is not None:
        ld['HRV_trend'] = hrv_trend_arr

    info = {
        'fonte':        fonte,
        'gamma_perf':   round(gamma_perf, 3),
        'gamma_rec':    round(gamma_rec,  3),
        'gamma_best':   round(best_gamma, 3),
        'gamma_source': best_source,
        'r2_perf':      round(r2_perf, 3),
        'r2_rec':       round(r2_rec,  3),
        'best_mod':     best_mod,
        'mods':         mod_info,
    }
    return ld, info


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

        # Overload guard: se share_base muito baixo (<5%), ratios são instáveis
        # Modalidade com <5% do treino histórico → não pode estar em overload
        _share_min_ol = 0.05
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

        # ── Overload guards — densidade histórica + slope ──────────────────
        share_ratio   = share_jan / share_base if share_base > 0 else 0.0
        quality_ratio = q_jan / q_base         if q_base    > 0 else 0.0

        # Sessões nos últimos 90d desta modalidade
        _sess_90d = len(df[(df['type']==mod) & (df['Data'] >= ini_base)])

        # Slope FTLM na janela activa
        ftlm_jan_series = fm.loc[ini_jan:hoje, 'FTLM'].dropna()
        slope_pct_jan = 0.0
        fi = 0.0
        ff = 0.0
        if len(ftlm_jan_series) >= 3:
            fi = float(ftlm_jan_series.iloc[0])
            ff = float(ftlm_jan_series.iloc[-1])
            if fi > 0:
                slope_pct_jan = (ff - fi) / fi

        # Guard 1: <4 sessões/90d — sem dados para avaliar overload
        _ignorar_overload = _sess_90d < 4

        # Guard 2: FTLM pequeno — slope instável (início de época)
        _ftlm_p10 = float(fm['FTLM'].quantile(0.10)) if len(fm['FTLM'].dropna()) >= 10 else 0.0
        _ignorar_slope = (fi < max(_ftlm_p10, ff * 0.10)) if ff > 0 else True

        # Frequência esperada para ajustar A (complementar ao A_share)
        _freq_base    = _sess_90d / 90.0
        _esperado_7d  = _freq_base * janela_dias
        _sess_jan     = len(df[(df['type']==mod) & (df['Data'] >= ini_jan)])
        _A_freq_boost = 0.0
        if _esperado_7d > 0.5:  # só se há histórico suficiente
            _freq_ratio = _sess_jan / _esperado_7d
            if _freq_ratio < 0.5:     _A_freq_boost = 0.20  # treinou muito menos
            elif _freq_ratio < 0.75:  _A_freq_boost = 0.10
        # Aplicar boost em A (max 1.0)
        A_adj = min(1.0, A + _A_freq_boost * (1.0 - A))

        # Dias desde último Z3 (para trigger de estímulo forte)
        _dm_z3 = df[(df['type']==mod) & (df['rpe_n'] >= 7) & (df['Data'] >= ini_base)]
        _ultima_z3 = _dm_z3['Data'].max() if len(_dm_z3) > 0 else pd.NaT
        _dias_z3 = int((hoje - _ultima_z3).days) if pd.notna(_ultima_z3) else 999
        _gap_z3_tipico = float(max(1, _dm_z3['Data'].sort_values()
                                   .diff().dt.days.dropna().median())
                               if len(_dm_z3) >= 2 else 14.0)
        _limite_z3 = _gap_z3_tipico * 1.5
        _forcar_z3 = (_dias_z3 > _limite_z3) and (not _ignorar_overload)

        # Score overload com guards
        score_overload = 0
        if not _ignorar_overload:
            if share_ratio   > 1.4 and share_base > 0.05: score_overload += 1
            if quality_ratio > 1.4 and q_base > 0.02:     score_overload += 1
            if slope_pct_jan > 0.15 and not _ignorar_slope: score_overload += 1
        overload = (score_overload >= 2) and not _ignorar_overload

        # Tipo de overload (para modulação da intensidade)
        _overload_vol  = share_ratio   > 1.4 and share_base > 0.05
        _overload_qual = quality_ratio > 1.4 and q_base > 0.02
        _overload_agudo= slope_pct_jan > 0.15 and not _ignorar_slope

        if overload:
            if _overload_qual or _overload_agudo:
                need_int_prescr = min(100.0, need_int * 0.65)   # reduzir forte
            else:
                need_int_prescr = min(100.0, need_int * 0.90)   # só volume alto — redução leve
        else:
            need_int_prescr = need_int

        # C_reforçado + gap
        datas_mod_dbg  = df[df['type']==mod]['Data'].sort_values()
        ultima_dbg     = datas_mod_dbg.max() if len(datas_mod_dbg) > 0 else pd.NaT
        dias_sem       = int((hoje - ultima_dbg).days) if pd.notna(ultima_dbg) else 999
        gap_tipico_dbg = float(max(1, datas_mod_dbg[datas_mod_dbg >= ini_base]
                                   .diff().dt.days.dropna().median())
                               if len(datas_mod_dbg[datas_mod_dbg >= ini_base]) >= 2
                               else 7.0)
        gap_score    = min(1.0, dias_sem / (gap_tipico_dbg * 2))
        c_reforcado  = max(C, gap_score)
        gap_ratio    = min(dias_sem / gap_tipico_dbg, 3.0) if gap_tipico_dbg > 0 else 0.0

        # ── Guardar estado intermédio — prescrição calculada em 2ª passagem ─
        results[mod] = dict(
            need_score=need, prioridade=prio_base, overload=overload,
            overload_score=score_overload,
            overload_tipo=('qualidade/agudo' if (_overload_qual or _overload_agudo)
                           else 'volume' if _overload_vol else ''),
            need_vol=round(need_vol, 1),
            need_int=round(need_int, 1),
            need_int_prescr=round(need_int_prescr, 1),
            prescricao='',    # preenchida em 2ª passagem
            dias_sem=dias_sem, gap_score=round(gap_score, 2),
            gap_ratio=gap_ratio, gap_z3=_dias_z3, forcar_z3=_forcar_z3,
            gap_z3_limite=round(_limite_z3, 1),
            c_reforcado=round(c_reforcado, 3),
            sess_90d=_sess_90d, ignorar_overload=_ignorar_overload,
            share_actual=share_jan, share_hist=share_base,
            quality_actual=q_jan, quality_hist=q_base,
            load_jan=load_jan, load_tipico=load_tipico,
            ftlm_slope_pct=slope_pct, ftlm_slope_jan=slope_pct_jan,
            A=A_adj, B=B, C=C, D=D, E=E, floor_peso=floor_peso)

        debug_rows.append({
            'Modalidade':            mod,
            'Need Score':            round(need, 1),
            'Need Volume':           round(need_vol, 1),
            'Need Intensity':        round(need_int, 1),
            'Need Int (prescrição)': round(need_int_prescr, 1),
            'Prescrição':            '(2ª passagem)',
            'dias_sem_sessao':       dias_sem,
            'gap_score':             round(gap_score, 2),
            'C_reforçado':           round(c_reforcado, 3),
            'Prioridade base':       prio_base,
            'Overload score':        score_overload,
            'Overload':              '⚠️ SIM' if overload else 'não',
            'Overload tipo':         results[mod]['overload_tipo'],
            'Sess 90d':              _sess_90d,
            'Ignorar OL':            _ignorar_overload,
            'Dias Z3':               _dias_z3,
            'Forçar Z3':             _forcar_z3,
            'A_freq_boost':          round(_A_freq_boost, 2),
            'A Share actual%':       round(share_jan  * 100, 1),
            'A Share hist90d%':      round(share_base * 100, 1),
            'A Deficit%':            round(A_raw * 100, 1),
            'A Floor peso':          round(floor_peso, 2),
            'A contribuição':        round(A_adj * 100 * 0.25, 1),
            'B Quality actual%':     round(q_jan  * 100, 1),
            'B Quality hist90d%':    round(q_base * 100, 1),
            'B Deficit%':            round(B_raw * 100, 1),
            'B contribuição':        round(B * 100 * 0.25, 1),
            'C Load janela':         round(load_jan, 1),
            'C Load típico':         round(load_tipico, 1),
            'C Deficit%':            round(C_raw * 100, 1),
            'C contribuição':        round(C * 100 * 0.20, 1),
            'D FTLM ini':            round(ftlm_ini if len(ftlm_base_series)>=10 else 0, 1),
            'D FTLM fim':            round(ftlm_fim if len(ftlm_base_series)>=10 else 0, 1),
            'D slope_pct%':          round(slope_pct * 100, 1),
            'D slope_jan%':          round(slope_pct_jan * 100, 1),
            'D contribuição':        round(D * 100 * 0.20, 1),
            'E Fator VQ':            fator_e,
            'E contribuição':        round(E * 100 * 0.10, 1),
        })

    # ══════════════════════════════════════════════════════
    # 2ª PASSAGEM — rank relativo + prescrição contextual
    # ══════════════════════════════════════════════════════
    # Calcular need_int_prescr de todos os mods para rank
    _all_scores = [d['need_int_prescr'] for d in results.values()]
    _std_scores  = float(np.std(_all_scores)) if len(_all_scores) > 1 else 0.0

    for mod, d in results.items():
        ni  = d['need_int_prescr']
        nv  = d['need_vol']
        ol  = d['overload']
        gr  = d['gap_ratio']
        fz3 = d['forcar_z3']
        dz3 = d['gap_z3']
        lz3 = d['gap_z3_limite']

        # Rank percentil dentro das modalidades (0-100)
        rank_int = float(sum(ni > s for s in _all_scores)) / max(len(_all_scores)-1, 1) * 100

        # Guard baixa dispersão: se todos scores semelhantes (std < 5)
        # → usar padrão histórico em vez de prescrever intenso a todos
        _baixa_dispersao = _std_scores < 5.0

        # Aviso Z3
        _aviso_z3 = (f" ⚡ Último Z3 há {dz3}d (limite {lz3:.0f}d)"
                     if fz3 else "")

        if ol:
            if d['overload_tipo'] == 'volume':
                prescricao = "🟡 Overload de volume — intensidade leve/moderada"
            else:
                prescricao = "🟢 Overload — reduzir intensidade" + _aviso_z3
        elif fz3 and not _baixa_dispersao:
            # Z3 em falta há muito tempo E há diferenciação entre mods
            prescricao = "🟠 Estímulo Z3 urgente" + _aviso_z3
        elif _baixa_dispersao:
            # Todos mods com need semelhante → usar histórico
            if nv >= 40:
                prescricao = "🔵 Sessão de base/volume (sem diferenciação clara)"
            else:
                prescricao = "⚪ Manutenção (estado equilibrado)"
        elif rank_int >= 75:
            # Top 25% de intensidade → sessão intensa
            if nv >= 50:
                prescricao = "🔴 Sessão completa (volume + intensidade)" + _aviso_z3
            elif gr <= 1.0:
                prescricao = "🟠 Sessão intensa/curta (fresco — défice qualidade)" + _aviso_z3
            elif gr <= 2.0:
                prescricao = "🟡 Sessão mista vol+int (reentrée gradual)" + _aviso_z3
            else:
                prescricao = "🔵 Sessão de base/reentrée (gap longo)" + _aviso_z3
        elif rank_int >= 50:
            # Top 50% → sessão moderada
            if nv >= 50:
                prescricao = "🔵 Sessão de volume/base + intensidade moderada"
            else:
                prescricao = "🟡 Sessão moderada"
        elif nv >= 50:
            prescricao = "🔵 Sessão de volume/base (défice de carga)"
        else:
            prescricao = "⚪ Manutenção ou descanso"

        results[mod]['prescricao'] = prescricao
        results[mod]['rank_int']   = round(rank_int, 0)
        results[mod]['std_scores'] = round(_std_scores, 1)

        # Actualizar debug
        for row in debug_rows:
            if row['Modalidade'] == mod:
                row['Prescrição'] = prescricao
                row['Rank int%']  = round(rank_int, 0)
                row['Std scores'] = round(_std_scores, 1)

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
