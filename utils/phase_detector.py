"""
phase_detector.py — ATHELTICA
Detecção de fases de treino por modalidade e global.

Baseado em:
- CTLγ (carga acumulada fraccionária)
- ΔCTLγ (slope/tendência 14d — dinâmica de carga)
- HRV_trend (estado autonómico — intercept + slope normalizado)
- WEED_z (stress subjectivo — média z-score 28d)

Thresholds por percentil rolante 60d — não valores fixos:
    BUILD:      dCTL > p70  AND  HRV > p20
    FATIGUE:    dCTL > p50  AND  HRV < p20
    OVERREACH:  HRV < p10   AND  WEED > p90
    RECOVERY:   dCTL < p30  AND  HRV > p50
    PEAK:       dCTL ∈ [p30,p70]  AND  HRV > p50  AND  dCTL declining
    TRANSITION: não se enquadra em nenhuma fase acima
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ── Phase labels and colours ──────────────────────────────────────────────────
PHASE_LABELS = {
    'BUILD':      {'label': '🔨 Build',      'color': '#2980b9', 'desc': 'Acumulação de carga — fitness a crescer'},
    'FATIGUE':    {'label': '🔴 Fatigue',     'color': '#e74c3c', 'desc': 'Carga alta com recuperação autonómica comprometida'},
    'OVERREACH':  {'label': '⚠️ Overreach',   'color': '#8e44ad', 'desc': 'HRV muito baixo + stress subjectivo elevado'},
    'RECOVERY':   {'label': '🟢 Recovery',    'color': '#27ae60', 'desc': 'Carga a reduzir, sistema autonómico a recuperar'},
    'PEAK':       {'label': '⭐ Peak',         'color': '#f39c12', 'desc': 'Carga estável, HRV alto — forma potencial'},
    'TRANSITION': {'label': '⬜ Transition',  'color': '#95a5a6', 'desc': 'Estado intermédio — sem padrão claro'},
}


# ── Core computation ──────────────────────────────────────────────────────────

def _rolling_slope(series, window=14):
    """Rolling linear slope over `window` days using linregress."""
    arr = np.array(series, dtype=np.float64)
    n   = len(arr)
    out = np.full(n, np.nan)
    x   = np.arange(window, dtype=np.float64)
    for t in range(window - 1, n):
        y = arr[t - window + 1: t + 1]
        mask = np.isfinite(y)
        if mask.sum() < max(4, window // 2):
            continue
        sl, *_ = linregress(x[mask], y[mask])
        out[t] = sl
    return out


def _rolling_pct(series, q, window=60):
    """Rolling quantile (e.g. q=0.20 for p20)."""
    s = pd.Series(series)
    return s.rolling(window, min_periods=max(10, window // 4)).quantile(q).values


def _zscore_rolling(series, window=60):
    """Relative z-score: (x - rolling_mean) / rolling_std."""
    s  = pd.Series(series)
    mu = s.rolling(window, min_periods=max(10, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 4)).std()
    return ((s - mu) / sd.replace(0, np.nan)).values


def detect_phases(ld: pd.DataFrame,
                  mod: str = 'overall',
                  slope_window: int = 14,
                  pct_window:   int = 60) -> pd.DataFrame:
    """
    Detect training phases for one modality (or overall).

    Parameters
    ----------
    ld          : DataFrame from calcular_series_carga (must have Data column)
    mod         : 'overall' or one of 'Bike','Row','Ski','Run'
    slope_window: days for dCTL slope calculation (default 14)
    pct_window  : days for rolling percentile thresholds (default 60)

    Returns
    -------
    DataFrame with columns:
        Data, CTL, dCTL, HRV_rel, WEED_z,
        CTL_z, dCTL_z, HRV_z, WEED_z_std,
        phase, phase_label, phase_color
    """
    out = pd.DataFrame({'Data': ld['Data']})

    # ── 1. Select CTLγ series ─────────────────────────────────────────────────
    ctlg_col = 'CTLg_perf' if mod == 'overall' else f'CTLg_{mod}'
    if ctlg_col not in ld.columns or ld[ctlg_col].notna().sum() < 30:
        # Fallback to classic CTL
        ctlg_col = 'CTL'
    ctlg = pd.to_numeric(ld[ctlg_col], errors='coerce').values

    # ── 2. ΔCTLγ — rolling slope (dinâmica de carga) ─────────────────────────
    dctlg = _rolling_slope(ctlg, window=slope_window)

    # ── 3. HRV_trend — relative to 60d baseline ──────────────────────────────
    if 'HRV_trend' in ld.columns and ld['HRV_trend'].notna().sum() >= 20:
        hrv_raw = pd.to_numeric(ld['HRV_trend'], errors='coerce').values
    else:
        hrv_raw = np.full(len(ld), np.nan)
    # Relative: (HRV - rolling_mean_60d) / rolling_std_60d
    hrv_rel = _zscore_rolling(hrv_raw, window=pct_window)

    # ── 4. WEED_z ─────────────────────────────────────────────────────────────
    if 'WEED_z' in ld.columns and ld['WEED_z'].notna().sum() >= 20:
        weed = pd.to_numeric(ld['WEED_z'], errors='coerce').values
    else:
        weed = np.full(len(ld), np.nan)

    # ── 5. Standardise all signals (comparable scale) ─────────────────────────
    def _zs(arr):
        mu, sd = np.nanmean(arr), np.nanstd(arr)
        return (arr - mu) / sd if sd > 1e-9 else arr - mu

    ctl_z   = _zs(ctlg)
    dctl_z  = _zs(dctlg)
    hrv_z   = hrv_rel    # already relative z-score
    weed_z  = _zs(weed)  # additional z-score for threshold comparison

    # ── 6. Rolling percentile thresholds ─────────────────────────────────────
    dctl_p70 = _rolling_pct(dctlg, 0.70, pct_window)
    dctl_p50 = _rolling_pct(dctlg, 0.50, pct_window)
    dctl_p30 = _rolling_pct(dctlg, 0.30, pct_window)
    hrv_p80  = _rolling_pct(hrv_z,  0.80, pct_window)
    hrv_p50  = _rolling_pct(hrv_z,  0.50, pct_window)
    hrv_p20  = _rolling_pct(hrv_z,  0.20, pct_window)
    hrv_p10  = _rolling_pct(hrv_z,  0.10, pct_window)
    weed_p90 = _rolling_pct(weed_z, 0.90, pct_window)

    # ── 7. Phase classification ───────────────────────────────────────────────
    n = len(ld)
    phases = np.array(['TRANSITION'] * n, dtype=object)

    for t in range(n):
        dc   = dctlg[t]
        hv   = hrv_z[t]
        wd   = weed_z[t]

        # Need valid dCTL at minimum
        if not np.isfinite(dc):
            continue

        d70 = dctl_p70[t]; d50 = dctl_p50[t]; d30 = dctl_p30[t]
        h80 = hrv_p80[t];  h50 = hrv_p50[t];  h20 = hrv_p20[t]
        h10 = hrv_p10[t];  w90 = weed_p90[t]

        # Check each threshold is valid
        hrv_ok  = np.isfinite(hv) and np.isfinite(h20) and np.isfinite(h10)
        weed_ok = np.isfinite(wd) and np.isfinite(w90)
        d_ok    = np.isfinite(d70) and np.isfinite(d50) and np.isfinite(d30)

        if not d_ok:
            continue

        # Priority order: OVERREACH > FATIGUE > BUILD > RECOVERY > PEAK > TRANSITION
        if hrv_ok and weed_ok and hv < h10 and wd > w90:
            phases[t] = 'OVERREACH'
        elif hrv_ok and dc > d50 and hv < h20:
            phases[t] = 'FATIGUE'
        elif hrv_ok and dc > d70 and hv > h20:
            phases[t] = 'BUILD'
        elif hrv_ok and dc < d30 and hv > h50:
            phases[t] = 'RECOVERY'
        elif hrv_ok and d30 <= dc <= d70 and hv > h50:
            phases[t] = 'PEAK'
        # else: TRANSITION (default)

    # ── 8. Build output DataFrame ─────────────────────────────────────────────
    phase_labels = np.array([PHASE_LABELS[p]['label'] for p in phases], dtype=object)
    phase_colors = np.array([PHASE_LABELS[p]['color'] for p in phases], dtype=object)
    phase_descs  = np.array([PHASE_LABELS[p]['desc']  for p in phases], dtype=object)

    out['CTLg']        = np.round(ctlg,   3)
    out['dCTLg_14d']   = np.round(dctlg,  5)
    out['HRV_rel']     = np.round(hrv_rel, 3)
    out['WEED_z']      = np.round(weed,    3)
    out['CTLg_z']      = np.round(ctl_z,   3)
    out['dCTLg_z']     = np.round(dctl_z,  3)
    out['HRV_z']       = np.round(hrv_z,   3)
    out['phase']       = phases
    out['phase_label'] = phase_labels
    out['phase_color'] = phase_colors
    out['phase_desc']  = phase_descs

    return out


def detect_all_phases(ld: pd.DataFrame,
                      slope_window: int = 14,
                      pct_window:   int = 60) -> dict:
    """
    Run phase detection for all modalities + overall.
    Returns dict: {'overall': df, 'Bike': df, 'Row': df, 'Ski': df, 'Run': df}
    Only includes modalities with sufficient data.
    """
    results = {}

    # Overall
    results['overall'] = detect_phases(ld, 'overall', slope_window, pct_window)

    # Per modality
    for mod in ['Bike', 'Row', 'Ski', 'Run']:
        ctlg_col = f'CTLg_{mod}'
        if ctlg_col in ld.columns and ld[ctlg_col].notna().sum() >= 30:
            results[mod] = detect_phases(ld, mod, slope_window, pct_window)

    return results


def phase_summary(phase_df: pd.DataFrame, last_n: int = 30) -> dict:
    """
    Summary of recent phase distribution and current phase.
    Returns dict with current phase, dominant phase in last_n days, transitions.
    """
    recent = phase_df.tail(last_n)
    current_phase = phase_df['phase'].iloc[-1]
    current_label = phase_df['phase_label'].iloc[-1]
    current_color = phase_df['phase_color'].iloc[-1]
    current_desc  = phase_df['phase_desc'].iloc[-1]

    # Phase distribution in last_n days
    dist = recent['phase'].value_counts().to_dict()

    # Count phase transitions in last_n days
    phases_seq = recent['phase'].values
    transitions = int(sum(phases_seq[i] != phases_seq[i-1]
                         for i in range(1, len(phases_seq))))

    # Days in current phase streak
    streak = 1
    for i in range(len(phase_df) - 2, -1, -1):
        if phase_df['phase'].iloc[i] == current_phase:
            streak += 1
        else:
            break

    # Trend: is current phase stable or just started?
    stable = streak >= 5

    return {
        'current_phase': current_phase,
        'current_label': current_label,
        'current_color': current_color,
        'current_desc':  current_desc,
        'streak_days':   streak,
        'stable':        stable,
        'transitions_30d': transitions,
        'distribution_30d': dist,
        'current_ctlg':  float(phase_df['CTLg'].iloc[-1]),
        'current_dctlg': float(phase_df['dCTLg_14d'].iloc[-1])
                         if np.isfinite(phase_df['dCTLg_14d'].iloc[-1]) else 0.0,
        'current_hrv_z': float(phase_df['HRV_z'].iloc[-1])
                         if np.isfinite(phase_df['HRV_z'].iloc[-1]) else None,
        'current_weed_z': float(phase_df['WEED_z'].iloc[-1])
                          if np.isfinite(phase_df['WEED_z'].iloc[-1]) else None,
    }
