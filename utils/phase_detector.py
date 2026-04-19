"""
phase_detector.py — ATHELTICA
Detecção de fases de treino por modalidade e global.

Sinais:
  CTLγ        — carga acumulada fraccionária (nível de fitness)
  ΔCTLγ       — slope 14d (dinâmica: está a crescer ou descer?)
  HRV_trend   — estado autonómico (intercept + slope normalizado, janela 7d)
  WEED_z      — stress subjectivo rolling 28d (stress + soreness + fatiga)

Thresholds: percentil rolante 60d — adaptativos ao histórico do atleta.

Fases (prioridade decrescente):
  OVERREACH   — HRV < p10 AND WEED > p90 AND dCTL > p50  (carga alta confirma)
  FATIGUE     — dCTL > p50 AND HRV < p20
  BUILD       — dCTL > p70 AND HRV > p30  (separação real das outras fases)
  PEAK        — p30 ≤ dCTL ≤ p70 AND HRV > p60  (carga estável + HRV alto)
  RECOVERY    — dCTL < p30 AND HRV > p50
  TRANSITION  — nenhuma das anteriores

Suavização: rolling mode 3 dias (evita flicker entre fases adjacentes).
Fase global: weighted_mode das fases modais, ponderado por CTLγ_mod actual.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ── Phase definitions ─────────────────────────────────────────────────────────
PHASE_LABELS = {
    'BUILD':      {'label': '🔨 Build',     'color': '#2980b9',
                   'desc': 'Acumulação de carga — fitness a crescer'},
    'FATIGUE':    {'label': '🔴 Fatigue',    'color': '#e74c3c',
                   'desc': 'Carga alta com recuperação autonómica comprometida'},
    'OVERREACH':  {'label': '⚠️ Overreach',  'color': '#8e44ad',
                   'desc': 'HRV muito baixo + stress elevado + carga alta'},
    'RECOVERY':   {'label': '🟢 Recovery',   'color': '#27ae60',
                   'desc': 'Carga a reduzir, sistema autonómico a recuperar'},
    'PEAK':       {'label': '⭐ Peak',        'color': '#f39c12',
                   'desc': 'Carga estável, HRV alto — forma potencial'},
    'TRANSITION': {'label': '⬜ Transition', 'color': '#95a5a6',
                   'desc': 'Estado intermédio — sem padrão claro'},
}

# hex → rgba for Plotly (opacity as 0-1 float)
def _hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ── Signal helpers ────────────────────────────────────────────────────────────

def _rolling_slope(series: np.ndarray, window: int = 14) -> np.ndarray:
    """Rolling linear slope over `window` days."""
    n   = len(series)
    out = np.full(n, np.nan)
    x   = np.arange(window, dtype=np.float64)
    for t in range(window - 1, n):
        y    = series[t - window + 1: t + 1]
        mask = np.isfinite(y)
        if mask.sum() < max(4, window // 2):
            continue
        sl, *_ = linregress(x[mask], y[mask])
        out[t] = sl
    return out


def _rolling_pct(series: np.ndarray, q: float, window: int = 60) -> np.ndarray:
    """Rolling quantile (e.g. q=0.20 → p20)."""
    return (pd.Series(series)
              .rolling(window, min_periods=max(10, window // 4))
              .quantile(q)
              .values)


def _zscore_rolling(series: np.ndarray, window: int = 60) -> np.ndarray:
    """Relative z-score: (x - rolling_mean) / rolling_std."""
    s  = pd.Series(series)
    mu = s.rolling(window, min_periods=max(10, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 4)).std()
    return ((s - mu) / sd.replace(0, np.nan)).values


def _rolling_mode_3(phases: np.ndarray) -> np.ndarray:
    """
    Smooth phase array: replace each day with majority vote of [t-2, t-1, t].
    Eliminates single-day flicker between adjacent phases.
    """
    n   = len(phases)
    out = phases.copy()
    for t in range(2, n):
        window = [phases[t-2], phases[t-1], phases[t]]
        # majority: if any value appears 2+ times use it, else keep current
        from collections import Counter
        counts = Counter(window)
        top, freq = counts.most_common(1)[0]
        if freq >= 2:
            out[t] = top
    return out


# ── Phase detection ───────────────────────────────────────────────────────────

def detect_phases(ld: pd.DataFrame,
                  mod: str = 'overall',
                  slope_window: int = 14,
                  pct_window:   int = 60) -> pd.DataFrame:
    """
    Detect training phases for one modality (or overall).

    Returns DataFrame: Data, CTLg, dCTLg_14d, HRV_rel, WEED_z,
                       CTLg_z, dCTLg_z, HRV_z,
                       phase, phase_smooth, phase_label, phase_color,
                       fase_id, dias_na_fase, fase_shift
    """
    out = pd.DataFrame({'Data': ld['Data'].values})

    # ── CTLγ series ───────────────────────────────────────────────────────────
    ctlg_col = 'CTLg_perf' if mod == 'overall' else f'CTLg_{mod}'
    if ctlg_col not in ld.columns or ld[ctlg_col].notna().sum() < 30:
        ctlg_col = 'CTL'
    ctlg = pd.to_numeric(ld[ctlg_col], errors='coerce').values

    # ── ΔCTLγ — rolling slope 14d ─────────────────────────────────────────────
    dctlg = _rolling_slope(ctlg, window=slope_window)

    # ── HRV_trend — relative to 60d baseline ─────────────────────────────────
    if 'HRV_trend' in ld.columns and ld['HRV_trend'].notna().sum() >= 20:
        hrv_raw = pd.to_numeric(ld['HRV_trend'], errors='coerce').values
    else:
        hrv_raw = np.full(len(ld), np.nan)
    hrv_rel = _zscore_rolling(hrv_raw, window=pct_window)

    # ── WEED_z ────────────────────────────────────────────────────────────────
    if 'WEED_z' in ld.columns and ld['WEED_z'].notna().sum() >= 20:
        weed = pd.to_numeric(ld['WEED_z'], errors='coerce').values
    else:
        weed = np.full(len(ld), np.nan)

    # ── Z-score all for comparability ─────────────────────────────────────────
    def _zs(arr):
        mu, sd = np.nanmean(arr), np.nanstd(arr)
        return (arr - mu) / sd if sd > 1e-9 else arr - mu

    ctl_z  = _zs(ctlg)
    dctl_z = _zs(dctlg)
    hrv_z  = hrv_rel    # already relative z-score

    # ── Rolling percentile thresholds (adaptive) ──────────────────────────────
    dctl_p70 = _rolling_pct(dctlg, 0.70, pct_window)
    dctl_p50 = _rolling_pct(dctlg, 0.50, pct_window)
    dctl_p30 = _rolling_pct(dctlg, 0.30, pct_window)

    hrv_p60  = _rolling_pct(hrv_z, 0.60, pct_window)
    hrv_p50  = _rolling_pct(hrv_z, 0.50, pct_window)
    hrv_p30  = _rolling_pct(hrv_z, 0.30, pct_window)
    hrv_p20  = _rolling_pct(hrv_z, 0.20, pct_window)
    hrv_p10  = _rolling_pct(hrv_z, 0.10, pct_window)

    weed_p90 = _rolling_pct(_zs(weed), 0.90, pct_window)
    weed_z_s = _zs(weed)

    # ── Phase classification (priority order) ─────────────────────────────────
    n      = len(ld)
    phases = np.array(['TRANSITION'] * n, dtype=object)

    for t in range(n):
        dc = dctlg[t]
        hv = hrv_z[t]
        wd = weed_z_s[t]

        if not np.isfinite(dc):
            continue

        d70 = dctl_p70[t]; d50 = dctl_p50[t]; d30 = dctl_p30[t]
        h60 = hrv_p60[t];  h50 = hrv_p50[t]
        h30 = hrv_p30[t];  h20 = hrv_p20[t];  h10 = hrv_p10[t]
        w90 = weed_p90[t]

        hrv_ok   = np.isfinite(hv) and np.isfinite(h20)
        weed_ok  = np.isfinite(wd) and np.isfinite(w90)
        d_ok     = np.isfinite(d70) and np.isfinite(d30)

        if not d_ok:
            continue

        # OVERREACH: HRV very low + high stress + carga alta (confirma overreach real)
        # dCTL > p50 distingue de "baixo HRV por sono/stress sem carga"
        if hrv_ok and weed_ok and hv < h10 and wd > w90 and dc > d50:
            phases[t] = 'OVERREACH'

        # FATIGUE: carga crescendo + HRV comprometido
        elif hrv_ok and np.isfinite(d50) and dc > d50 and hv < h20:
            phases[t] = 'FATIGUE'

        # BUILD: carga crescendo forte + HRV aceitável (> p30, não só > p20)
        # p30 cria separação real: atleta consegue acumular e recuperar
        elif hrv_ok and dc > d70 and hv > h30:
            phases[t] = 'BUILD'

        # PEAK: carga estável (não crescendo nem caindo muito) + HRV alto
        # p60 confirma HRV acima da mediana — forma real
        elif hrv_ok and np.isfinite(h60) and np.isfinite(d30) and \
             d30 <= dc <= d70 and hv > h60:
            phases[t] = 'PEAK'

        # RECOVERY: carga a cair + HRV a recuperar
        elif hrv_ok and dc < d30 and hv > h50:
            phases[t] = 'RECOVERY'

        # else: TRANSITION

    # ── Smooth phases (rolling mode 3d) ───────────────────────────────────────
    phases_smooth = _rolling_mode_3(phases)

    # ── Phase duration tracking ───────────────────────────────────────────────
    phase_series = pd.Series(phases_smooth)
    fase_shift   = (phase_series != phase_series.shift(1)).values
    fase_id      = fase_shift.cumsum()
    dias_na_fase = pd.Series(fase_id).groupby(fase_id).cumcount().values

    # ── Build output ──────────────────────────────────────────────────────────
    out['CTLg']         = np.round(ctlg,        3)
    out['dCTLg_14d']    = np.round(dctlg,       5)
    out['HRV_rel']      = np.round(hrv_rel,      3)
    out['WEED_z']       = np.round(weed,         3)
    out['CTLg_z']       = np.round(ctl_z,        3)
    out['dCTLg_z']      = np.round(dctl_z,       3)
    out['HRV_z']        = np.round(hrv_z,        3)
    out['phase']        = phases          # raw (before smoothing)
    out['phase_smooth'] = phases_smooth   # smoothed (use this for display)
    out['fase_shift']   = fase_shift.astype(bool)
    out['fase_id']      = fase_id
    out['dias_na_fase'] = dias_na_fase
    out['phase_label']  = [PHASE_LABELS[p]['label'] for p in phases_smooth]
    out['phase_color']  = [PHASE_LABELS[p]['color'] for p in phases_smooth]
    out['phase_desc']   = [PHASE_LABELS[p]['desc']  for p in phases_smooth]

    return out


def detect_all_phases(ld: pd.DataFrame,
                      slope_window: int = 14,
                      pct_window:   int = 60) -> dict:
    """
    Run phase detection for all modalities + weighted global.

    Global phase = weighted_mode of modal phases, weights = CTLγ_mod current.
    Only modalities with >= 30 data points are included.

    Returns dict: {'overall': df, 'Bike': df, ...}
    The 'overall' key uses CTLγ_perf (all-sport combined).
    An extra 'global_weighted' key shows the CTLγ-weighted consensus.
    """
    results = {}

    # Per-modality
    mod_phases    = {}  # modal phase for today
    mod_ctlg_vals = {}  # modal CTLγ for weighting

    for mod in ['Bike', 'Row', 'Ski', 'Run']:
        ctlg_col = f'CTLg_{mod}'
        if ctlg_col in ld.columns and ld[ctlg_col].notna().sum() >= 30:
            df_mod = detect_phases(ld, mod, slope_window, pct_window)
            results[mod]        = df_mod
            mod_phases[mod]     = df_mod['phase_smooth'].iloc[-1]
            mod_ctlg_vals[mod]  = float(ld[ctlg_col].dropna().iloc[-1])

    # Overall (CTLγ_perf all sports combined)
    df_overall       = detect_phases(ld, 'overall', slope_window, pct_window)
    results['overall'] = df_overall

    # Weighted global: mode of modal phases weighted by CTLγ current value
    if mod_phases:
        total_w = sum(mod_ctlg_vals.values())
        if total_w > 0:
            phase_weights: dict = {}
            for mod, phase in mod_phases.items():
                w = mod_ctlg_vals.get(mod, 0.0)
                phase_weights[phase] = phase_weights.get(phase, 0.0) + w
            global_phase = max(phase_weights, key=phase_weights.get)
        else:
            global_phase = df_overall['phase_smooth'].iloc[-1]

        # Build a synthetic global series using the overall df but weighted phase
        # (just for the current day — the series itself stays as overall)
        results['_global_phase_today'] = global_phase
        results['_modal_weights']      = {
            m: round(mod_ctlg_vals.get(m, 0) / max(sum(mod_ctlg_vals.values()), 1e-9), 3)
            for m in mod_phases
        }

    return results


def phase_summary(phase_df: pd.DataFrame, last_n: int = 30) -> dict:
    """
    Summary dict for one modality's phase DataFrame.
    Uses phase_smooth (not raw phase).
    """
    recent  = phase_df.tail(last_n)
    current = phase_df.iloc[-1]

    cp      = current['phase_smooth']
    cl      = current['phase_label']
    cc      = current['phase_color']
    cd      = current['phase_desc']

    dist    = recent['phase_smooth'].value_counts().to_dict()
    streak  = int(current['dias_na_fase']) + 1  # cumcount is 0-indexed
    stable  = streak >= 5

    transitions = int(recent['fase_shift'].sum())

    return {
        'current_phase':    cp,
        'current_label':    cl,
        'current_color':    cc,
        'current_desc':     cd,
        'streak_days':      streak,
        'stable':           stable,
        'transitions_30d':  transitions,
        'distribution_30d': dist,
        'current_ctlg':     float(current['CTLg'])
                            if np.isfinite(current['CTLg']) else 0.0,
        'current_dctlg':    float(current['dCTLg_14d'])
                            if np.isfinite(current['dCTLg_14d']) else 0.0,
        'current_hrv_z':    float(current['HRV_z'])
                            if np.isfinite(current['HRV_z']) else None,
        'current_weed_z':   float(current['WEED_z'])
                            if np.isfinite(current['WEED_z']) else None,
    }
