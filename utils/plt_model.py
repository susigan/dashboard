# ══════════════════════════════════════════════════════════════════════════════
# utils/plt_model.py — ATHELTICA
# Pure-Load Tensor (PLT) + Impulse-Response (IR)
#
# Implementação fiel a:
#   Gabriel Della Mattia / enydog — github.com/enydog/PLT (implementation.py, 2024)
#   "Pure-Load Tensor (PLT)" — Part I do framework (FMT é a variante futura)
#
# DIFERENÇA vs FMT:
#   FMT  → tensor 5×5, κ=trace(cov(Δx)), detecção de fadiga oculta (curvatura)
#   PLT  → vector de carga por sessão → impulso diário u_plt → IR fitness-fatigue
#          → P_hat (performance latente prevista)
#   São pipelines DIFERENTES e complementares. O PLT NÃO substitui o FMT nem o NLSS.
#
# ADAPTAÇÃO ÀS FONTES DE DADOS ATHELTICA (confirmadas no repo):
#   TSS         ← icu_training_load        (load real do Intervals.icu)
#   Ekg         ← (icu_joules/1000) / Peso  ← kJ TOTAL (sem desconto warm-up);
#                                              Peso vem do WELLNESS, não icu_weight
#   decoupling  ← hq_4 − hq_1               (DIFERENÇA bpm, fiel ao PLT; já no data.py)
#   hrv_drop    ← baseline_7d(hrv) − hrv    (morning HRV do wellness; proxy do pre-post)
#   λ_HRV       ← OMITIDO                    (não há 2+ medições HRV/dia no ATHELTICA)
#
# kJ: icu_joules/1000 = energia mecânica TOTAL (ex: 681613 J → 681.6 kJ).
#     KJ TOTAL sempre, sem warm-up. NÃO usa z1/z2/z3_kj (que é por zona de potência).
#
# ALVO DO FIT IR: MMP real (mmp20_w Bike/Run, mmp5_w Row/Ski) — NUNCA eFTP.
#   Razão: icu_eftp é estimador ruidoso (o próprio tab_eftp trata-o com banda ±MDC).
#   Ajustar kf/kg a eFTP seria ajustar a ruído. MMP = potência realmente produzida.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd


# ── β weights (idênticos ao implementation.py, subset disponível) ─────────────
# Original PLT rz_cols/beta tem 12 termos (inclui Q1..Q4 individuais, HRV pre/post,
# λ_HRV). Aqui usamos o subset que as nossas fontes suportam, mantendo os mesmos
# pesos relativos do paper para os termos que existem:
#   TSS=1.0, Ekg=0.8, decoupling=0.8, hrv_drop=0.8
_PLT_BETA = {
    'tss':        1.0,
    'ekg':        0.8,
    'decoupling': 0.8,
    'hrv_drop':   0.8,
}

# ── Parâmetros IR fixos do paper (default quando não há MMP para fit) ─────────
_IR_DEFAULT = dict(tau_f=42.0, tau_g=7.0, k_f=1.0, k_g=2.0, P0=0.0)

# Grelha de fit (igual ao gridfit_ir do implementation.py)
_TAUS_F = (20, 30, 42, 60)
_TAUS_G = (3, 5, 7, 10, 14)
_KFS    = (0.3, 0.5, 1.0, 1.5)
_KGS    = (0.8, 1.0, 1.5, 2.0, 3.0)


# ── Utilities (fiéis ao implementation.py) ────────────────────────────────────

def _softplus(x: np.ndarray) -> np.ndarray:
    """log(1+e^x), numericamente estável."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -30, 30))))


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score via mediana/IQR (igual ao PLT). 0 se IQR≈0."""
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).sum() < 3:
        return np.zeros_like(x)
    med = np.nanmedian(x)
    q1, q3 = np.nanpercentile(x[np.isfinite(x)], [25, 75])
    iqr = q3 - q1
    if iqr <= 1e-8:
        return np.zeros_like(x)
    return np.nan_to_num((x - med) / iqr, nan=0.0)


# ── IR fitness-fatigue (Banister recursivo, fiel ao implementation.py) ────────

def _simulate_ir(u_f: np.ndarray, u_g: np.ndarray,
                 tau_f: float, tau_g: float, k_f: float, k_g: float,
                 P0: float):
    """F[i]=ρf·F[i-1]+u[i-1]; G idem; P_hat=P0+kf·F−kg·G."""
    n = len(u_f)
    F = np.zeros(n)
    G = np.zeros(n)
    rho_f = np.exp(-1.0 / float(tau_f))
    rho_g = np.exp(-1.0 / float(tau_g))
    for i in range(1, n):
        F[i] = rho_f * F[i - 1] + u_f[i - 1]
        G[i] = rho_g * G[i - 1] + u_g[i - 1]
    P_hat = float(P0) + float(k_f) * F - float(k_g) * G
    return F, G, P_hat


def _gridfit_ir(u: np.ndarray, y: np.ndarray) -> dict:
    """
    Ajusta (τf,τg,kf,kg,P0) que minimizam MSE entre P_hat e o alvo y (MMP real).
    Replica gridfit_ir() do implementation.py. y pode ter NaN (só fita nos válidos).
    Devolve dict de parâmetros; se <6 pontos válidos, devolve os defaults do paper.
    """
    mask = np.isfinite(y)
    if mask.sum() < 6:
        return dict(_IR_DEFAULT, fitted=False, n_target=int(mask.sum()))

    P0 = float(np.nanmedian(y[mask]))
    best = None
    for tf in _TAUS_F:
        for tg in _TAUS_G:
            rho_f = np.exp(-1.0 / tf)
            rho_g = np.exp(-1.0 / tg)
            F = np.zeros_like(u)
            G = np.zeros_like(u)
            for i in range(1, len(u)):
                F[i] = rho_f * F[i - 1] + u[i - 1]
                G[i] = rho_g * G[i - 1] + u[i - 1]
            for kf in _KFS:
                for kg in _KGS:
                    P_hat = P0 + kf * F - kg * G
                    mse = np.nanmean((y[mask] - P_hat[mask]) ** 2)
                    if best is None or mse < best[0]:
                        best = (mse, tf, tg, kf, kg, P0)
    _, tf, tg, kf, kg, P0 = best
    return dict(tau_f=tf, tau_g=tg, k_f=kf, k_g=kg, P0=P0,
                fitted=True, n_target=int(mask.sum()))


# ── Construção do impulso diário u_plt ────────────────────────────────────────

def build_daily_plt(df_act: pd.DataFrame,
                    df_wellness: pd.DataFrame = None,
                    modality: str = None) -> pd.DataFrame:
    """
    Constrói a série diária do PLT para uma modalidade (ou todas se modality=None).

    Inputs:
      df_act      — actividades (precisa: Data, type, icu_training_load, icu_joules,
                    hq_1, hq_4; moving_time/power_avg como fallback de kJ)
      df_wellness — wellness (precisa: Data, peso, hrv)  [peso = WELLNESS, não icu_weight]
      modality    — 'Bike'|'Row'|'Ski'|'Run' ou None (atleta inteiro)

    Output DataFrame (por dia): Data, v_tss, v_ekg, v_decoupling, v_hrv_drop, u_plt
    """
    from utils.helpers import norm_tipo  # mesma normalização do resto do dashboard

    df = df_act.copy()
    if 'Data' not in df.columns:
        return pd.DataFrame()
    df['Data'] = pd.to_datetime(df['Data'])
    if 'type' in df.columns:
        df['_tipo'] = df['type'].apply(norm_tipo)
        if modality is not None:
            df = df[df['_tipo'] == modality]
    if len(df) == 0:
        return pd.DataFrame()

    # ── TSS ← icu_training_load (fallback: min × RPE) ─────────────────────────
    if 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 5:
        df['_tss'] = pd.to_numeric(df['icu_training_load'], errors='coerce')
    elif 'moving_time' in df.columns and 'rpe' in df.columns:
        _rpe = pd.to_numeric(df['rpe'], errors='coerce')
        df['_tss'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * \
                     _rpe.fillna(_rpe.median())
    else:
        df['_tss'] = np.nan

    # ── kJ ← icu_joules/1000 (fallback power×time/1000) — convenção do dashboard ─
    # icu_joules é a energia mecânica TOTAL da sessão em joules (ex: 681613 J →
    # 681.6 kJ), por isso o /1000. Não usa z1/z2/z3_kj (que é por zona de potência).
    # KJ TOTAL sempre — SEM desconto de warm-up (decisão de projecto: consistência
    # com tab_corporal / tab_correlacoes, que já usam o total).
    if 'icu_joules' in df.columns and df['icu_joules'].notna().sum() > 5:
        df['_kj'] = pd.to_numeric(df['icu_joules'], errors='coerce') / 1000.0
        _mask = df['_kj'].isna() | (df['_kj'] <= 0)
        if 'power_avg' in df.columns and 'moving_time' in df.columns and _mask.any():
            df.loc[_mask, '_kj'] = (pd.to_numeric(df['power_avg'], errors='coerce') *
                                    pd.to_numeric(df['moving_time'], errors='coerce')
                                    / 1000.0)[_mask]
    elif 'power_avg' in df.columns and 'moving_time' in df.columns:
        df['_kj'] = (pd.to_numeric(df['power_avg'], errors='coerce') *
                     pd.to_numeric(df['moving_time'], errors='coerce') / 1000.0)
    else:
        df['_kj'] = np.nan

    # ── decoupling ← hq_4 − hq_1 (DIFERENÇA bpm, fiel ao PLT) ─────────────────
    if 'hq_1' in df.columns and 'hq_4' in df.columns:
        _h1 = pd.to_numeric(df['hq_1'], errors='coerce').replace(0, np.nan)
        _h4 = pd.to_numeric(df['hq_4'], errors='coerce').replace(0, np.nan)
        df['_decoup'] = np.where((_h1 > 40) & (_h4 > 40), _h4 - _h1, np.nan)
    else:
        df['_decoup'] = np.nan

    # ── Agregação diária (soma de carga, média de decoupling) ─────────────────
    daily = (df.groupby('Data')
               .agg(v_tss=('_tss', 'sum'),
                    _kj_day=('_kj', 'sum'),
                    v_decoupling=('_decoup', 'mean'))
               .reset_index())

    # ── Ekg ← kJ_dia / Peso (Peso do WELLNESS) ───────────────────────────────
    _peso_por_data = None
    if df_wellness is not None and len(df_wellness) > 0:
        _wc = df_wellness.copy()
        if 'Data' in _wc.columns and 'peso' in _wc.columns:
            _wc['Data'] = pd.to_datetime(_wc['Data'])
            _peso_por_data = (_wc.dropna(subset=['peso'])
                                 .groupby('Data')['peso'].mean())
    if _peso_por_data is not None and len(_peso_por_data) > 0:
        # Peso por dia: forward-fill (peso muda devagar; usa último conhecido)
        _full_idx = pd.date_range(daily['Data'].min(), daily['Data'].max(), freq='D')
        _peso_ff = (_peso_por_data.reindex(_full_idx).ffill().bfill())
        _peso_map = daily['Data'].map(_peso_ff)
        daily['v_ekg'] = np.where(_peso_map > 0, daily['_kj_day'] / _peso_map, np.nan)
    else:
        daily['v_ekg'] = np.nan  # sem peso → sem Ekg (não inventa)

    # ── hrv_drop ← baseline_7d(hrv) − hrv (morning HRV do wellness) ───────────
    daily['v_hrv_drop'] = np.nan
    if df_wellness is not None and len(df_wellness) > 0:
        _wc = df_wellness.copy()
        if 'Data' in _wc.columns and 'hrv' in _wc.columns:
            _wc['Data'] = pd.to_datetime(_wc['Data'])
            _hrv_d = (_wc.dropna(subset=['hrv'])
                         .groupby('Data')['hrv'].mean())
            if len(_hrv_d) >= 7:
                _full_idx = pd.date_range(daily['Data'].min(),
                                          daily['Data'].max(), freq='D')
                _hrv_full = _hrv_d.reindex(_full_idx)
                # baseline 7d (média móvel) − hrv do dia → queda positiva = stress
                _base7 = _hrv_full.rolling(7, min_periods=3).mean()
                _drop  = (_base7 - _hrv_full)
                daily['v_hrv_drop'] = daily['Data'].map(_drop)

    daily = daily.drop(columns=['_kj_day'])

    # ── Impulso u_plt = softplus(Σ βᵢ · robust_z(xᵢ)) ────────────────────────
    _terms = {
        'tss':        daily['v_tss'].values,
        'ekg':        daily['v_ekg'].values,
        'decoupling': daily['v_decoupling'].values,
        'hrv_drop':   daily['v_hrv_drop'].values,
    }
    proj = np.zeros(len(daily))
    for _name, _arr in _terms.items():
        if np.isfinite(_arr).sum() >= 3:
            proj = proj + _PLT_BETA[_name] * _robust_z(_arr)
    daily['u_plt'] = _softplus(proj)

    return daily


# ── API de alto nível: série PLT-IR para uma modalidade ───────────────────────

def compute_plt_ir(df_act: pd.DataFrame,
                   df_wellness: pd.DataFrame = None,
                   modality: str = None,
                   mmp_target: pd.Series = None) -> dict:
    """
    Pipeline completo PLT → IR para uma modalidade.

    mmp_target : Série indexada por Data com o MMP real (mmp20_w/mmp5_w) para fit.
                 Se None ou insuficiente (<6 pontos) → usa parâmetros fixos do paper.
                 NUNCA passar eFTP aqui (é estimador ruidoso).

    Devolve dict:
      'daily'   — DataFrame (Data, v_*, u_plt, F_plt, G_plt, P_hat_plt)
      'params'  — dict de parâmetros IR usados (+ flag 'fitted')
      'ok'      — bool
      'reason'  — str (se ok=False)
    """
    daily = build_daily_plt(df_act, df_wellness, modality)
    if len(daily) == 0 or daily['u_plt'].notna().sum() < 10:
        return {'ok': False, 'reason': 'dados insuficientes para PLT',
                'daily': daily, 'params': dict(_IR_DEFAULT, fitted=False)}

    # Série diária contínua (reindex para dias sem treino = impulso 0)
    _full = pd.date_range(daily['Data'].min(), daily['Data'].max(), freq='D')
    daily = (daily.set_index('Data').reindex(_full)
                  .rename_axis('Data').reset_index())
    daily['u_plt'] = daily['u_plt'].fillna(0.0)  # dia sem treino → impulso 0
    u = daily['u_plt'].values.astype(float)

    # Alinhar alvo MMP à grelha diária (para o fit)
    y = np.full(len(daily), np.nan)
    if mmp_target is not None and len(mmp_target) > 0:
        _mt = mmp_target.copy()
        _mt.index = pd.to_datetime(_mt.index)
        _y_ser = _mt.reindex(daily['Data'])
        y = pd.to_numeric(_y_ser, errors='coerce').values

    params = _gridfit_ir(u, y)
    F, G, P_hat = _simulate_ir(u, u, params['tau_f'], params['tau_g'],
                               params['k_f'], params['k_g'], params['P0'])
    daily['F_plt']     = F
    daily['G_plt']     = G
    daily['P_hat_plt'] = P_hat

    return {'ok': True, 'reason': '', 'daily': daily, 'params': params}
