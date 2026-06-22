# ══════════════════════════════════════════════════════════════════════════════
# utils/plt_compare.py — ATHELTICA
# Comparação de métodos de parametrização do IR do PLT.
#
# Objectivo: gerar, por modalidade, o P̂ segundo 4 métodos de parametrização,
# para exportar a CSV e decidir empiricamente qual usar no PMC.
#
# MÉTODOS:
#   M1 PAPER     — τf=42, τg=7, kf=1, kg=2 (fixos do paper Della Mattia)
#   M2 NLSS_kfix — τf←T1, τg←T2 (herdados do calcular_nlss, ajustados aos dados)
#                  + kf=1, kg=2 fixos do paper
#   M3 NLSS_kfit — τf←T1, τg←T2 herdados + kf/kg ajustados ao MMP real
#   M4 GRIDFIT   — varre τf,τg,kf,kg,P0 minimizando erro vs MMP real
#
# ALVO DE FIT (M3, M4): MMP real "Yes - Xw" — o MELHOR por período (ano/semestre),
#   para ter pontos-âncora espaçados no tempo sem acumular ruído de uma só época.
#   Bike/Run → MMP20 ; Row/Ski → MMP5. NUNCA eFTP.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

from utils.plt_model import (build_daily_plt, _simulate_ir, _softplus,
                             parse_mmp_real, _IR_DEFAULT)

# Grelhas de fit (iguais ao plt_model)
_TAUS_F = (20, 30, 42, 60)
_TAUS_G = (3, 5, 7, 10, 14)
_KFS    = (0.3, 0.5, 1.0, 1.5)
_KGS    = (0.8, 1.0, 1.5, 2.0, 3.0)


# ── Pontos-âncora MMP: melhor "Yes" por período ───────────────────────────────

def mmp_anchors(df_act: pd.DataFrame, modality: str,
                period: str = 'year') -> pd.Series:
    """
    Série de pontos-âncora MMP reais ("Yes - Xw"), 1 por período (o melhor/máximo).
      period='year'    → melhor Yes de cada ano civil
      period='semester'→ melhor Yes de cada semestre (H1/H2)
      period='quarter' → melhor Yes de cada trimestre
    Coluna: MMP20 (Bike/Run) | MMP5 (Row/Ski), lida do nome ORIGINAL da sheet.
    Devolve Série indexada por Data (a data do PR) → watts. Vazia se não houver.
    """
    from utils.helpers import norm_tipo
    if df_act is None or len(df_act) == 0 or 'type' not in df_act.columns:
        return pd.Series(dtype=float)
    if 'Data' not in df_act.columns:
        return pd.Series(dtype=float)

    _cands = (['MMP5', 'mmp5'] if modality in ('Row', 'Ski')
              else ['MMP20', 'mmp20'])
    _col = next((c for c in _cands if c in df_act.columns), None)
    if _col is None:
        return pd.Series(dtype=float)

    _sub = df_act[df_act['type'].apply(norm_tipo) == modality].copy()
    if len(_sub) == 0:
        return pd.Series(dtype=float)
    _sub['Data'] = pd.to_datetime(_sub['Data'])
    _sub['_w'] = _sub[_col].apply(lambda v: parse_mmp_real(str(v)))
    _sub = _sub.dropna(subset=['_w'])
    if len(_sub) == 0:
        return pd.Series(dtype=float)

    if period == 'year':
        _sub['_per'] = _sub['Data'].dt.year
    elif period == 'semester':
        _sub['_per'] = (_sub['Data'].dt.year.astype(str) + '-H' +
                        ((_sub['Data'].dt.month > 6).astype(int) + 1).astype(str))
    else:  # quarter
        _sub['_per'] = (_sub['Data'].dt.year.astype(str) + '-Q' +
                        _sub['Data'].dt.quarter.astype(str))

    # melhor (máximo) por período, na data em que ocorreu
    _idx = _sub.groupby('_per')['_w'].idxmax()
    _best = _sub.loc[_idx, ['Data', '_w']].sort_values('Data')
    return pd.Series(_best['_w'].values, index=_best['Data'].values)


# ── Fits ──────────────────────────────────────────────────────────────────────

def _ir_from_u(u, tau_f, tau_g, kf, kg, P0):
    F, G, P_hat = _simulate_ir(u, u, tau_f, tau_g, kf, kg, P0)
    return F, G, P_hat


def _align_targets(daily_dates, anchors: pd.Series):
    """Alinha os pontos-âncora MMP à grelha diária → array y com NaN fora dos PRs."""
    y = np.full(len(daily_dates), np.nan)
    if anchors is None or len(anchors) == 0:
        return y
    _a = anchors.copy()
    _a.index = pd.to_datetime(_a.index)
    _dser = pd.Series(range(len(daily_dates)), index=pd.to_datetime(daily_dates))
    for _dt, _w in _a.items():
        # mapear ao dia mais próximo da grelha
        _pos = _dser.index.get_indexer([_dt], method='nearest')
        if len(_pos) and _pos[0] >= 0:
            y[_pos[0]] = _w
    return y


def _fit_kf_kg(u, y, tau_f, tau_g):
    """Ajusta kf,kg,P0 (τ fixos) minimizando MSE vs y. P0=mediana(y)."""
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return None
    P0 = float(np.nanmedian(y[mask]))
    rho_f = np.exp(-1.0 / tau_f); rho_g = np.exp(-1.0 / tau_g)
    F = np.zeros_like(u); G = np.zeros_like(u)
    for i in range(1, len(u)):
        F[i] = rho_f * F[i-1] + u[i-1]
        G[i] = rho_g * G[i-1] + u[i-1]
    best = None
    for kf in _KFS:
        for kg in _KGS:
            P_hat = P0 + kf*F - kg*G
            mse = np.nanmean((y[mask] - P_hat[mask])**2)
            if best is None or mse < best[0]:
                best = (mse, kf, kg, P0)
    _, kf, kg, P0 = best
    return dict(tau_f=tau_f, tau_g=tau_g, k_f=kf, k_g=kg, P0=P0,
                mse=best[0], n_target=int(mask.sum()))


def _gridfit_full(u, y):
    """Varre τf,τg,kf,kg,P0 minimizando MSE vs y."""
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return None
    P0 = float(np.nanmedian(y[mask]))
    best = None
    for tf in _TAUS_F:
        for tg in _TAUS_G:
            rho_f = np.exp(-1.0/tf); rho_g = np.exp(-1.0/tg)
            F = np.zeros_like(u); G = np.zeros_like(u)
            for i in range(1, len(u)):
                F[i] = rho_f*F[i-1] + u[i-1]
                G[i] = rho_g*G[i-1] + u[i-1]
            for kf in _KFS:
                for kg in _KGS:
                    P_hat = P0 + kf*F - kg*G
                    mse = np.nanmean((y[mask] - P_hat[mask])**2)
                    if best is None or mse < best[0]:
                        best = (mse, tf, tg, kf, kg, P0)
    _, tf, tg, kf, kg, P0 = best
    return dict(tau_f=tf, tau_g=tg, k_f=kf, k_g=kg, P0=P0,
                mse=best[0], n_target=int(mask.sum()))


# ── API principal ─────────────────────────────────────────────────────────────

def compare_methods(df_act, df_wellness, modality,
                    nlss_fn=None, period='year',
                    start_date='2021-01-01') -> dict:
    """
    Calcula P̂ por 4 métodos para uma modalidade.

    nlss_fn : referência a calcular_nlss (de utils.data). Se None, M2/M3 caem
              nos τ do paper (sem herança). Chamado com df_act filtrado pela
              modalidade para obter T1/T2 específicos do desporto.
    period  : período dos pontos-âncora MMP ('year'|'semester'|'quarter').

    Devolve dict com 'daily' (DataFrame: Data + P_hat_* dos 4 métodos + u_plt)
    e 'params' (dict de parâmetros por método) e 'anchors' (Série MMP usada).
    """
    daily = build_daily_plt(df_act, df_wellness, modality, start_date=start_date)
    if len(daily) == 0 or daily['u_plt'].notna().sum() < 10:
        return {'ok': False, 'reason': 'dados insuficientes', 'modality': modality}

    # série contínua
    _full = pd.date_range(daily['Data'].min(), daily['Data'].max(), freq='D')
    daily = (daily.set_index('Data').reindex(_full)
                  .rename_axis('Data').reset_index())
    daily['u_plt'] = daily['u_plt'].fillna(0.0)
    u = daily['u_plt'].values.astype(float)

    # pontos-âncora MMP
    anchors = mmp_anchors(df_act, modality, period=period)
    y = _align_targets(daily['Data'].values, anchors)

    # T1/T2 do NLSS por modalidade (se fornecido)
    tau_f_nlss, tau_g_nlss = 42.0, 7.0
    nlss_info = {}
    if nlss_fn is not None:
        try:
            from utils.helpers import norm_tipo
            _dm = df_act[df_act['type'].apply(norm_tipo) == modality].copy()
            if len(_dm) >= 20:
                _r = nlss_fn(_dm, df_wellness)
                tau_f_nlss = float(_r.get('T1', 42.0))
                tau_g_nlss = float(_r.get('T2', 7.0))
                nlss_info = {'T1': tau_f_nlss, 'T2': tau_g_nlss,
                             'K1': _r.get('K1'), 'K2': _r.get('K2')}
        except Exception as _e:
            nlss_info = {'erro': str(_e)}

    params = {}

    # M1 PAPER
    p1 = dict(_IR_DEFAULT)
    _, _, ph1 = _ir_from_u(u, p1['tau_f'], p1['tau_g'], p1['k_f'], p1['k_g'], p1['P0'])
    daily['P_hat_paper'] = ph1
    params['paper'] = dict(p1, fitted=False)

    # M2 NLSS τ herdado + k fixo
    _, _, ph2 = _ir_from_u(u, tau_f_nlss, tau_g_nlss, 1.0, 2.0, 0.0)
    daily['P_hat_nlss_kfix'] = ph2
    params['nlss_kfix'] = dict(tau_f=tau_f_nlss, tau_g=tau_g_nlss,
                               k_f=1.0, k_g=2.0, P0=0.0, **nlss_info)

    # M3 NLSS τ herdado + k ajustado ao MMP
    f3 = _fit_kf_kg(u, y, tau_f_nlss, tau_g_nlss)
    if f3 is not None:
        _, _, ph3 = _ir_from_u(u, f3['tau_f'], f3['tau_g'], f3['k_f'], f3['k_g'], f3['P0'])
        daily['P_hat_nlss_kfit'] = ph3
        params['nlss_kfit'] = dict(f3, fitted=True)
    else:
        daily['P_hat_nlss_kfit'] = np.nan
        params['nlss_kfit'] = {'fitted': False, 'reason': 'sem MMP suficiente'}

    # M4 GRIDFIT completo
    f4 = _gridfit_full(u, y)
    if f4 is not None:
        _, _, ph4 = _ir_from_u(u, f4['tau_f'], f4['tau_g'], f4['k_f'], f4['k_g'], f4['P0'])
        daily['P_hat_gridfit'] = ph4
        params['gridfit'] = dict(f4, fitted=True)
    else:
        daily['P_hat_gridfit'] = np.nan
        params['gridfit'] = {'fitted': False, 'reason': 'sem MMP suficiente'}

    return {'ok': True, 'modality': modality, 'daily': daily,
            'params': params, 'anchors': anchors, 'period': period}


def compare_all_to_csv(df_act, df_wellness, nlss_fn=None,
                       period='year', start_date='2021-01-01',
                       mods=('Bike', 'Row', 'Ski', 'Run')) -> tuple:
    """
    Corre compare_methods para todas as modalidades e devolve (df_long, df_params).
      df_long   : Data, Modalidade, u_plt, P_hat_paper, P_hat_nlss_kfix,
                  P_hat_nlss_kfit, P_hat_gridfit
      df_params : resumo dos parâmetros e MSE por método e modalidade
    """
    _frames = []
    _prows = []
    for _m in mods:
        _r = compare_methods(df_act, df_wellness, _m,
                             nlss_fn=nlss_fn, period=period, start_date=start_date)
        if not _r.get('ok'):
            continue
        _d = _r['daily'].copy()
        _d.insert(1, 'Modalidade', _m)
        _frames.append(_d)
        for _meth, _p in _r['params'].items():
            _prows.append({
                'Modalidade': _m, 'Metodo': _meth,
                'tau_f': _p.get('tau_f'), 'tau_g': _p.get('tau_g'),
                'k_f': _p.get('k_f'), 'k_g': _p.get('k_g'),
                'P0': _p.get('P0'), 'mse': _p.get('mse'),
                'n_anchors': _p.get('n_target'),
                'fitted': _p.get('fitted', False),
                'T1_nlss': _p.get('T1'), 'T2_nlss': _p.get('T2'),
            })
    df_long = (pd.concat(_frames, ignore_index=True)
               if _frames else pd.DataFrame())
    df_params = pd.DataFrame(_prows)
    return df_long, df_params
