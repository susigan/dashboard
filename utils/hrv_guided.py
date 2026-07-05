# ══════════════════════════════════════════════════════════════════════════════
# utils/hrv_guided.py — ATHELTICA
# Módulo central de análise HRV-Guided. Fonte ÚNICA de verdade para:
#   • Modelo β (Della Mattia)          → calcular_modelo_beta()
#   • Regra de convergência β          → regra_convergencia()
#   • Máquina de estados (Fig.1)       → state_machine()
#   • Javaloyes (LnRMSSD 7d, banda SWC)→ calcular_javaloyes()
#   • Kiviniemi (HF power 10d)         → calcular_kiviniemi()
#   • Prescrição de hoje (helper)      → prescricao_hoje()
#
# Todas as funções são PURAS (sem Streamlit) para poderem ser chamadas por
# qualquer tab (tab_recovery, tab_visao_geral, etc.) e garantir consistência.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

# Rótulos e cores partilhados
LABEL_MAP = {'HIGH': '🟢 HIGH', 'LOW': '🔵 LOW', 'REST': '🔴 REST'}
COR_MAP   = {'HIGH': '#27ae60', 'LOW': '#3498db', 'REST': '#e74c3c'}


# ── Helper z-score rolling ─────────────────────────────────────────────────────

def _zc(s, win=28, minp=7):
    """z-score rolling (centrado na distribuição pessoal dos últimos `win` dias)."""
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std().replace(0, np.nan)
    return (s - m) / sd


# ── MODELO β (Della Mattia) ────────────────────────────────────────────────────

def calcular_modelo_beta(wc_src, da_src=None, modo='hrv', dw_fallback=None):
    """
    Modelo β de frescura (Della Mattia).
      modo='hrv'   → β só do LnRMSSD (core; correlaciona bem com HRV matinal).
      modo='multi' → funde HRV + sono(qualidade) + RHR(inv) + carga(kJ+km+TSS 7d, inv).

    Agudo/Crónico em PONTOS do β (média₃d−média₇d ; média₇d−média₂₈d) — escala 0-100.
    Devolve DataFrame (índice Data) com: LnrMSSD, bm28, bs28, beta, beta_agudo,
    beta_cronico (+ z_hrv/z_sono/z_fc/z_carga se modo='multi'). tail(90).
    """
    import scipy.stats as _sst
    src = wc_src.copy() if wc_src is not None and len(wc_src) > 0 else (
        dw_fallback.copy() if dw_fallback is not None else None)
    if src is None or len(src) == 0:
        return None
    src['Data'] = pd.to_datetime(src['Data'])
    src = src.sort_values('Data').set_index('Data')
    date_range = pd.date_range(src.index.min(), src.index.max(), freq='D')
    src = src.reindex(date_range)
    if 'hrv' not in src.columns:
        return None
    src['LnrMSSD'] = np.where(src['hrv'].notna() & (src['hrv'] > 0), np.log(src['hrv']), np.nan)

    z_hrv = _zc(src['LnrMSSD'])

    if modo == 'multi':
        # Sono: qualidade 1-5 → reescala ×2 (1-10), z-score direto
        if 'sleep_quality' in src.columns and src['sleep_quality'].notna().sum() >= 7:
            z_sono = _zc(pd.to_numeric(src['sleep_quality'], errors='coerce') * 2.0)
        else:
            z_sono = pd.Series(np.nan, index=src.index)
        # FC: RHR z-score, INVERTIDO (RHR alto = pior)
        if 'rhr' in src.columns and src['rhr'].notna().sum() >= 7:
            z_fc = -_zc(pd.to_numeric(src['rhr'], errors='coerce'))
        else:
            z_fc = pd.Series(np.nan, index=src.index)
        # Carga: kJ (z1+z2+z3 ou icu_joules/1000) + km (distance/1000) + TSS,
        # 7d acumulado, z-score de cada, média, INVERTIDO
        z_carga = pd.Series(np.nan, index=src.index)
        if da_src is not None and len(da_src) > 0:
            _da = da_src.copy()
            _da['Data'] = pd.to_datetime(_da['Data']).dt.normalize()
            _kjz = [c for c in ['z1_kj', 'z2_kj', 'z3_kj'] if c in _da.columns]
            if _kjz:
                _da['_kj'] = sum(pd.to_numeric(_da[c], errors='coerce').fillna(0) for c in _kjz)
            elif 'icu_joules' in _da.columns:
                _da['_kj'] = pd.to_numeric(_da['icu_joules'], errors='coerce') / 1000.0
            else:
                _da['_kj'] = np.nan
            _da['_km'] = (pd.to_numeric(_da.get('distance'), errors='coerce') / 1000.0
                          if 'distance' in _da.columns else np.nan)
            _da['_tss'] = (pd.to_numeric(_da.get('icu_training_load'), errors='coerce')
                           if 'icu_training_load' in _da.columns else np.nan)
            _daily = _da.groupby('Data').agg(_kj=('_kj', 'sum'), _km=('_km', 'sum'),
                                             _tss=('_tss', 'sum'))
            _daily = _daily.reindex(date_range).fillna(0.0)
            _z_kj  = _zc(_daily['_kj'].rolling(7, min_periods=3).sum())
            _z_km  = _zc(_daily['_km'].rolling(7, min_periods=3).sum())
            _z_tss = _zc(_daily['_tss'].rolling(7, min_periods=3).sum())
            z_carga = -pd.concat([_z_kj, _z_km, _z_tss], axis=1).mean(axis=1)

        _pesos = {'hrv': 0.40, 'sono': 0.20, 'fc': 0.20, 'carga': 0.20}
        _canais = {'hrv': z_hrv, 'sono': z_sono, 'fc': z_fc, 'carga': z_carga}
        _zt = pd.Series(0.0, index=src.index); _wsum = pd.Series(0.0, index=src.index)
        for _nome, _zcanal in _canais.items():
            _w = _pesos[_nome]; _mask = _zcanal.notna()
            _zt = _zt.add((_zcanal * _w).where(_mask, 0.0), fill_value=0.0)
            _wsum = _wsum.add(pd.Series(_w, index=src.index).where(_mask, 0.0), fill_value=0.0)
        z28 = _zt / _wsum.replace(0, np.nan)
        src['z_hrv'] = z_hrv; src['z_sono'] = z_sono
        src['z_fc'] = z_fc; src['z_carga'] = z_carga
    else:
        z28 = z_hrv

    src['z28'] = z28
    src['beta'] = src['z28'].apply(lambda z: round(float(_sst.norm.cdf(z) * 100), 1) if pd.notna(z) else np.nan)
    # Agudo/Crónico em PONTOS do β
    _bser = src['beta']
    _bm3  = _bser.rolling(3, min_periods=2).mean()
    _bm7  = _bser.rolling(7, min_periods=4).mean()
    _bm28 = _bser.rolling(28, min_periods=7).mean()
    src['bm28'] = src['LnrMSSD'].rolling(28, min_periods=7).mean()
    src['bs28'] = src['LnrMSSD'].rolling(28, min_periods=7).std()
    src['beta_agudo']   = _bm3 - _bm7
    src['beta_cronico'] = _bm7 - _bm28
    _cols = ['LnrMSSD', 'bm28', 'bs28', 'beta', 'beta_agudo', 'beta_cronico']
    if modo == 'multi':
        _cols += ['z_hrv', 'z_sono', 'z_fc', 'z_carga']
    return src[_cols].tail(90)


def regra_convergencia(beta, b_agudo, b_cronico, hrv_hoje_notna):
    """
    Regra de decisão do β: actuar quando ≥2 dos 3 indicadores convergem.
    Cortes: β 60/40 ; agudo/crónico ±2 pontos.
    Devolve (prescricao, cor, n_pos, n_neg, n_inc, sinais).
    """
    sinais = []
    if pd.isna(beta): sinais.append(('β actual', 0, 'NaN — sem medição hoje', '#888'))
    elif beta >= 60: sinais.append(('β actual', +1, f'{beta:.0f} ≥ 60 ✅', '#27ae60'))
    elif beta <= 40: sinais.append(('β actual', -1, f'{beta:.0f} ≤ 40 ⚠️', '#e74c3c'))
    else: sinais.append(('β actual', 0, f'{beta:.0f} zona neutra (40-60)', '#f39c12'))
    if pd.isna(b_agudo): sinais.append(('βAgudo 3d', 0, 'NaN — dados insuficientes', '#888'))
    elif b_agudo >= 2.0: sinais.append(('βAgudo 3d', +1, f'{b_agudo:+.1f} pts ≥ +2 ✅', '#27ae60'))
    elif b_agudo <= -2.0: sinais.append(('βAgudo 3d', -1, f'{b_agudo:+.1f} pts ≤ -2 ⚠️', '#e74c3c'))
    else: sinais.append(('βAgudo 3d', 0, f'{b_agudo:+.1f} pts zona neutra', '#f39c12'))
    if pd.isna(b_cronico): sinais.append(('βCrónico 7d', 0, 'NaN — dados insuficientes', '#888'))
    elif b_cronico >= 2.0: sinais.append(('βCrónico 7d', +1, f'{b_cronico:+.1f} pts ≥ +2 ✅', '#27ae60'))
    elif b_cronico <= -2.0: sinais.append(('βCrónico 7d', -1, f'{b_cronico:+.1f} pts ≤ -2 ⚠️', '#e74c3c'))
    else: sinais.append(('βCrónico 7d', 0, f'{b_cronico:+.1f} pts zona neutra', '#f39c12'))
    n_pos = sum(1 for _, s, _, _ in sinais if s == +1)
    n_neg = sum(1 for _, s, _, _ in sinais if s == -1)
    n_inc = sum(1 for _, s, _, _ in sinais if s == 0)
    if not hrv_hoje_notna:
        return "⚠️ SEM MEDIÇÃO HOJE — Não prescrever HIIT", "#e67e22", n_pos, n_neg, n_inc, sinais
    if n_pos >= 2: return "✅ HIIT / Alta intensidade — ≥2 sinais positivos", "#27ae60", n_pos, n_neg, n_inc, sinais
    elif n_neg >= 2: return "🔴 Recuperação activa — ≥2 sinais negativos", "#e74c3c", n_pos, n_neg, n_inc, sinais
    elif n_neg >= 1 and n_inc >= 1: return "🟠 Sessão moderada Z1/Z2 — 1 sinal negativo + incerteza", "#e67e22", n_pos, n_neg, n_inc, sinais
    elif n_pos == 1 and n_inc >= 2: return "🟡 Sessão moderada Z1/Z2 — sinais insuficientes para HIIT", "#f39c12", n_pos, n_neg, n_inc, sinais
    else: return "🟡 Zona neutra — manter intensidade planeada", "#f39c12", n_pos, n_neg, n_inc, sinais


# ── MÁQUINA DE ESTADOS (Fig.1 Kiviniemi/Javaloyes) ─────────────────────────────

def state_machine(vals, sig_fn, max_high=2, max_train_days=None):
    """
    Máquina de estados fiel à Fig.1.
      max_high: teto de HIGH consecutivos (Javaloyes=2; Kiviniemi=None).
      max_train_days: dias de treino consec. → REST forçado (Kiviniemi=9; Javaloyes=None).
      sig_fn(i, v, prev, prev2) → 'HRV+'/'HF+' (recuperado) | 'HRV−'/'HF−' | '·' (indef).
    Devolve (prescricoes, sinais) — listas do mesmo tamanho de vals.
    """
    pres = []; sig = []; estado = 'START'; ch = 0; cr = 0; ct = 0
    prev = np.nan; prev2 = np.nan
    for i, v in enumerate(vals):
        s = sig_fn(i, v, prev, prev2)
        if pd.isna(v) or s == '·':
            p = 'LOW'; estado = 'LOW'; ch = 0; cr = 0; ct = 0
        elif estado == 'START':
            p = 'LOW'; estado = 'LOW'; ch = 0; cr = 0; ct = 1
        elif max_train_days is not None and ct >= max_train_days:
            p = 'REST'; estado = 'REST'; cr = 1; ch = 0; ct = 0
        elif estado == 'HIGH':
            if s.endswith('+'):
                if max_high is not None and ch >= max_high:
                    p = 'LOW'; estado = 'LOW'; ch = 0; ct += 1
                else:
                    p = 'HIGH'; estado = 'HIGH'; ch += 1; ct += 1
            else:
                p = 'LOW'; estado = 'LOW'; ch = 0; ct += 1
            cr = 0
        elif estado == 'LOW':
            if s.endswith('+'):
                p = 'HIGH'; estado = 'HIGH'; ch = 1; cr = 0; ct += 1
            else:
                p = 'REST'; estado = 'REST'; cr = 1; ct = 0
        elif estado == 'REST':
            if cr >= 2: p = 'LOW'; estado = 'LOW'; cr = 0; ct = 1
            elif s.endswith('+'): p = 'LOW'; estado = 'LOW'; cr = 0; ct = 1
            else: p = 'REST'; estado = 'REST'; cr += 1; ct = 0
            ch = 0
        else:
            p = 'LOW'; estado = 'LOW'; ct = 1
        pres.append(p); sig.append(s)
        if pd.notna(v): prev2 = prev; prev = v
    return pres, sig


# ── JAVALOYES (LnRMSSD 7d, banda SWC ±0.5·SD) ──────────────────────────────────

def calcular_javaloyes(wc_src, baseline_win=28):
    """
    Prescrição Javaloyes (máquina de estados HIGH/LOW/REST).
      • LnRMSSD, rolling 7d.
      • Banda SWC = mean ± 0.5·SD (baseline dos últimos `baseline_win` dias reais).
      • Sinal pela banda: HRV+ = ln7 dentro/acima ; HRV− = abaixo do limite inf.
      • Teto de 2 HIGH consecutivos (fiel ao paper).
    Devolve DataFrame com Data, LnrMSSD, ln7, prescricao, hrv_sinal, swc_inf, swc_sup.
    """
    _wc = wc_src.copy()
    _wc['Data'] = pd.to_datetime(_wc['Data'])
    _wc = _wc.sort_values('Data').set_index('Data')
    _full = pd.date_range(_wc.index.min(), _wc.index.max(), freq='D')
    _wc = _wc.reindex(_full).rename_axis('Data').reset_index()
    _wc['LnrMSSD'] = np.where(_wc['hrv'].notna() & (_wc['hrv'] > 0), np.log(_wc['hrv']), np.nan)
    _wc['sem_medicao'] = _wc['LnrMSSD'].isna()

    _ln_real = _wc[~_wc['sem_medicao']]['LnrMSSD'].dropna()
    if len(_ln_real) < 7:
        _wc['ln7'] = np.nan; _wc['prescricao'] = 'LOW'; _wc['hrv_sinal'] = '·'
        _wc['swc_inf'] = np.nan; _wc['swc_sup'] = np.nan
        return _wc
    _lnb = _ln_real.tail(baseline_win)
    _swc_m = float(_lnb.mean()); _swc_s = float(_lnb.std())
    swc_sup = _swc_m + 0.5 * _swc_s; swc_inf = _swc_m - 0.5 * _swc_s
    _wc['ln7'] = _wc['LnrMSSD'].rolling(7, min_periods=4).mean()

    def _sig_jav(i, v, p, p2):
        if pd.isna(v): return '·'
        return 'HRV+' if v >= swc_inf else 'HRV−'
    _pj, _sj = state_machine(_wc['ln7'].tolist(), _sig_jav, max_high=2, max_train_days=None)
    _wc['prescricao'] = _pj; _wc['hrv_sinal'] = _sj
    _wc['swc_inf'] = swc_inf; _wc['swc_sup'] = swc_sup
    return _wc


# ── KIVINIEMI (HF power 10d, ref mean−1·SD, sem teto de 2, Rest após 9 dias) ────

def calcular_kiviniemi(wc_src):
    """
    Prescrição Kiviniemi (HF power).
      • Métrica = HF power (heurística ln se mediana>10, senão cru).
      • Referência 10d = mean − 1·SD ; + tendência decrescente 2 dias (>0.1).
      • SEM teto de 2 HIGH ; Rest forçado após 9 dias de treino consecutivo.
    Devolve DataFrame com Data, hf_metric, hf_ref, prescricao_k, hf_sinal.
    Se não houver HF power suficiente, devolve None.
    """
    if 'hf_power' not in wc_src.columns:
        return None
    _wc = wc_src.copy()
    _wc['Data'] = pd.to_datetime(_wc['Data'])
    _wc = _wc.sort_values('Data').set_index('Data')
    _full = pd.date_range(_wc.index.min(), _wc.index.max(), freq='D')
    _wc = _wc.reindex(_full).rename_axis('Data').reset_index()
    if _wc['hf_power'].notna().sum() < 5:
        return None

    _hf = pd.to_numeric(_wc['hf_power'], errors='coerce').replace(0, np.nan)
    _med = _hf.median()
    _wc['hf_metric'] = np.log(_hf.where(_hf > 0)) if (_med and _med > 10) else _hf
    _wc['hf_mean10'] = _wc['hf_metric'].rolling(10, min_periods=5).mean()
    _wc['hf_sd10'] = _wc['hf_metric'].rolling(10, min_periods=5).std()
    _wc['hf_ref'] = _wc['hf_mean10'] - 1.0 * _wc['hf_sd10']
    _refs = _wc['hf_ref'].tolist()

    def _sig_kiv(i, v, p, p2):
        ref = _refs[i]
        if pd.isna(v) or pd.isna(ref): return '·'
        below = v < ref
        trend = (pd.notna(p) and pd.notna(p2) and (p2 - p) > 0.1 and (p - v) > 0.1)
        return 'HF−' if (below or trend) else 'HF+'
    _pk, _sk = state_machine(_wc['hf_metric'].tolist(), _sig_kiv,
                             max_high=None, max_train_days=9)
    _wc['prescricao_k'] = _pk; _wc['hf_sinal'] = _sk
    return _wc


# ── HELPER: prescrição de HOJE (para cards compactos / visão geral) ────────────

def prescricao_hoje(wc_src, da_src=None):
    """
    Devolve um dict resumo do estado de HOJE, consolidando os métodos:
      {
        'javaloyes': 'HIGH'|'LOW'|'REST'|None,
        'javaloyes_label': '🟢 HIGH'|...,
        'kiviniemi': 'HIGH'|'LOW'|'REST'|None,
        'beta': float|None, 'beta_prescricao': str|None, 'beta_cor': str|None,
        'hrv_hoje': float|None, 'dias_sem_medicao': int,
      }
    Uso típico na visão geral: puxar 1 linha sem recalcular nada à mão.
    """
    out = {'javaloyes': None, 'javaloyes_label': None, 'javaloyes_cor': None,
           'kiviniemi': None, 'kiviniemi_label': None,
           'beta': None, 'beta_prescricao': None, 'beta_cor': None,
           'hrv_hoje': None, 'dias_sem_medicao': 0}
    if wc_src is None or len(wc_src) == 0 or 'hrv' not in wc_src.columns:
        return out

    # Javaloyes
    try:
        _jav = calcular_javaloyes(wc_src)
        _jav_val = _jav[_jav['ln7'].notna()]
        if len(_jav_val) > 0:
            _last = _jav_val.iloc[-1]
            out['javaloyes'] = _last['prescricao']
            out['javaloyes_label'] = LABEL_MAP.get(_last['prescricao'])
            out['javaloyes_cor'] = COR_MAP.get(_last['prescricao'])
    except Exception:
        pass

    # Kiviniemi
    try:
        _kiv = calcular_kiviniemi(wc_src)
        if _kiv is not None:
            _kiv_val = _kiv[_kiv['hf_metric'].notna()]
            if len(_kiv_val) > 0:
                _lastk = _kiv_val.iloc[-1]
                out['kiviniemi'] = _lastk['prescricao_k']
                out['kiviniemi_label'] = LABEL_MAP.get(_lastk['prescricao_k'])
    except Exception:
        pass

    # Modelo β
    try:
        _bdf = calcular_modelo_beta(wc_src, da_src=da_src, modo='hrv')
        if _bdf is not None and not _bdf.empty and not _bdf['beta'].isna().all():
            _ub = _bdf.iloc[-1]
            out['beta'] = _ub['beta']
            _hrv_notna = pd.notna(_ub['LnrMSSD'])
            _presc, _cor, *_ = regra_convergencia(
                _ub['beta'], _ub['beta_agudo'], _ub['beta_cronico'], _hrv_notna)
            out['beta_prescricao'] = _presc; out['beta_cor'] = _cor
            # hrv hoje + dias sem medição
            _med = _bdf['LnrMSSD'].dropna()
            if not _med.empty:
                _wc2 = wc_src.copy(); _wc2['Data'] = pd.to_datetime(_wc2['Data'])
                _wc2 = _wc2.sort_values('Data')
                _hrv_last = _wc2[_wc2['hrv'].notna()]
                if len(_hrv_last) > 0:
                    out['hrv_hoje'] = float(_hrv_last.iloc[-1]['hrv'])
                    _last_date = pd.to_datetime(_hrv_last.iloc[-1]['Data']).normalize()
                    _today = pd.Timestamp('today').normalize()
                    out['dias_sem_medicao'] = max(0, (_today - _last_date).days)
    except Exception:
        pass

    return out
