# ════════════════════════════════════════════════════════════════════════════════
# utils/helpers.py — ATHELTICA Dashboard — funções auxiliares reutilizáveis
# ════════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from datetime import datetime, timedelta
from config import CORES, CORES_ATIV, TYPE_MAP, VALID_TYPES

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

def analisar_falta_estimulo(df_act_full, janela_dias=14):
    """Need Score por modalidade — quanto estímulo está em falta."""
    ld, _ = calcular_series_carga(df_act_full)
    if len(ld) == 0: return None
    df = filtrar_principais(df_act_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    data_lim = pd.Timestamp.now() - pd.Timedelta(days=janela_dias)
    cr = ld[ld['Data'] >= data_lim].copy()
    if len(cr) == 0: return None
    res = {}
    for mod in ['Bike', 'Run', 'Row', 'Ski']:
        dm = df[(df['type'] == mod) & (df['Data'] >= data_lim)]
        dias_a = dm['Data'].nunique(); freq = dias_a / max(janela_dias, 1)
        atl_m = cr['ATL'].mean(); ctl_m = cr['CTL'].mean()
        gap = ((ctl_m - atl_m) / ctl_m * 100) if ctl_m > 0 else 0
        dias_b = (cr['ATL'] < cr['CTL']).sum()
        sl = np.polyfit(np.arange(len(cr)), cr['ATL'].values, 1)[0] if len(cr) > 1 else 0
        sl_n = max(0, min(1, (sl + 5) / 10))
        need = (min(1, max(0, gap / 50)) * 100 * 0.4 +
                min(1, dias_b / max(len(cr), 1)) * 100 * 0.3 +
                (1 - sl_n) * 100 * 0.2 +
                (1 - freq) * 100 * 0.1)
        prio = 'ALTA' if need >= 70 else 'MÉDIA' if need >= 40 else 'BAIXA'
        res[mod] = {'need_score': need, 'prioridade': prio, 'gap_relativo': gap,
                    'dias_atl_menor_ctl': int(dias_b), 'dias_com_atividade': dias_a,
                    'atl_medio': atl_m, 'ctl_medio': ctl_m}
    return dict(sorted(res.items(), key=lambda x: x[1]['need_score'], reverse=True))



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
