# ══════════════════════════════════════════════════════════════════════════════
# app.py — ATHELTICA Dashboard
# ══════════════════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os
import json

credentials = json.loads(os.getenv("GCP_SERVICE_ACCOUNT"))


import streamlit as st
st.set_page_config(
    page_title="ATHELTICA",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.config   import *
from utils.helpers  import *
from utils.data     import *
from utils.sidebar  import render_sidebar

from tabs.tab_visao_geral  import tab_visao_geral
from tabs.tab_pmc          import tab_pmc
from tabs.tab_volume       import tab_volume
from tabs.tab_eftp         import tab_eftp
from tabs.tab_zones        import tab_zones
# ❌ REMOVIDO: tab_correlacoes agora em app_hrv.py (app independente)
from tabs.tab_recovery     import tab_recovery
from tabs.tab_wellness     import tab_wellness
from tabs.tab_analises     import tab_analises
from tabs.tab_aquecimento  import tab_aquecimento
from tabs.tab_corporal     import tab_corporal
from tabs.tab_padrao       import tab_padrao
from tabs.tab_ctl_kj       import tab_ctl_kj
from tabs.tab_cp_model     import tab_cp_model
from tabs.tab_fmt_tensor   import tab_fmt_tensor
# ❌ REMOVIDO: tab_hrv_analyzer agora em app_hrv.py (app independente)


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULOS CUSTOSOS COM CACHE (reduz CPU throttling)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner=False)
def _calc_alpha_polar_startup(ac_full_pickle):
    """
    eFTP regressão (Kalman CTLγ + OLS múltipla por modalidade).
    Custo: ~800ms (4 Kalman + 4 OLS). Cacheado 2h para evitar throttle.
    """
    import pickle, numpy as np
    from scipy import stats as _sp_stats
    from utils.data import kalman_ctlg as _kctlg
    
    ac_full = pickle.loads(ac_full_pickle)
    _alpha_cache = {}
    
    _gmap = {"Bike":0.250, "Row":0.900, "Ski":0.600, "Run":0.900}
    
    def _zslope(series, n=14):
        s = series.dropna().tail(n)
        if len(s) < 5: return 0.0
        sl, *_ = _sp_stats.linregress(np.arange(len(s), dtype=float), s.values.astype(float))
        return float(sl)

    for _mv in ["Bike","Row","Ski","Run"]:
        try:
            _z1c = next((c for c in ["z1_kj","Z1KJ","z1kj"] if c in ac_full.columns), None)
            _z2c = next((c for c in ["z2_kj","Z2KJ","z2kj"] if c in ac_full.columns), None)
            _z3c = next((c for c in ["z3_kj","Z3KJ","z3kj"] if c in ac_full.columns), None)
            _ce  = next((c for c in ["icu_eftp","eFTP","eftp"] if c in ac_full.columns), None)
            _cd  = next((c for c in ["date","Data"] if c in ac_full.columns), None)
            
            if not all([_z1c, _z2c, _z3c, _ce, _cd]): continue
            
            _ef = ac_full[ac_full['type']==_mv][[_cd,_ce,_z1c,_z2c,_z3c]].copy()
            _ef[_cd] = pd.to_datetime(_ef[_cd]).dt.normalize()
            _ef = (_ef.rename(columns={_cd:"Data",_ce:"eftp",
                                       _z1c:"z1",_z2c:"z2",_z3c:"z3"})
                        .sort_values("Data").drop_duplicates("Data").reset_index(drop=True))
            _ef[["eftp","z1","z2","z3"]] = _ef[["eftp","z1","z2","z3"]].apply(pd.to_numeric, errors="coerce")
            _ef = _ef.dropna(subset=["eftp"])
            _ef[["z1","z2","z3"]] = _ef[["z1","z2","z3"]].fillna(0)
            
            if len(_ef) < 10:
                _alpha_cache[_mv] = {'ok':False,'reason':f'{len(_ef)} sessões (<10)'}
                continue

            _gam = _gmap[_mv]
            _span = int(round(max(42.0*(1.0-_gam) + 7.0*_gam, 7.0)))

            _dr = pd.date_range(_ef["Data"].min(), pd.Timestamp.now().normalize(), freq="D")
            _ei = _ef.set_index("Data")
            _cz1 = _kctlg(_ei["z1"].reindex(_dr).fillna(0), tau=float(_span))
            _cz2 = _kctlg(_ei["z2"].reindex(_dr).fillna(0), tau=float(_span))
            _cz3 = _kctlg(_ei["z3"].reindex(_dr).fillna(0), tau=float(_span))
            _ef["cz1"] = _ef["Data"].map(_cz1.to_dict())
            _ef["cz2"] = _ef["Data"].map(_cz2.to_dict())
            _ef["cz3"] = _ef["Data"].map(_cz3.to_dict())
            _ef = _ef.dropna(subset=["cz1","cz2","cz3"])
            
            if len(_ef) < 10:
                _alpha_cache[_mv] = {'ok':False,'reason':'insuficientes após CTLγ'}
                continue

            _X = np.column_stack([_ef["cz3"].values.astype(float),
                                  _ef["cz2"].values.astype(float),
                                  _ef["cz1"].values.astype(float),
                                  np.ones(len(_ef))])
            _y = _ef["eftp"].values.astype(float)
            _coef, _, _, _ = np.linalg.lstsq(_X, _y, rcond=None)
            _a3,_a2,_a1,_intc = (float(_coef[0]),float(_coef[1]),float(_coef[2]),float(_coef[3]))
            _yp = _X @ _coef
            _r2 = float(1 - np.sum((_y-_yp)**2) / max(np.sum((_y-_y.mean())**2), 1e-9))

            _cz3n = float(_cz3.iloc[-1])
            _cz2n = float(_cz2.iloc[-1])
            _cz1n = float(_cz1.iloc[-1])
            _sl3 = _zslope(_cz3)
            _sl2 = _zslope(_cz2)
            _sl1 = _zslope(_cz1)
            _eftp_now_v = float(_ef["eftp"].iloc[-1])

            _delta_eftp_tendencia = _sl3 * 90 * _a3
            _delta_eftp_alvo = min(max(_delta_eftp_tendencia, 5.0), 15.0)
            _eftp_alvo = _eftp_now_v + _delta_eftp_alvo
            _eftp_3m = _eftp_alvo

            if abs(_a3) > 0.01:
                _cz3_alvo = (_eftp_alvo - _a2*_cz2n - _a1*_cz1n - _intc) / _a3
                _cz3_alvo = max(_cz3_alvo, _cz3n)
            else:
                _cz3_alvo = _cz3n * 1.10

            _l4w = _ef[_ef["Data"] >= pd.Timestamp.now().normalize()-pd.Timedelta(weeks=4)]
            _kj3 = float(_l4w["z3"].sum()/4) if len(_l4w) > 0 else 0
            _kj2 = float(_l4w["z2"].sum()/4) if len(_l4w) > 0 else 0
            _kj1 = float(_l4w["z1"].sum()/4) if len(_l4w) > 0 else 0

            _kj3_sem_alvo = float(_cz3_alvo * 7)
            _kj3_esta_sem = _kj3 + (_kj3_sem_alvo - _kj3) / 12.0
            _kj3_esta_sem = max(0, _kj3_esta_sem)

            _total_act = _kj3 + _kj2 + _kj1
            _kj2_esta_sem = _kj2 * 1.05 if _total_act > 0 else _kj2
            _kj1_esta_sem = _kj1 * 1.05 if _total_act > 0 else _kj1

            _alpha_cache[_mv] = {
                'ok': True,
                'alpha_z3': _a3, 'alpha_z2': _a2, 'alpha_z1': _a1,
                'r2': _r2,
                'eftp_now': _eftp_now_v,
                'cz3_now': _cz3n, 'cz2_now': _cz2n, 'cz1_now': _cz1n,
                'kj_z3_semana_actual': _kj3,
                'kj_z2_semana_actual': _kj2,
                'kj_z1_semana_actual': _kj1,
                'alvos': {'3m': {
                    'eftp_proj': _eftp_3m,
                    'delta_w': _eftp_3m - _eftp_now_v,
                    'kj_z3_semana': _kj3_esta_sem,
                    'kj_z2_semana': _kj2_esta_sem,
                    'kj_z1_semana': _kj1_esta_sem,
                    'kj_z3_sem_alvo': _kj3_sem_alvo,
                    'cz3_alvo': _cz3_alvo,
                }},
            }
        except Exception:
            continue

    return _alpha_cache if any(v.get('ok') for v in _alpha_cache.values()) else {}


@st.cache_data(ttl=7200, show_spinner=False)
def _calc_trimp_kj_startup(ac_full_pickle, wr_full_pickle):
    """
    dTRIMP/dKJ e eff_delta por modalidade.
    Custo: ~300ms (groupby + ewm × 4 + regressão). Cacheado 2h.
    """
    import pickle, numpy as np
    from scipy import stats as _sp_stats2
    
    ac_full = pickle.loads(ac_full_pickle)
    wr_full = pickle.loads(wr_full_pickle)
    
    result = {}
    try:
        for _me in ['Bike','Row','Ski','Run']:
            _s = ac_full[ac_full['type']==_me].copy()
            if len(_s) < 5: continue
            
            _s['Data'] = pd.to_datetime(_s['Data']).dt.normalize()
            _s = _s.sort_values('Data').drop_duplicates('Data').reset_index(drop=True)
            
            _s['trimp'] = (pd.to_numeric(_s.get('moving_time', 0), errors='coerce') / 60 *
                          pd.to_numeric(_s.get('rpe', 0), errors='coerce'))
            _s['kj'] = pd.to_numeric(_s.get('icu_joules', 0), errors='coerce') / 1000
            _s = _s.dropna(subset=['trimp', 'kj'])
            
            if len(_s) < 10: continue
            
            _s['trimp_span'] = _s['trimp'].ewm(span=28, adjust=False).mean()
            _s['kj_span'] = _s['kj'].ewm(span=28, adjust=False).mean()
            _s['eff'] = _s['trimp_span'] / (_s['kj_span'] + 0.1)
            
            _x_vals = np.arange(len(_s), dtype=float)
            _y_eff = _s['eff'].fillna(_s['eff'].mean()).values
            _sl_eff, _int_eff, _r_eff, _, _ = _sp_stats2.linregress(_x_vals, _y_eff)
            
            eff_delta_14d = _sl_eff * 14
            result[_me] = {
                'eff_trend_14d': float(eff_delta_14d),
                'eff_trend_pct': float(eff_delta_14d / max(_y_eff[-1], 0.01) * 100) if len(_y_eff) > 0 else 0,
            }
    except Exception:
        pass
    
    return result


def main():
    days_back, di, df_, mods_sel = render_sidebar()

    st.title("🏃 ATHELTICA Analytics Dashboard")
    st.caption(f"Período: {di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}  |  "
               f"Modalidades: {', '.join(mods_sel)}")

    with st.spinner("A carregar dados..."):
        wr                    = carregar_wellness(days_back)
        wr_full               = carregar_wellness(9999)
        ar_max                = carregar_atividades(9999)
        ar                    = carregar_atividades(days_back) if days_back < 9999 else ar_max
        dfs_annual, df_annual = carregar_annual()
        dc                    = carregar_corporal()

    if wr.empty and ar_max.empty:
        st.error("Não foi possível carregar dados. Verifica as credenciais e os URLs.")
        st.stop()

    wc       = preproc_wellness(wr)
    wc_full  = preproc_wellness(wr_full)
    ac_full  = preproc_ativ(ar_max)
    ac       = preproc_ativ(ar)

    st.session_state['da_full']  = ac_full
    st.session_state['mods_sel'] = mods_sel

    # ✨ Usar funções cached ao invés de recalcular
    import pickle
    ac_full_pickle = pickle.dumps(ac_full)
    wr_full_pickle = pickle.dumps(wc_full)
    
    try:
        alpha_cache = _calc_alpha_polar_startup(ac_full_pickle)
        if alpha_cache:
            st.session_state['alpha_polar_cache'] = alpha_cache
    except Exception:
        pass
    
    try:
        trimp_kj_info = _calc_trimp_kj_startup(ac_full_pickle, wr_full_pickle)
        if trimp_kj_info:
            st.session_state['trimp_kj_info'] = trimp_kj_info
    except Exception:
        pass  # silencioso — trimp_kj cacheado

    dw      = filtrar_datas(wc, di, df_)
    da      = filtrar_datas(ac, di, df_)
    da_filt = (da[da['type'].isin(mods_sel + ['WeightTraining'])]
               if len(da) > 0 and 'type' in da.columns else da)

    st.success(f"✅ {len(dw)} registos wellness  |  "
               f"{len(da_filt)} actividades  |  "
               f"Histórico PMC: {len(ac_full)} actividades")

    with st.expander("🔍 Diagnóstico de dados", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Atividades**")
            if len(ar) > 0:
                ar2 = ar.copy(); ar2['Data'] = ar2['Data'].astype(str)
                cs = [c for c in ['Data','type','name','moving_time','rpe','power_avg','icu_eftp'] if c in ar2.columns]
                st.dataframe(ar2[cs].sort_values('Data', ascending=False).head(5), hide_index=True)
        with c2:
            st.markdown("**Wellness**")
            if len(wr) > 0:
                wr2 = wr.copy(); wr2['Data'] = wr2['Data'].astype(str)
                cw = [c for c in ['Data','hrv','rhr','sleep_quality','fatiga','stress'] if c in wr2.columns]
                st.dataframe(wr2[cw].sort_values('Data', ascending=False).head(5), hide_index=True)

    (tab1, tab2, tab3, tab4, tab5, tab6, tab7,
     tab8, tab9, tab10, tab11, tab12, tab13, tab14) = st.tabs([
        "📊 Visão Geral", "📈 PMC",        "📦 Volume",     "⚡ eFTP",
        "❤️ HR & RPE",   "🔋 Recovery",   "🧘 Wellness",
        "🔬 Análises",   "🌡️ Aquecimento", "🧬 Corporal",   "🔄 Padrão",
        "⚗️ CTL vs KJ",  "🏁 CP Model",    "📐 FMT Tensor",
    ])
    # ❌ REMOVIDAS: 🧠 Correlações, 🫀 HRV Analyzer (agora em app_hrv.py)

    with tab1:  tab_visao_geral(dw, da_filt, di, df_, da_full=ac_full, wc_full=wc_full, dc=dc)
    with tab2:  tab_pmc(ac_full, wc=wc_full)
    with tab3:  tab_volume(da_filt, dw)
    with tab4:  tab_eftp(da_filt, mods_sel, ac_full, wc_full=wc_full)
    with tab5:  tab_zones(da_filt, mods_sel)
    with tab6:  tab_recovery(dw, da, wc_full=wc_full, da_full=ac_full)
    with tab7:  tab_wellness(dw, wc_full=wc_full)
    with tab8:  tab_analises(ac_full, dw, dfs_annual, df_annual)
    with tab9:  tab_aquecimento(dfs_annual, df_annual, di)
    with tab10: tab_corporal(dc, ac_full, wc=wc_full)
    with tab11: tab_padrao(ac_full, wc_full)
    with tab12: tab_ctl_kj(ac_full)
    with tab13: tab_cp_model(ac_full=ac_full)
    with tab14: tab_fmt_tensor(ac_full, wc=wc_full)
    # ❌ REMOVIDAS: tab_correlacoes, tab_hrv_analyzer (agora em app_hrv.py)


if __name__ == "__main__":
    main()
