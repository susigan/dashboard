# ══════════════════════════════════════════════════════════════════════════════
# app.py — ATHELTICA Dashboard
# Ponto de entrada. Toda a lógica está em tabs/ e utils/
# ══════════════════════════════════════════════════════════════════════════════

import sys, os
# Garante que o directório do app está no path — necessário no Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from tabs.tab_correlacoes  import tab_correlacoes
from tabs.tab_recovery     import tab_recovery
from tabs.tab_wellness     import tab_wellness
from tabs.tab_analises     import tab_analises
from tabs.tab_aquecimento  import tab_aquecimento
from tabs.tab_corporal     import tab_corporal
from tabs.tab_padrao       import tab_padrao
from tabs.tab_ctl_kj       import tab_ctl_kj
from tabs.tab_cp_model     import tab_cp_model


def main():
    days_back, di, df_, mods_sel = render_sidebar()

    st.title("🏃 ATHELTICA Analytics Dashboard")
    st.caption(f"Período: {di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}  |  "
               f"Modalidades: {', '.join(mods_sel)}")

    with st.spinner("A carregar dados..."):
        wr                    = carregar_wellness(days_back)
        ar_max                = carregar_atividades(9999)
        ar                    = carregar_atividades(days_back) if days_back < 9999 else ar_max
        dfs_annual, df_annual = carregar_annual()
        dc                    = carregar_corporal()

    if wr.empty and ar_max.empty:
        st.error("Não foi possível carregar dados. Verifica as credenciais e os URLs.")
        st.stop()

    wc       = preproc_wellness(wr)
    ac_full  = preproc_ativ(ar_max)
    ac       = preproc_ativ(ar)

    st.session_state['da_full']  = ac_full
    st.session_state['mods_sel'] = mods_sel

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

    (tab1,tab2,tab3,tab4,tab5,tab6,tab7,
     tab8,tab9,tab10,tab11,tab12,tab13,tab14) = st.tabs([
        "📊 Visão Geral","📈 PMC","📦 Volume","⚡ eFTP","❤️ HR & RPE",
        "🧠 Correlações","🔋 Recovery","🧘 Wellness","🔬 Análises",
        "🌡️ Aquecimento","🧬 Corporal","🔄 Padrão","⚗️ CTL vs KJ","🏁 CP Model",
    ])

    with tab1:  tab_visao_geral(dw, da_filt, di, df_, da_full=ac_full, wc_full=wc, dc=dc)
    with tab2:  tab_pmc(da_filt)
    with tab3:  tab_volume(da_filt, dw)
    with tab4:  tab_eftp(da_filt, mods_sel, ac_full)
    with tab5:  tab_zones(da_filt, mods_sel)
    with tab6:  tab_correlacoes(ac_full, wc)
    with tab7:  tab_recovery(dw, da)
    with tab8:  tab_wellness(dw)
    with tab9:  tab_analises(ac_full, dw, dfs_annual, df_annual)
    with tab10: tab_aquecimento(dfs_annual, df_annual, di)
    with tab11: tab_corporal(dc, ac_full)
    with tab12: tab_padrao(ac_full, wc)
    with tab13: tab_ctl_kj(ac_full)
    with tab14: tab_cp_model()


if __name__ == "__main__":
    main()
