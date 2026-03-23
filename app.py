# ════════════════════════════════════════════════════════════════════════════════
# app.py — ATHELTICA Analytics Dashboard
# Entry point para o Streamlit Cloud.
# Cada aba está no seu próprio ficheiro em tabs/.
# Para editar uma aba: edita tabs/tab_<nome>.py e faz commit.
# ════════════════════════════════════════════════════════════════════════════════

import streamlit as st
from datetime import datetime, timedelta

from config import CORES, CORES_ATIV
from data_loader import (carregar_wellness, carregar_atividades,
                          preproc_wellness, preproc_ativ, filtrar_datas)

# Importar cada tab individualmente
from tabs.tab_visao_geral import tab_visao_geral
from tabs.tab_pmc         import tab_pmc
from tabs.tab_volume      import tab_volume
from tabs.tab_eftp        import tab_eftp
from tabs.tab_zones       import tab_zones
from tabs.tab_correlacoes import tab_correlacoes
from tabs.tab_recovery    import tab_recovery
from tabs.tab_wellness    import tab_wellness
from tabs.tab_analises    import tab_analises

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATHELTICA", page_icon="🏃",
    layout="wide", initial_sidebar_state="expanded"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.image("https://img.icons8.com/emoji/96/runner-emoji.png", width=60)
    st.sidebar.title("ATHELTICA")
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Filtros Globais")

    dias_op = {"30 dias": 30, "60 dias": 60, "90 dias": 90,
               "180 dias": 180, "1 ano": 365, "2 anos": 730}
    periodo = st.sidebar.selectbox("📅 Período", list(dias_op.keys()), index=2)
    days_back = dias_op[periodo]

    usar_custom = st.sidebar.checkbox("Datas manuais")
    if usar_custom:
        di = st.sidebar.date_input("Início", datetime.now().date() - timedelta(days=days_back))
        df_ = st.sidebar.date_input("Fim",   datetime.now().date())
    else:
        df_ = datetime.now().date()
        di  = df_ - timedelta(days=days_back)

    st.sidebar.markdown("---")
    st.sidebar.header("🏃 Modalidades")
    mods_all = ['Bike', 'Row', 'Run', 'Ski']
    mods_sel = st.sidebar.multiselect("Mostrar modalidades", mods_all, default=mods_all)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Recarregar dados"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"📅 {di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}")
    st.sidebar.caption(f"🕐 Atualizado: {datetime.now().strftime('%H:%M')}")

    return days_back, di, df_, mods_sel


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    days_back, di, df_, mods_sel = render_sidebar()

    st.title("🏃 ATHELTICA Analytics Dashboard")
    st.caption(f"Período: {di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}  |  "
               f"Modalidades: {', '.join(mods_sel)}")

    # ── Carregamento de dados ────────────────────────────────────────────────
    with st.spinner("A carregar dados..."):
        wr      = carregar_wellness(days_back)
        # PMC precisa sempre do histório máximo para CTL/ATL convergirem
        ar_full = carregar_atividades(730)
        ar      = carregar_atividades(days_back) if days_back < 730 else ar_full

    if wr.empty and ar_full.empty:
        st.error("Não foi possível carregar dados. Verifica as credenciais e os URLs.")
        st.stop()

    # ── Preprocessing ────────────────────────────────────────────────────────
    wc       = preproc_wellness(wr)
    ac_full  = preproc_ativ(ar_full)   # histórico completo para PMC e Análises
    ac       = preproc_ativ(ar)        # período filtrado

    # Guardar histórico completo no session_state (usado pelo tab_pmc)
    st.session_state['da_full'] = ac_full

    dw      = filtrar_datas(wc, di, df_)
    da      = filtrar_datas(ac, di, df_)
    da_filt = (da[da['type'].isin(mods_sel + ['WeightTraining'])]
               if len(da) > 0 and 'type' in da.columns else da)

    st.success(f"✅ {len(dw)} registros wellness  |  "
               f"{len(da_filt)} atividades ({di.strftime('%d/%m/%y')}→{df_.strftime('%d/%m/%y')})  |  "
               f"Histórico PMC: {len(ac_full)} atividades")

    # ── Diagnóstico (expansível) ─────────────────────────────────────────────
    with st.expander("🔍 Diagnóstico de dados", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Atividades (RAW — período filtrado)**")
            if len(ar) > 0:
                ar2 = ar.copy()
                ar2['Data'] = ar2['Data'].astype(str)
                cs = [c for c in ['Data', 'type', 'name', 'moving_time', 'rpe', 'power_avg', 'icu_eftp'] if c in ar2.columns]
                st.write(f"Total: {len(ar)} | Datas: {ar2['Data'].min()[:10]} → {ar2['Data'].max()[:10]}")
                st.dataframe(ar2[cs].sort_values('Data', ascending=False).head(5), hide_index=True)
        with c2:
            st.markdown("**Wellness (RAW)**")
            if len(wr) > 0:
                wr2 = wr.copy()
                wr2['Data'] = wr2['Data'].astype(str)
                cw = [c for c in ['Data', 'hrv', 'rhr', 'sleep_quality', 'fatiga', 'stress'] if c in wr2.columns]
                st.write(f"Total: {len(wr)} | Datas: {wr2['Data'].min()[:10]} → {wr2['Data'].max()[:10]}")
                st.dataframe(wr2[cw].sort_values('Data', ascending=False).head(5), hide_index=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Visão Geral",
        "📈 PMC",
        "📦 Volume",
        "⚡ eFTP",
        "❤️ HR & RPE",
        "🧠 Correlações",
        "🔋 Recovery",
        "🧘 Wellness",
        "🔬 Análises",
    ])

    with tab1: tab_visao_geral(dw, da_filt, di, df_)
    with tab2: tab_pmc(da_filt)
    with tab3: tab_volume(da_filt, dw)
    with tab4: tab_eftp(da_filt, mods_sel)
    with tab5: tab_zones(da_filt, mods_sel)
    with tab6: tab_correlacoes(da_filt, dw)
    with tab7: tab_recovery(dw)
    with tab8: tab_wellness(dw)
    with tab9: tab_analises(ac_full, dw)   # usa histórico completo


if __name__ == "__main__":
    main()
