from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
from datetime import datetime, timedelta
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


def render_sidebar():
    st.sidebar.image("https://img.icons8.com/emoji/96/runner-emoji.png", width=60)
    st.sidebar.title("ATHELTICA")
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Filtros Globais")

    dias_op = {"30 dias": 30, "60 dias": 60, "90 dias": 90,
               "180 dias": 180, "1 ano": 365, "2 anos": 730,
               "3 anos": 1095, "5 anos": 1825, "Todo histórico": 9999}
    periodo = st.sidebar.selectbox("📅 Período", list(dias_op.keys()), index=2)
    days_back = dias_op[periodo]

    usar_custom = st.sidebar.checkbox("📅 Datas manuais")
    if usar_custom:
        di  = st.sidebar.date_input("Início", datetime(2017, 1, 1).date())
        df_ = st.sidebar.date_input("Fim",    datetime.now().date())
        # days_back para carregar: diferença em dias + margem
        days_back = (df_ - di).days + 30
    else:
        df_ = datetime.now().date()
        di  = df_ - timedelta(days=min(days_back, 9999))

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
