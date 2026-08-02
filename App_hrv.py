"""
ATHELTICA — HRV Analysis & Correlations
COM DEBUG COMPLETO DOS IMPORTS
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys, os
import traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

st.set_page_config(
    page_title="ATHELTICA — HRV Analysis",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 ATHELTICA — HRV Analysis")

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTS COM DEBUG
# ════════════════════════════════════════════════════════════════════════════════

tab_hrv_analyzer = None
tab_correlacoes = None
tabs_available = False

# IMPORTS PRINCIPAIS
try:
    from Data_loader import carregar_wellness, carregar_atividades
    from drive_utils import upload_resultado_drive, list_results_drive, download_resultado_drive
    from utils.config import CORES, CORES_ATIV, TYPE_MAP, VALID_TYPES
    from utils.data import preproc_wellness, preproc_ativ
    st.success("✅ Imports principais OK")
except Exception as e:
    st.error(f"❌ Erro imports principais: {e}")
    st.error(traceback.format_exc())
    st.stop()

# TAB: tab_hrv_analyzer
st.info("⏳ Importando tab_hrv_analyzer...")
try:
    from tabs.tab_hrv_analyzer import tab_hrv_analyzer
    st.success("✅ tab_hrv_analyzer OK")
except Exception as e:
    st.error(f"❌ tab_hrv_analyzer FALHOU!")
    st.error(f"**Erro:** {str(e)}")
    st.error("**Traceback completo:**")
    st.code(traceback.format_exc())
    tab_hrv_analyzer = None

# TAB: tab_correlacoes
st.info("⏳ Importando tab_correlacoes...")
try:
    from tabs.tab_correlacoes import tab_correlacoes
    st.success("✅ tab_correlacoes OK")
except Exception as e:
    st.error(f"❌ tab_correlacoes FALHOU!")
    st.error(f"**Erro:** {str(e)}")
    st.error("**Traceback completo:**")
    st.code(traceback.format_exc())
    tab_correlacoes = None

# ════════════════════════════════════════════════════════════════════════════════
# VERIFICAÇÃO FINAL
# ════════════════════════════════════════════════════════════════════════════════

if tab_hrv_analyzer and tab_correlacoes:
    tabs_available = True
    st.success("✅✅✅ AMBAS AS TABS CARREGADAS COM SUCESSO!")
else:
    st.error("❌ Uma ou mais tabs falharam. Vê os erros acima!")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# CARREGAR DADOS
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200)
def load_data():
    wc = carregar_wellness(9999)
    ac = carregar_atividades(9999)
    return wc, ac

wc, ac = load_data()
st.success(f"✅ Dados: {len(wc)} wellness, {len(ac)} atividades")

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🔬 HRV Analysis")

# ════════════════════════════════════════════════════════════════════════════════
# CONTEÚDO
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.success("✅ Renderizando tabs...")

tabs = st.tabs(["🔬 Recovery Patterns", "🧠 Correlações"])

with tabs[0]:
    try:
        tab_hrv_analyzer(wc, ac, wc_full=wc, da_full=ac)
    except Exception as e:
        st.error(f"❌ Erro em tab_hrv_analyzer: {e}")
        st.code(traceback.format_exc())

with tabs[1]:
    try:
        tab_correlacoes(ac, wc)
    except Exception as e:
        st.error(f"❌ Erro em tab_correlacoes: {e}")
        st.code(traceback.format_exc())

st.markdown("---")
st.caption(f"ATHELTICA HRV | {len(wc)} wellness | {len(ac)} atividades")
