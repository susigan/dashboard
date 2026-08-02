"""
ATHELTICA — HRV Analysis & Correlations
App com ERRO HANDLING MELHORADO (mostra erros reais!)
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys, os
import traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ATHELTICA — HRV Analysis",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 ATHELTICA — HRV Analysis")

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTS COM DEBUGGING COMPLETO
# ════════════════════════════════════════════════════════════════════════════════

# 1. Data_loader
try:
    from Data_loader import carregar_wellness, carregar_atividades
    st.success("✅ Data_loader carregado")
except Exception as e:
    st.error(f"❌ Erro importar Data_loader: {str(e)}")
    st.error(traceback.format_exc())
    st.stop()

# 2. drive_utils
try:
    from drive_utils import upload_resultado_drive, list_results_drive, download_resultado_drive
    st.success("✅ drive_utils carregado")
except Exception as e:
    st.warning(f"⚠️ Erro importar drive_utils: {str(e)}")
    upload_resultado_drive = None
    list_results_drive = None

# 3. Tabs HRV
try:
    st.info("⏳ Tentando importar tab_hrv_analyzer...")
    from tabs.tab_hrv_analyzer import tab_hrv_analyzer
    st.success("✅ tab_hrv_analyzer carregado")
except Exception as e:
    st.error(f"❌ Erro importar tab_hrv_analyzer: {str(e)}")
    st.error("**TRACEBACK COMPLETO:**")
    st.error(traceback.format_exc())
    tab_hrv_analyzer = None

# 4. Tabs Correlacoes
try:
    st.info("⏳ Tentando importar tab_correlacoes...")
    from tabs.tab_correlacoes import tab_correlacoes
    st.success("✅ tab_correlacoes carregado")
except Exception as e:
    st.error(f"❌ Erro importar tab_correlacoes: {str(e)}")
    st.error("**TRACEBACK COMPLETO:**")
    st.error(traceback.format_exc())
    tab_correlacoes = None

# Verificar se tabs estão disponíveis
tabs_available = (tab_hrv_analyzer is not None) and (tab_correlacoes is not None)

if not tabs_available:
    st.error("❌ Uma ou mais tabs falharam ao importar. Vê os erros acima!")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# CARREGAR DADOS
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200)
def load_data():
    try:
        wc = carregar_wellness(9999)
        ac = carregar_atividades(9999)
        return wc, ac
    except Exception as e:
        st.error(f"❌ Erro carregar dados: {e}")
        return None, None

st.markdown("---")
st.subheader("📊 Carregando dados...")

wc, ac = load_data()

if wc is not None:
    st.success(f"✅ Wellness: {len(wc)} registos")
else:
    st.error("❌ Erro ao carregar wellness")
    st.stop()

if ac is not None:
    st.success(f"✅ Atividades: {len(ac)} registos")
else:
    st.error("❌ Erro ao carregar atividades")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🔬 HRV Analysis")
st.sidebar.info("App dedicada para análises HRV")

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Google Drive Storage")

if upload_resultado_drive and list_results_drive:
    with st.sidebar.expander("📂 Histórico", expanded=False):
        try:
            results = list_results_drive(folder_name="SQLite")
            if results:
                st.write(f"✅ {len(results)} resultados")
            else:
                st.info("📭 Sem resultados ainda")
        except Exception as e:
            st.warning(f"⚠️ Erro Drive: {e}")
else:
    st.sidebar.info("⚠️ Drive storage não disponível")

# ════════════════════════════════════════════════════════════════════════════════
# CONTEÚDO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.success("✅ Tabs carregadas com sucesso!")

tabs = st.tabs(["🔬 Recovery Patterns", "🧠 Correlações"])

with tabs[0]:
    st.info("Loading: Recovery Patterns (HRV Analysis)...")
    try:
        tab_hrv_analyzer(wc, ac, wc_full=wc, da_full=ac)
    except Exception as e:
        st.error(f"❌ Erro em Recovery Patterns: {str(e)}")
        st.error("**TRACEBACK:**")
        st.error(traceback.format_exc())

with tabs[1]:
    st.info("Loading: Correlações & Impacto...")
    try:
        tab_correlacoes(ac, wc)
    except Exception as e:
        st.error(f"❌ Erro em Correlações: {str(e)}")
        st.error("**TRACEBACK:**")
        st.error(traceback.format_exc())

# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(f"ATHELTICA HRV | {len(wc)} wellness | {len(ac)} atividades")
