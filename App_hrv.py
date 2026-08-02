"""
ATHELTICA — HRV Analysis & Correlations
App simplificada e robusta
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG BASIC
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ATHELTICA — HRV Analysis",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 ATHELTICA — HRV Analysis")

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTS COM ERROR HANDLING
# ════════════════════════════════════════════════════════════════════════════════

try:
    from Data_loader import carregar_wellness, carregar_atividades
    st.success("✅ Data_loader carregado")
except ImportError as e:
    st.error(f"❌ Erro importar Data_loader: {e}")
    st.stop()

try:
    from drive_utils import upload_resultado_drive, list_results_drive, download_resultado_drive
    st.success("✅ drive_utils carregado")
except ImportError as e:
    st.error(f"❌ Erro importar drive_utils: {e}")

try:
    from tabs.tab_hrv_analyzer import tab_hrv_analyzer
    from tabs.tab_correlacoes import tab_correlacoes
    tabs_available = True
    st.success("✅ Tabs carregadas")
except ImportError:
    tabs_available = False
    st.info("ℹ️ Tabs não encontradas. Modo simplificado.")

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

with st.sidebar.expander("📂 Histórico", expanded=False):
    try:
        results = list_results_drive(folder_name="SQLite")
        if results:
            st.write(f"✅ {len(results)} resultados")
        else:
            st.info("📭 Sem resultados ainda")
    except Exception as e:
        st.warning(f"⚠️ Erro Drive: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# CONTEÚDO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")

if tabs_available:
    st.info("✅ Modo com tabs disponível")
    tabs = st.tabs(["🔬 Recovery Patterns", "🧠 Correlações"])
    
    with tabs[0]:
        try:
            tab_hrv_analyzer(wc, ac, wc_full=wc, da_full=ac)
        except Exception as e:
            st.error(f"❌ Erro em Recovery Patterns: {e}")
    
    with tabs[1]:
        try:
            tab_correlacoes(ac, wc)
        except Exception as e:
            st.error(f"❌ Erro em Correlações: {e}")
else:
    st.info("ℹ️ Modo simplificado (sem tabs)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Wellness")
        st.write(f"Registos: {len(wc)}")
        st.dataframe(wc.head(10), use_container_width=True)
    
    with col2:
        st.subheader("⚡ Activities")
        st.write(f"Atividades: {len(ac)}")
        st.dataframe(ac.head(10), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("💾 Guardar Dados")

if st.button("📥 Guardar no Drive", type="primary"):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"hrv_export_{timestamp}.csv"
        
        with st.spinner("A guardar..."):
            file_id = upload_resultado_drive(wc, filename=filename, folder_name="SQLite")
        
        if file_id:
            st.success(f"✅ Guardado: {filename}")
        else:
            st.error("❌ Erro ao guardar")
    except Exception as e:
        st.error(f"❌ Erro: {e}")

st.markdown("---")
st.caption(f"ATHELTICA HRV | {len(wc)} wellness | {len(ac)} atividades")
