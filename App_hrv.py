"""
ATHELTICA — HRV Analysis & Correlations
Versão que MOSTRA o erro real em vez de esconder!
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
# IMPORTS COM DEBUG TOTAL
# ════════════════════════════════════════════════════════════════════════════════

print("="*80)
print("INICIANDO APP_HRV — DEBUG MODE")
print("="*80)

try:
    from Data_loader import carregar_wellness, carregar_atividades
    st.success("✅ Data_loader OK")
    print("✅ Data_loader importado")
except Exception as e:
    st.error(f"❌ Data_loader: {e}")
    traceback.print_exc()
    st.stop()

try:
    from drive_utils import upload_resultado_drive, list_results_drive, download_resultado_drive
    st.success("✅ drive_utils OK")
    print("✅ drive_utils importado")
except Exception as e:
    st.warning(f"⚠️ drive_utils: {e}")
    print(f"⚠️ drive_utils: {e}")

try:
    from utils.config import CORES, CORES_ATIV, TYPE_MAP, VALID_TYPES
    from utils.data import preproc_wellness, preproc_ativ
    st.success("✅ utils (config + data) OK")
    print("✅ utils importados")
except Exception as e:
    st.error(f"❌ utils: {e}")
    traceback.print_exc()
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# TABS — COM ERRO COMPLETO
# ────────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📦 Tentando importar tabs...")

tab_hrv_analyzer = None
tab_correlacoes = None
tabs_available = False

# TAB 1: tab_hrv_analyzer
st.info("⏳ Importando: tab_hrv_analyzer...")
try:
    print("\n>>> Tentando importar tab_hrv_analyzer...")
    from tabs.tab_hrv_analyzer import tab_hrv_analyzer
    st.success("✅ tab_hrv_analyzer carregado")
    print("✅ tab_hrv_analyzer importado com sucesso")
except Exception as e:
    st.error(f"❌ tab_hrv_analyzer FALHOU!")
    st.error(f"**Erro:** {str(e)}")
    print(f"❌ ERRO em tab_hrv_analyzer:")
    print(traceback.format_exc())
    with st.expander("📋 Ver traceback completo"):
        st.code(traceback.format_exc())

# TAB 2: tab_correlacoes
st.info("⏳ Importando: tab_correlacoes...")
try:
    print("\n>>> Tentando importar tab_correlacoes...")
    from tabs.tab_correlacoes import tab_correlacoes
    st.success("✅ tab_correlacoes carregado")
    print("✅ tab_correlacoes importado com sucesso")
    tabs_available = True
except Exception as e:
    st.error(f"❌ tab_correlacoes FALHOU!")
    st.error(f"**Erro:** {str(e)}")
    print(f"❌ ERRO em tab_correlacoes:")
    print(traceback.format_exc())
    with st.expander("📋 Ver traceback completo"):
        st.code(traceback.format_exc())

if tab_hrv_analyzer and tab_correlacoes:
    tabs_available = True
    st.success("✅ Ambas as tabs carregadas com sucesso!")
    print("✅ Ambas as tabs OK")
else:
    st.warning("⚠️ Uma ou mais tabs falharam ao importar")
    print("⚠️ Tabs falharam")

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
        st.error(f"Erro carregar dados: {e}")
        return None, None

st.markdown("---")
st.subheader("📊 Carregando dados...")

wc, ac = load_data()

if wc is not None:
    st.success(f"✅ Wellness: {len(wc)} registos")
else:
    st.stop()

if ac is not None:
    st.success(f"✅ Atividades: {len(ac)} registos")
else:
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🔬 HRV Analysis")
st.sidebar.info("App para análise HRV")

# ════════════════════════════════════════════════════════════════════════════════
# CONTEÚDO
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")

if tabs_available and tab_hrv_analyzer and tab_correlacoes:
    st.success("✅ Renderizando com tabs completas!")
    tabs = st.tabs(["🔬 Recovery Patterns", "🧠 Correlações"])
    
    with tabs[0]:
        try:
            tab_hrv_analyzer(wc, ac, wc_full=wc, da_full=ac)
        except Exception as e:
            st.error(f"❌ Erro em tab_hrv_analyzer: {e}")
            with st.expander("📋 Traceback"):
                st.code(traceback.format_exc())
    
    with tabs[1]:
        try:
            tab_correlacoes(ac, wc)
        except Exception as e:
            st.error(f"❌ Erro em tab_correlacoes: {e}")
            with st.expander("📋 Traceback"):
                st.code(traceback.format_exc())
else:
    st.warning("⚠️ Tabs não disponíveis. Mostrando dados brutos...")
    
    st.subheader("📊 Wellness Data")
    st.write(f"{len(wc)} registos")
    st.dataframe(wc.head(20), use_container_width=True)
    
    st.subheader("⚡ Activities Data")
    st.write(f"{len(ac)} registos")
    st.dataframe(ac.head(20), use_container_width=True)

st.markdown("---")
st.caption(f"DEBUG: tabs_available={tabs_available}")
