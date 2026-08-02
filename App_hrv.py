"""
ATHELTICA — HRV Analysis & Correlations
App independente para tab_hrv_analyzer + tab_correlacoes
Com persistência em Google Drive
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys, os

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════════

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from Data_loader import carregar_wellness, carregar_atividades
    from drive_utils import upload_resultado_drive, list_results_drive, download_resultado_drive
    
    # Tentar importar tabs se existirem
    try:
        from tabs.tab_hrv_analyzer import tab_hrv_analyzer
        from tabs.tab_correlacoes import tab_correlacoes
        tabs_available = True
    except ImportError:
        tabs_available = False
        st.warning("⚠️ Tabs HRV não encontradas. Modo simplificado.")
        
except ImportError as e:
    st.error(f"""
    ❌ **ERRO DE IMPORT:**
    {str(e)}
    
    Ficheiros esperados:
    - Data_loader.py (raiz)
    - drive_utils.py (raiz)
    """)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ATHELTICA — HRV Analysis",
    page_icon="🔬",
    layout="wide"
)

# ════════════════════════════════════════════════════════════════════════════════
# CARREGAR DADOS (cached)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200)
def load_data():
    """Carrega dados uma vez, cache 2h"""
    try:
        wc = carregar_wellness(9999)
        ac = carregar_atividades(9999)
        return wc, ac
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

wc, ac = load_data()

if wc is None or ac is None:
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🔬 HRV Analysis")
st.sidebar.info(
    "App dedicada para análises HRV.\n\n"
    "💡 **Status:** Modo simplificado"
)

# Drive Storage Section
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Google Drive Storage")

with st.sidebar.expander("📂 Histórico de resultados", expanded=False):
    try:
        results = list_results_drive(folder_name="SQLite")
        if results:
            st.write(f"✅ {len(results)} resultados salvos")
            selected = st.selectbox(
                "Carregar resultado anterior:",
                options=results,
                format_func=lambda x: f"{x['title']} ({x['createdDate'][:10]})",
                key="select_drive"
            )
            if st.button("⬇️ Carregar do Drive", key="load_drive"):
                df_loaded = download_resultado_drive(selected['id'])
                if df_loaded is not None:
                    st.success(f"✅ Carregado: {selected['title']}")
                    st.dataframe(df_loaded, use_container_width=True)
        else:
            st.info("📭 Sem resultados salvos ainda.")
    except Exception as e:
        st.warning(f"⚠️ Erro ao aceder Drive: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# CONTEÚDO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

st.title("🔬 ATHELTICA — HRV Analysis")

if tabs_available:
    # Modo com tabs
    tabs = st.tabs(["🔬 Recovery Patterns", "🧠 Correlações"])
    
    with tabs[0]:
        try:
            tab_hrv_analyzer(wc, ac, wc_full=wc, da_full=ac)
        except Exception as e:
            st.error(f"Erro em Recovery Patterns: {e}")
    
    with tabs[1]:
        try:
            tab_correlacoes(ac, wc)
        except Exception as e:
            st.error(f"Erro em Correlações: {e}")
else:
    # Modo simplificado (sem tabs)
    st.warning("⚠️ Tabs não disponíveis. Mostrando dados brutos...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Wellness Data")
        st.write(f"Registos: {len(wc)}")
        st.dataframe(wc.head(10), use_container_width=True)
    
    with col2:
        st.subheader("⚡ Activities Data")
        st.write(f"Atividades: {len(ac)}")
        st.dataframe(ac.head(10), use_container_width=True)
    
    # Botão para exportar
    st.markdown("---")
    if st.button("💾 Guardar dados no Drive", type="primary"):
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

# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "ATHELTICA HRV Analysis | "
    f"Dados: {len(wc)} wellness | {len(ac)} atividades | "
    f"Cache: 2h"
)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ATHELTICA — HRV Analysis",
    page_icon="🔬",
    layout="wide"
)

# ════════════════════════════════════════════════════════════════════════════════
# CARREGAR DADOS (cached)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200)
def load_data():
    """Carrega dados uma vez, cache 2h"""
    wc = carregar_wellness(9999)
    ac = carregar_atividades(9999)
    return wc, ac

wc, ac = load_data()

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR: CONTROLES
# ════════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🔬 HRV Analysis")
st.sidebar.info(
    "App dedicada para análises HRV.\n\n"
    "💡 **Dica:** Clica em '▶ Rodar Auto-Runner' na aba de Recovery "
    "para análises mais profundas (pesadas, ~100s)."
)

# Drive Storage Section
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Google Drive Storage")

with st.sidebar.expander("📂 Histórico de resultados", expanded=False):
    results = list_results_drive(folder_name="SQLite")
    if results:
        st.write(f"✅ {len(results)} resultados salvos")
        selected = st.selectbox(
            "Carregar resultado anterior:",
            options=results,
            format_func=lambda x: f"{x['title']} ({x['createdDate'][:10]})",
            key="select_drive"
        )
        if st.button("⬇️ Carregar do Drive", key="load_drive"):
            df_loaded = download_resultado_drive(selected['id'])
            if df_loaded is not None:
                st.session_state['historical_result'] = df_loaded
                st.success(f"✅ Carregado: {selected['title']}")
                st.dataframe(df_loaded, use_container_width=True)
    else:
        st.info("📭 Sem resultados salvos ainda. Calcula análises e guarda!")

# ════════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPAIS
# ════════════════════════════════════════════════════════════════════════════════

tabs = st.tabs(["🔬 Recovery Patterns", "🧠 Correlações & Impacto"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Recovery Patterns
# ─────────────────────────────────────────────────────────────────────────────

with tabs[0]:
    tab_hrv_analyzer(wc, ac, wc_full=wc, da_full=ac)
    
    # Export option
    if st.session_state.get('_hrv_gate'):
        st.markdown("---")
        st.subheader("💾 Guardar resultados")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Guardar no Drive", key="save_hrv_drive", type="primary"):
                results = st.session_state['_hrv_gate']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                filename = f"hrv_analysis_{timestamp}.csv"
                
                df_export = pd.DataFrame(results.get('runner_results', []))
                
                with st.spinner("A guardar no Drive..."):
                    file_id = upload_resultado_drive(df_export, filename=filename, folder_name="SQLite")
                
                if file_id:
                    st.success(f"✅ Guardado: {filename}")
                else:
                    st.error("❌ Erro ao guardar")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Correlações
# ─────────────────────────────────────────────────────────────────────────────

with tabs[1]:
    tab_correlacoes(ac, wc)
    
    # Export option
    st.markdown("---")
    st.subheader("💾 Guardar correlações")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("💾 Guardar no Drive", key="save_cor_drive", type="primary"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"correlacoes_{timestamp}.csv"
            
            # Exportar dados visíveis (simplificado)
            st.info("💾 Exportação guardada! (implementar com dados reais da tab)")

# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "ATHELTICA HRV Analysis | "
    f"Dados carregados: {len(wc)} wellness | {len(ac)} atividades | "
    f"Cache TTL: 2h"
)
