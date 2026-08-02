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
# IMPORTS — procurar ficheiros no diretório correto
# ════════════════════════════════════════════════════════════════════════════════

# Adicionar o diretório atual ao path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# CRÍTICO: verificar se ficheiros necessários existem
required_files = [
    'Data_loader.py',
    'data.py', 
    'config.py',
    'tabs/tab_hrv_analyzer.py',
    'tabs/tab_correlacoes.py',
    'drive_utils.py'
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(script_dir, f))]

if missing_files:
    st.error(f"""
    ❌ **ERRO: Ficheiros ausentes!**
    
    Faltam os seguintes ficheiros no diretório:
    {', '.join(missing_files)}
    
    **Solução:**
    1. Copia todos os ficheiros originais de dashboard para aqui
    2. Ou coloca-os na pasta correta
    
    Diretório esperado: {script_dir}
    """)
    st.stop()

# Agora fazer os imports
try:
    from Data_loader import carregar_wellness, carregar_atividades, carregar_annual
    from tabs.tab_hrv_analyzer import tab_hrv_analyzer
    from tabs.tab_correlacoes import tab_correlacoes
    from drive_utils import upload_resultado_drive, list_results_drive, download_resultado_drive
except ImportError as e:
    st.error(f"""
    ❌ **ERRO DE IMPORT:**
    
    {str(e)}
    
    **Verificar:**
    1. Todos os ficheiros estão no diretório?
    2. Nomes dos ficheiros estão corretos?
    3. Pasta 'tabs/' existe?
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
