# FIX para data.py - NameError: name 'st' is not defined

# PASSO 1: Verifica que streamlit está importado NO TOPO
# Procura linha com: import streamlit as st
# Se não existir, adiciona (geralmente já está em utils.config)

# PASSO 2: SUBSTITUI a função _get_gcp_credentials() E get_gc() (linhas ~1583-1593):

# ✅ SEM decorator na função auxiliar!
def _get_gcp_credentials():
    """Carrega credenciais GCP com fallback (Streamlit → Railway)."""
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except:
        pass
    try:
        import os
        creds_json = os.getenv("GCP_SERVICE_ACCOUNT")
        if creds_json:
            return json.loads(creds_json)
    except:
        pass
    return None

# ✅ COM decorator APENAS na função principal!
@st.cache_resource
def get_gc():
    """Autentica Google Sheets (Streamlit Cloud ou Railway)."""
    try:
        creds_dict = _get_gcp_credentials()
        if creds_dict is None:
            st.error("❌ Erro autenticação Google")
            return None
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Erro: {e}")
        return None
