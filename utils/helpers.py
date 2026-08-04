# MODIFICAÇÃO SEGURA para helpers.py
# Adiciona suporte a Railway SEM quebrar Streamlit Cloud

# PASSO 1: Adiciona estes imports (após imports existentes):

import json

# PASSO 2: Adiciona esta função ANTES da função get_gc():

def _get_gcp_credentials_helpers():
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

# PASSO 3: Substitui a função get_gc() completamente por:

@st.cache_resource
def get_gc():
    """Autentica Google Sheets com Service Account (Streamlit Cloud ou Railway)."""
    try:
        creds_dict = _get_gcp_credentials_helpers()
        
        if creds_dict is None:
            st.error("❌ Erro autenticação Google: Nenhuma credencial encontrada")
            return None
        
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Erro autenticação Google: {e}")
        return None
