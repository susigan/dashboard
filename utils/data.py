# MODIFICAÇÃO SEGURA para data.py
# Adiciona suporte a Railway SEM quebrar Streamlit Cloud

# PASSO 1: Adiciona esta linha APÓS linha 2 (import sys, os as _os):

import json

# PASSO 2: Substitui a função get_gc() (linhas 1583-1593) por isto:

def _get_gcp_credentials():
    """
    Carrega credenciais GCP com fallback seguro:
    1. Tenta st.secrets (Streamlit Cloud)
    2. Se falhar, tenta variáveis de ambiente (Railway)
    """
    # Primeiro tenta Streamlit Cloud
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except:
        pass
    
    # Depois tenta Railway (variáveis de ambiente)
    try:
        import os
        creds_json = os.getenv("GCP_SERVICE_ACCOUNT")
        if creds_json:
            return json.loads(creds_json)
    except:
        pass
    
    # Nenhuma credencial encontrada
    return None

@st.cache_resource
def get_gc():
    """Autentica Google Sheets com Service Account (Streamlit Cloud ou Railway)."""
    try:
        creds_dict = _get_gcp_credentials()
        
        if creds_dict is None:
            st.error("❌ Erro autenticação Google: Nenhuma credencial encontrada")
            st.info("Streamlit Cloud: Settings → Secrets\nRailway: Variables → GCP_SERVICE_ACCOUNT")
            return None
        
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Erro autenticação Google: {e}")
        return None
