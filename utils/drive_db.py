# FIX para drive_db.py - NameError: name 'st' is not defined

# PASSO 1: Verifica que streamlit está importado NO TOPO
# import streamlit as st

# PASSO 2: Adiciona esta função ANTES de _fresh_sh() (aproximadamente linha 64):

# ✅ SEM decorator na função auxiliar!
def _get_gcp_credentials_db():
    """Carrega credenciais GCP com fallback."""
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

# PASSO 3: A função _fresh_sh() fica normal (sem mudanças no decorator):
# Substitui APENAS as linhas da parte de autenticação (71-73):

def _fresh_sh():
    """Sempre cria nova conexão — evita cache stale."""
    try:
        from utils.data import get_gc
        gc = get_gc()
    except Exception:
        creds_dict = _get_gcp_credentials_db()
        if creds_dict is None:
            raise Exception("Nenhuma credencial GCP encontrada")
        creds = Credentials.from_service_account_info(creds_dict, scopes=_SCOPES)
        gc = gspread.authorize(creds)
    return gc.open_by_key(_SPREADSHEET_ID)
