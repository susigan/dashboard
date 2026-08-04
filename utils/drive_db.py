# MODIFICAÇÃO SEGURA para drive_db.py
# Adiciona suporte a Railway SEM quebrar Streamlit Cloud

# PASSO 1: Adiciona estes imports NO TOPO (após imports existentes):

import json

# PASSO 2: Adiciona esta função ANTES da função _fresh_sh() (aproximadamente linha 64):

def _get_gcp_credentials_db():
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

# PASSO 3: Substitui as linhas 71-73 em _fresh_sh() por:

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
