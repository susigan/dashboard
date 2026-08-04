# FIX para helpers.py - NameError: name 'st' is not defined

# PASSO 1: Verifica que streamlit está importado NO TOPO

# PASSO 2: SUBSTITUI a função _get_gcp_credentials_helpers() E get_gc():

# ✅ SEM decorator na função auxiliar!
def _get_gcp_credentials_helpers():
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

# ✅ COM decorator APENAS na função principal!
@st.cache_resource
def get_gc():
    """Autentica Google Sheets (Streamlit Cloud ou Railway)."""
    try:
        creds_dict = _get_gcp_credentials_helpers()
        if creds_dict is None:
            st.error("❌ Erro autenticação")
            return None
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Erro: {e}")
        return None
