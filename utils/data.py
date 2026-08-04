# VERSÃO COM DEBUG - data.py

# NO TOPO, DEPOIS DOS IMPORTS:

def _get_gcp_credentials():
    """Carrega credenciais GCP com fallback e DEBUG."""
    import os
    import sys
    
    # 1. Tenta st.secrets (Streamlit Cloud)
    try:
        if "gcp_service_account" in st.secrets:
            print("✅ DEBUG: Credenciais encontradas em st.secrets", file=sys.stderr)
            return dict(st.secrets["gcp_service_account"])
    except Exception as e:
        print(f"ℹ️ DEBUG: st.secrets falhou: {e}", file=sys.stderr)
    
    # 2. Tenta variável de ambiente (Railway)
    try:
        creds_json = os.getenv("GCP_SERVICE_ACCOUNT")
        if creds_json:
            print(f"✅ DEBUG: Variável GCP_SERVICE_ACCOUNT encontrada (tamanho: {len(creds_json)})", file=sys.stderr)
            creds_dict = json.loads(creds_json)
            print("✅ DEBUG: JSON decodificado com sucesso", file=sys.stderr)
            return creds_dict
        else:
            print("❌ DEBUG: Variável GCP_SERVICE_ACCOUNT é vazia ou None", file=sys.stderr)
    except Exception as e:
        print(f"❌ DEBUG: Erro ao ler variável de ambiente: {e}", file=sys.stderr)
    
    print("❌ DEBUG: NENHUMA credencial encontrada (st.secrets E GCP_SERVICE_ACCOUNT)", file=sys.stderr)
    return None

@st.cache_resource
def get_gc():
    """Autentica Google Sheets (Streamlit Cloud ou Railway) COM DEBUG."""
    try:
        creds_dict = _get_gcp_credentials()
        
        if creds_dict is None:
            st.error("❌ Erro autenticação Google: Nenhuma credencial encontrada")
            st.info("**Streamlit Cloud:** Settings → Secrets → gcp_service_account\n**Railway:** Variables → GCP_SERVICE_ACCOUNT (JSON completo)")
            return None
        
        print(f"✅ DEBUG: Criando credenciais com project_id: {creds_dict.get('project_id', 'ERRO')}", file=sys.stderr)
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        gc = gspread.authorize(creds)
        print("✅ DEBUG: Autenticação Google bem-sucedida!", file=sys.stderr)
        return gc
    except Exception as e:
        print(f"❌ DEBUG: Erro na autenticação: {e}", file=sys.stderr)
        st.error(f"❌ Erro autenticação: {e}")
        return None
