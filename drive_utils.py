"""
Google Drive Utils — com Fallback (sem PyDrive)
Funciona mesmo que PyDrive não esteja instalado
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Tentar importar PyDrive, se não existir, usar fallback
try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    PYDRIVE_AVAILABLE = True
except ImportError:
    PYDRIVE_AVAILABLE = False
    st.warning("⚠️ PyDrive não disponível. Modo offline.")

# ════════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE CONNECTION (com fallback)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_drive_connection():
    """Tenta conectar ao Google Drive. Retorna None se PyDrive não disponível."""
    
    if not PYDRIVE_AVAILABLE:
        st.warning("⚠️ PyDrive não instalado. Drive storage desactivado.")
        return None
    
    try:
        gauth = GoogleAuth()
        
        # Tentar usar credentials do Streamlit secrets
        try:
            creds_dict = dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
            gauth.auth_method = 'service_account'
            gauth.credentials = creds_dict
        except:
            st.warning("⚠️ Google credentials não encontradas em secrets.")
            return None
        
        drive = GoogleDrive(gauth)
        st.success("✅ Google Drive conectado")
        return drive
        
    except Exception as e:
        st.warning(f"⚠️ Erro Google Drive: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# UPLOAD (com fallback)
# ════════════════════════════════════════════════════════════════════════════════

def upload_resultado_drive(dataframe, filename, folder_name="SQLite"):
    """Upload ficheiro para Google Drive. Retorna None se falhar."""
    
    if not PYDRIVE_AVAILABLE:
        st.warning("⚠️ PyDrive não disponível. Ficheiro NÃO foi guardado no Drive.")
        return None
    
    try:
        drive = get_drive_connection()
        if drive is None:
            return None
        
        # Procurar pasta
        folder_list = drive.ListFile({'q': f"title='{folder_name}' and trashed=false"}).GetList()
        
        if not folder_list:
            st.error(f"❌ Pasta '{folder_name}' não encontrada no Drive")
            return None
        
        folder_id = folder_list[0]['id']
        
        # Guardar CSV temporariamente
        csv_path = f"/tmp/{filename}"
        dataframe.to_csv(csv_path, index=False)
        
        # Upload para Google Drive
        file = drive.CreateFile({'title': filename, 'parents': [{'id': folder_id}]})
        file.SetContentFile(csv_path)
        file.Upload()
        
        st.success(f"✅ Guardado: {filename}")
        return file['id']
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao guardar: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# LIST RESULTS (com fallback)
# ════════════════════════════════════════════════════════════════════════════════

def list_results_drive(folder_name="SQLite"):
    """Lista ficheiros no Drive. Retorna lista vazia se falhar."""
    
    if not PYDRIVE_AVAILABLE:
        return []
    
    try:
        drive = get_drive_connection()
        if drive is None:
            return []
        
        folder_list = drive.ListFile({'q': f"title='{folder_name}' and trashed=false"}).GetList()
        
        if not folder_list:
            return []
        
        folder_id = folder_list[0]['id']
        
        # Listar ficheiros na pasta
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        
        results = []
        for file in file_list:
            if file['title'].endswith('.csv'):
                results.append({
                    'id': file['id'],
                    'title': file['title'],
                    'createdDate': file['createdDate']
                })
        
        return results
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao listar: {e}")
        return []

# ════════════════════════════════════════════════════════════════════════════════
# DOWNLOAD (com fallback)
# ════════════════════════════════════════════════════════════════════════════════

def download_resultado_drive(file_id):
    """Download ficheiro do Drive. Retorna None se falhar."""
    
    if not PYDRIVE_AVAILABLE:
        st.warning("⚠️ PyDrive não disponível.")
        return None
    
    try:
        drive = get_drive_connection()
        if drive is None:
            return None
        
        file = drive.CreateFile({'id': file_id})
        file.FetchMetadata()
        
        file_path = f"/tmp/{file['title']}"
        file.GetContentFile(file_path)
        
        df = pd.read_csv(file_path)
        st.success(f"✅ Carregado: {file['title']}")
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao carregar: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# FALLBACK FUNCTIONS (se PyDrive não disponível)
# ════════════════════════════════════════════════════════════════════════════════

def get_drive_connection_offline():
    """Versão offline (retorna None)"""
    return None

def upload_resultado_drive_offline(dataframe, filename, folder_name="SQLite"):
    """Versão offline (retorna None)"""
    st.info("ℹ️ Drive storage desactivado. Instala PyDrive para activar.")
    return None

def list_results_drive_offline(folder_name="SQLite"):
    """Versão offline (retorna lista vazia)"""
    return []

def download_resultado_drive_offline(file_id):
    """Versão offline (retorna None)"""
    return None

# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

if not PYDRIVE_AVAILABLE:
    # Se PyDrive não está disponível, usar fallback
    get_drive_connection = get_drive_connection_offline
    upload_resultado_drive = upload_resultado_drive_offline
    list_results_drive = list_results_drive_offline
    download_resultado_drive = download_resultado_drive_offline

__all__ = [
    'get_drive_connection',
    'upload_resultado_drive',
    'list_results_drive',
    'download_resultado_drive'
]
