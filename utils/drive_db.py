# ══════════════════════════════════════════════════════════════════════════════
# utils/drive_db.py — ATHELTICA
# Persistência via SQLite no Google Drive
# Usa a mesma service account do Google Sheets já configurada
# ══════════════════════════════════════════════════════════════════════════════

import os
import io
import sqlite3
import streamlit as st
from datetime import datetime

# Google API
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

# ── Configuração ───────────────────────────────────────────────────────────────
_DB_FILENAME  = "atheltica_cpmodel.db"
_LOCAL_PATH   = f"/tmp/{_DB_FILENAME}"
_SCOPES       = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

# ── Autenticação (reutiliza service account do Sheets) ────────────────────────
@st.cache_resource
def _get_drive_service():
    """Cria e cacheia o cliente Google Drive."""
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=_SCOPES,
        )
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        st.error(f"[drive_db] Erro de autenticação: {e}")
        return None


def _get_folder_id() -> str:
    """Lê o folder_id dos Streamlit Secrets."""
    try:
        return st.secrets["drive"]["folder_id"]
    except KeyError:
        st.error("[drive_db] folder_id não encontrado em st.secrets['drive'].")
        return ""


# ── Download ──────────────────────────────────────────────────────────────────
def _find_file_id(service, folder_id: str) -> str | None:
    """Procura o .db na pasta do Drive. Retorna o file_id ou None."""
    try:
        q = (f"name='{_DB_FILENAME}' "
             f"and '{folder_id}' in parents "
             f"and trashed=false")
        res = service.files().list(q=q, fields="files(id,name)").execute()
        files = res.get("files", [])
        return files[0]["id"] if files else None
    except HttpError as e:
        st.error(f"[drive_db] Erro ao procurar ficheiro: {e}")
        return None


def download_db() -> bool:
    """
    Descarrega o .db do Drive para /tmp/.
    Se não existir, cria um novo .db local.
    Retorna True se OK.
    """
    service   = _get_drive_service()
    folder_id = _get_folder_id()
    if not service or not folder_id:
        return False

    file_id = _find_file_id(service, folder_id)

    if file_id:
        try:
            request = service.files().get_media(fileId=file_id)
            with open(_LOCAL_PATH, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            return True
        except HttpError as e:
            st.error(f"[drive_db] Erro ao descarregar: {e}")
            return False
    else:
        # Primeira vez — criar .db local vazio
        _init_db()
        return True


# ── Upload ────────────────────────────────────────────────────────────────────
def upload_db() -> bool:
    """
    Faz upload do .db local para o Drive.
    Actualiza se já existir, cria se não existir.
    Retorna True se OK.
    """
    if not os.path.exists(_LOCAL_PATH):
        st.warning("[drive_db] Ficheiro local não encontrado.")
        return False

    service   = _get_drive_service()
    folder_id = _get_folder_id()
    if not service or not folder_id:
        return False

    file_id = _find_file_id(service, folder_id)
    media   = MediaFileUpload(_LOCAL_PATH, mimetype="application/x-sqlite3")

    try:
        if file_id:
            service.files().update(
                fileId=file_id, media_body=media
            ).execute()
        else:
            service.files().create(
                body={"name": _DB_FILENAME, "parents": [folder_id]},
                media_body=media,
            ).execute()
        return True
    except HttpError as e:
        st.error(f"[drive_db] Erro ao fazer upload: {e}")
        return False


# ── Inicializar schema ────────────────────────────────────────────────────────
def _init_db():
    """Cria as tabelas se não existirem."""
    conn = sqlite3.connect(_LOCAL_PATH)
    cur  = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS cp_results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at    TEXT NOT NULL,          -- timestamp do save
        modalidade  TEXT NOT NULL,          -- Bike/Row/Ski/Run
        modelo      TEXT,                   -- M1/M2/M3
        cp_watts    REAL,
        wp_joules   REAL,
        see_pct     REAL,
        pontos      TEXT,                   -- "1min=378W,5min=302W,12min=182W"
        mmp1        REAL,
        mmp3        REAL,
        mmp5        REAL,
        mmp12       REAL,
        mmp20       REAL,
        pmax        REAL
    );

    CREATE TABLE IF NOT EXISTS metab_results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at    TEXT NOT NULL,
        modalidade  TEXT NOT NULL,
        vo2max      REAL,
        vlamax      REAL,
        mlss_w      REAL,
        fatmax_w    REAL,
        lt1_w       REAL,
        lt2_w       REAL,
        fat_fatmax  REAL,                   -- g/h de gordura no FatMax
        glycogen_g  REAL,
        perfil      TEXT                    -- Endurance puro / Speed etc.
    );

    CREATE TABLE IF NOT EXISTS hr_thresholds (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at    TEXT NOT NULL,
        modalidade  TEXT NOT NULL,
        hrvt1_bpm   REAL,
        hrvt1plus_bpm REAL,
        hrvtmss_bpm REAL,
        hrvt2_bpm   REAL,
        aethr_bpm   REAL,
        pbp_w       REAL,
        pvo2max_w   REAL
    );
    """)

    conn.commit()
    conn.close()


# ── API pública ────────────────────────────────────────────────────────────────
def get_connection() -> sqlite3.Connection | None:
    """
    Retorna uma conexão SQLite ao .db local.
    Faz download automático se o ficheiro não existir.
    """
    if not os.path.exists(_LOCAL_PATH):
        ok = download_db()
        if not ok:
            return None
    _init_db()  # garante que as tabelas existem
    return sqlite3.connect(_LOCAL_PATH)


def save_cp_result(
    modalidade: str,
    modelo: str,
    cp_watts: float,
    wp_joules: float,
    see_pct: float,
    combo_pts: list,          # lista de (watts, secs)
    mmp_dict: dict = None,    # {"MMP1": 378, "MMP3": 302, ...}
    pmax: float = None,
) -> bool:
    """Guarda resultado de CP na DB e faz upload para o Drive."""
    conn = get_connection()
    if not conn: return False

    pontos_str = ",".join([f"{int(t//60)}min={p:.0f}W" for p, t in combo_pts])
    mmp = mmp_dict or {}

    try:
        conn.execute("""
            INSERT INTO cp_results
            (saved_at, modalidade, modelo, cp_watts, wp_joules, see_pct,
             pontos, mmp1, mmp3, mmp5, mmp12, mmp20, pmax)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(),
            modalidade, modelo,
            round(cp_watts, 2), round(wp_joules, 1), round(see_pct, 3),
            pontos_str,
            mmp.get("MMP1"), mmp.get("MMP3"), mmp.get("MMP5"),
            mmp.get("MMP12"), mmp.get("MMP20"),
            pmax,
        ))
        conn.commit()
        conn.close()
        return upload_db()
    except Exception as e:
        st.error(f"[drive_db] Erro ao guardar CP: {e}")
        conn.close()
        return False


def save_metab_result(
    modalidade: str,
    vo2max: float,
    vlamax: float,
    mlss_w: float,
    fatmax_w: float,
    lt1_w: float = None,
    lt2_w: float = None,
    fat_fatmax: float = None,
    glycogen_g: float = None,
    perfil: str = None,
) -> bool:
    """Guarda resultado metabólico na DB e faz upload."""
    conn = get_connection()
    if not conn: return False

    try:
        conn.execute("""
            INSERT INTO metab_results
            (saved_at, modalidade, vo2max, vlamax, mlss_w, fatmax_w,
             lt1_w, lt2_w, fat_fatmax, glycogen_g, perfil)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(),
            modalidade,
            round(vo2max, 2), round(vlamax, 4),
            round(mlss_w, 1), round(fatmax_w, 1),
            round(lt1_w, 1) if lt1_w else None,
            round(lt2_w, 1) if lt2_w else None,
            round(fat_fatmax, 1) if fat_fatmax else None,
            round(glycogen_g, 0) if glycogen_g else None,
            perfil,
        ))
        conn.commit()
        conn.close()
        return upload_db()
    except Exception as e:
        st.error(f"[drive_db] Erro ao guardar metab: {e}")
        conn.close()
        return False


def load_cp_history(modalidade: str = None, n: int = 20) -> list[dict]:
    """
    Lê histórico de CP da DB.
    Retorna lista de dicts ordenada do mais recente.
    """
    conn = get_connection()
    if not conn: return []

    q = "SELECT * FROM cp_results"
    params = []
    if modalidade:
        q += " WHERE modalidade = ?"
        params.append(modalidade)
    q += " ORDER BY saved_at DESC LIMIT ?"
    params.append(n)

    try:
        cur = conn.execute(q, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        st.error(f"[drive_db] Erro ao ler CP: {e}")
        conn.close()
        return []


def load_metab_history(modalidade: str = None, n: int = 20) -> list[dict]:
    """Lê histórico metabólico da DB."""
    conn = get_connection()
    if not conn: return []

    q = "SELECT * FROM metab_results"
    params = []
    if modalidade:
        q += " WHERE modalidade = ?"
        params.append(modalidade)
    q += " ORDER BY saved_at DESC LIMIT ?"
    params.append(n)

    try:
        cur = conn.execute(q, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        st.error(f"[drive_db] Erro ao ler metab: {e}")
        conn.close()
        return []


def save_hr_thresholds(
    modalidade: str,
    hr_zones: dict,   # dict com 'HRVT1', 'HRVTMSS', etc. cada com 'med'
    pbp_w: float = None,
    pvo2max_w: float = None,
) -> bool:
    """Guarda limiares HR por modalidade."""
    conn = get_connection()
    if not conn: return False

    def _get(key): return hr_zones.get(key, {}).get('med')

    try:
        conn.execute("""
            INSERT INTO hr_thresholds
            (saved_at, modalidade, hrvt1_bpm, hrvt1plus_bpm, hrvtmss_bpm,
             hrvt2_bpm, aethr_bpm, pbp_w, pvo2max_w)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(), modalidade,
            _get('HRVT1'), _get('HRVT1PLUS'), _get('HRVTMSS'),
            _get('HRVT2'), _get('AeTHR'),
            pbp_w, pvo2max_w,
        ))
        conn.commit()
        conn.close()
        return upload_db()
    except Exception as e:
        st.error(f"[drive_db] Erro ao guardar HR: {e}")
        conn.close()
        return False
